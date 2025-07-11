""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_ros import TransformListener, Buffer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    # obj_mask = np.array(Image.open(os.path.join(data_dir, 'real_test_bbox_mask_1bit.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    N = min(100, len(gg))  # 可视化前N个
    for i in range(N):
        best_gg = gg[i:i+1]
        print("translation in camera coordinate:", best_gg.translations)
        print("rotation_matrix in camera coordinate:\n", best_gg.rotation_matrices)
        grippers = best_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        
        user_input = input(f"是否选择抓取点 {i+1}？(y=选择，n=下一个，q=退出): ")
        if user_input.lower() == 'y':
            print(f"你选择了抓取点 {i+1}，即将发布...")
            break
        elif user_input.lower() == 'q':
            print("已退出选择。")
            break

    # === 新增：eye-to-hand变换（与YOLO代码一致） ===
    try:
        # 使用验证函数加载H矩阵
        T_eye_to_base = load_and_assert_transform_matrix('eye-to-hand-transform.npy')
        
        # 单个抓取点
        translation = best_gg.translations[0]  # 相机坐标系下的位置
        rotation_matrix = best_gg.rotation_matrices[0]  # 相机坐标系下的旋转矩阵
        
        # === 坐标系修正（在eye-to-hand变换之前） ===
        # [0,0,1;1,0,0;0,1,0]
        correction_matrix = np.array([
            [0, 0, 1],  # GraspNet y轴 → 相机x轴
            [1, 0, 0],  # GraspNet z轴 → 相机y轴
            [0, 1, 0]   # GraspNet x轴 → 相机z轴
        ])
        A = np.diag([-1, -1, 1])  # 末端姿态修正矩阵
        # 对旋转矩阵进行修正
        rotation_matrix_corrected =  rotation_matrix@correction_matrix

        # 2. 构建完整的齐次变换矩阵
        grasp_transform = np.eye(4)
        grasp_transform[:3, :3] = rotation_matrix_corrected  # 旋转部分
        grasp_transform[:3, 3] = translation  # 平移部分
        
        # if np.max(np.abs(T_eye_to_base[:3, 3])) > 10:  # 如果大于10，说明是毫米
        T_eye_to_base[:3, 3] = T_eye_to_base[:3, 3] / 1000
        # zuo乘H，得到baselink坐标系下的变换矩阵
        robot_transform = T_eye_to_base @ grasp_transform        
        # 提取位置和旋转
        R_robot = robot_transform[:3, :3]           # 旋转部分
        translation_robot = robot_transform[:3, 3]  # 平移部分

        tool_offset = np.array([0, 0, 0.28])  # 工具长度
        tool_offset_in_base = R_robot @ tool_offset
        tcp_target_position = translation_robot - tool_offset_in_base
        print('机械臂基坐标系下的位置修正之qian（米）:', translation_robot)

        translation_robot = A@ tcp_target_position[:3]  # 只取前3个坐标

        R_robot_ = A@ R_robot[:3, :3] # 只取前3行3列


        print('机械臂基坐标系下的位置修正之后（米）:', translation_robot)
        print('机械臂基坐标系下的旋转矩阵:\n', R_robot)
        
        try:
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(R_robot_)
            quat_robot = r.as_quat()  # [x, y, z, w]
            print('机械臂基坐标系下的四元数 [x, y, z, w]:', quat_robot)
        except ImportError:
            print('如需四元数输出，请安装 scipy：pip install scipy')
        
    except Exception as e:
        print('eye-to-hand变换失败:', e)
    # === 变换结束 ===
    # translation_robot[2] += 0.05  # 只取前3个坐标
    publish_pose_continuously(translation_robot[:3], quat_robot)
    # publish_pose_continuously(translation_robot[:3])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)


class GraspPosePublisher(Node):
    def __init__(self, ee_target_position, ee_target_quat, frame_id='base_link', topic='/ee_target_pose', freq=10):
        super().__init__('grasp_pose_publisher')
        self.publisher_ = self.create_publisher(PoseStamped, topic, 10)
        self.ee_target_position = ee_target_position  # 末端目标点
        self.ee_target_quat = ee_target_quat  # 末端目标姿态
        self.frame_id = frame_id
        self.tool_offset = np.array([0, 0, 0.28])  # 工具长度
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0/freq, self.timer_callback)

    def timer_callback(self):
        try:
            # 获取当前末端到基座的变换
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'wrist_3_link', rclpy.time.Time())
            # 当前末端四元数
            q = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            from scipy.spatial.transform import Rotation as R
            R_current = R.from_quat(q).as_matrix()

            R_target = R.from_quat(self.ee_target_quat).as_matrix()


            # === 打印欧拉角对比 ===
            # 当前末端欧拉角
            euler_current = R.from_quat(q).as_euler('xyz', degrees=True)
            print('当前末端欧拉角(度):', euler_current)

            # 规划目标欧拉角
            euler_target = R.from_quat(self.ee_target_quat).as_euler('xyz', degrees=True)
            print('规划目标欧拉角(度):', euler_target)
            # 角度差异
            angle_diff = euler_target - euler_current
            print('角度差异(度):', angle_diff)
            print('---')
            # === 打印结束 ===
            
            # 用当前末端姿态做补偿，反推TCP目标点
            # tool_offset_in_base = R_current @ self.tool_offset
            # tcp_target_position = self.ee_target_position - tool_offset_in_base
            # print('规划的目标位置(m):', tcp_target_position)
            msg = PoseStamped()
            msg.header.frame_id = self.frame_id
            msg.header.stamp = self.get_clock().now().to_msg()
            # 1. 对位置做x、y取反
            msg.pose.position.x = float(self.ee_target_position[0])
            msg.pose.position.y = float(self.ee_target_position[1])
            msg.pose.position.z = float(self.ee_target_position[2])

            msg.pose.orientation.x = self.ee_target_quat[0]
            msg.pose.orientation.y = self.ee_target_quat[1]
            msg.pose.orientation.z = self.ee_target_quat[2]
            msg.pose.orientation.w = self.ee_target_quat[3]
            # msg.pose.orientation.x = q[0]
            # msg.pose.orientation.y = q[1]
            # msg.pose.orientation.z = q[2]
            # msg.pose.orientation.w = q[3]

            self.publisher_.publish(msg)
            self.get_logger().info('发布目标位姿（末端目标点+目标姿态）...')
        except Exception as e:
            self.get_logger().warn(f'{e}')

# 调用时只需要传位置
def publish_pose_continuously(ee_target_position, ee_target_quat):
    rclpy.init()
    node = GraspPosePublisher(ee_target_position, ee_target_quat)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

def load_and_assert_transform_matrix(npy_file: str) -> np.ndarray:
    """加载并验证变换矩阵H"""
    matrix = np.load(npy_file)
    assert matrix.shape == (4, 4), f"Invalid matrix shape {matrix.shape}"
    assert np.allclose(matrix[3, :], [0, 0, 0, 1]), "Invalid last row"
    return matrix

if __name__=='__main__':
    data_dir = 'orbbec/my_data'
    demo(data_dir)