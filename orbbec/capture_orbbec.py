#!/usr/bin/env python3
"""
Orbbec 相机图像采集脚本

用于 GraspNet 数据准备，按空格键采集图像.
"""

import cv2
import numpy as np
import os
import argparse
from orbbec_cam import OrbbecRGBDCamera, IMAGE_WIDTH, IMAGE_HEIGHT
import scipy.io as scio
from PIL import Image

def create_output_dir(output_dir='captured_data'):
    """创建输出目录."""
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录: {output_dir}')
    return output_dir

def create_meta_file(output_dir, camera_params):
    """创建 meta.mat 文件."""
    meta_path = os.path.join(output_dir, 'meta.mat')
    scio.savemat(meta_path, camera_params)
    print(f'相机参数已保存: {meta_path}')
    print(f'  内参矩阵: {camera_params["intrinsic_matrix"].shape}')
    print(f'  深度因子: {camera_params["factor_depth"][0][0]}')

def get_camera_params(camera: OrbbecRGBDCamera):
    """从 Orbbec 相机对象获取内参和深度因子."""
    # 获取彩色相机内参
    K = np.array([
        [camera.color_intrinsics.fx, 0, camera.color_intrinsics.cx],
        [0, camera.color_intrinsics.fy, camera.color_intrinsics.cy],
        [0, 0, 1]
    ], dtype=np.float64)
    # 深度缩放因子（Orbbec 通常为 1000.0，单位为毫米）
    factor_depth = np.array([[1000.0]], dtype=np.float64)
    return {
        'intrinsic_matrix': K,
        'factor_depth': factor_depth
    }


def create_workspace_mask(image_shape, rect=None):
    """
    image_shape: (height, width)
    rect: (x, y, w, h)  # x, y 是左上角坐标
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    if rect is None:
        # 默认左下角 500x300 区域
        h, w = image_shape
        rect = (int(0.2*w), int(0.2*h), int(0.6*w), int(0.6*h))
    x, y, rw, rh = rect
    mask[y:y+rh, x:x+rw] = 255
    return mask


def capture_images(output_dir='captured_data', camera_params=None):
    """采集 RGB 和深度图像."""
    output_dir = create_output_dir(output_dir)
    print('初始化 Orbbec 相机...')
    try:
        camera = OrbbecRGBDCamera()
        camera.start()
        print('相机初始化成功!')
    except Exception as e:
        print(f'相机初始化失败: {e}')
        return False

    try:
        print('按空格键采集图像，按 ESC 退出...')
        while True:
            # 获取一帧数据
            try:
                color_img, depth_data = camera.get_rgbd_data()
            except Exception as e:
                print(f'读取图像失败: {e}')
                continue

            # 显示彩色图像
            cv2.imshow('Orbbec Color', cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

            # 显示深度图像（归一化显示）
            depth_display = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            cv2.imshow('Orbbec Depth', depth_display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                print('采集图像...')

                # 保存彩色图像
                rgb_path = os.path.join(output_dir, 'color.png')
                cv2.imwrite(rgb_path, cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
                print(f'RGB图像已保存: {rgb_path}')

                # 保存深度图像（16位PNG）
                depth_path = os.path.join(output_dir, 'depth.png')
                depth_to_save = depth_data.astype(np.uint16)
                cv2.imwrite(depth_path, depth_to_save)
                print(f'深度图像已保存: {depth_path}')

                # 创建全白工作空间掩码
                mask_path = os.path.join(output_dir, 'workspace_mask.png')
                mask = create_workspace_mask((480, 640))
                Image.fromarray(mask).convert('1').save(mask_path)
                print(f'工作空间掩码已保存: {mask_path}')

                # 保存相机参数
                create_meta_file(output_dir, camera_params)

                print('\n所有文件已保存到: {}'.format(output_dir))
                print('文件列表:')
                for file in os.listdir(output_dir):
                    file_path = os.path.join(output_dir, file)
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        print(f'  {file} ({size} bytes)')
                print('\n按空格键继续采集，按 ESC 退出...')

            elif key == 27:
                print('退出采集...')
                break

    except Exception as e:
        print(f'采集过程中出错: {e}')
        return False
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print('相机已释放')

def parse_args():
    parser = argparse.ArgumentParser(description='Orbbec 相机图像采集脚本，用于 GraspNet 数据准备')
    parser.add_argument('--output_dir', type=str, default='orbbec/my_data', help='输出目录名称 (默认: orbbec/my_data)')
    return parser.parse_args()

def main():
    args = parse_args()
    print('=== Orbbec 图像采集脚本 ===')
    print('用于 GraspNet 数据准备')
    print('按空格键采集图像，按 ESC 退出')
    print(f'输出目录: {args.output_dir}')
    print()

    # 获取相机参数
    print('正在读取相机内参...')
    try:
        temp_camera = OrbbecRGBDCamera()
        params = get_camera_params(temp_camera)
        temp_camera.stop()
        print('相机内参读取成功')
        print('内参矩阵:')
        print(params['intrinsic_matrix'])
        print('深度因子:', params['factor_depth'][0][0])
    except Exception as e:
        print(f'无法读取相机内参: {e}')
        return

    # 采集图像
    capture_images(args.output_dir, params)
    print('\n采集完成!')

if __name__ == '__main__':
    main()
