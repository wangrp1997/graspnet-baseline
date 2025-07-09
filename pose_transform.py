import numpy as np

# 示例数据，单位：米
translation = np.array([0.02030951, -0.12040397,  0.723])
rotation_matrix = np.array([
    [0.02277477, -0.9989326,  -0.04018656],
    [0.49252543,  0.04619145, -0.86907136],
    [0.87,        0.0,        0.4930517 ]
])

H = np.load('eye-to-hand-transform.npy')
# 只对H的平移部分做单位转换
if np.max(np.abs(H[:3, 3])) > 10:  # 如果大于10，说明是毫米
    H[:3, 3] = H[:3, 3] / 1000

translation_homo = np.append(translation, 1)
translation_robot = H @ translation_homo
print('机械臂基坐标系下的位置（米）:', translation_robot[:3])

R_robot = H[:3, :3] @ rotation_matrix
print('机械臂基坐标系下的旋转矩阵:\n', R_robot)

try:
    from scipy.spatial.transform import Rotation as R
    r = R.from_matrix(R_robot)
    quat = r.as_quat()  # [x, y, z, w]
    print('机械臂基坐标系下的四元数 [x, y, z, w]:', quat)
except ImportError:
    print('如需四元数输出，请安装 scipy：pip install scipy')
