#!/usr/bin/env python
import rospy
from demoros2.msg import DetectionsWithOdom  # 替换为实际的包名
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

def transform_detections_to_global(detections_with_odom):
    """
    将 DetectionsWithOdom 消息中的所有检测目标的 position 字段从传感器坐标系转换到全局坐标系。

    参数：
        detections_with_odom: DetectionsWithOdom 类型的消息。

    返回：
        一个形状为 (N, 3) 的 NumPy 数组，每行对应一个目标在全局坐标系下的 (x, y, z) 位置。
    """
    # 提取小车在全局坐标系下的位置和姿态
    odom_pose = detections_with_odom.odom.pose.pose
    car_position = np.array([
        odom_pose.position.x,
        odom_pose.position.y,
        odom_pose.position.z
    ])
    car_orientation = np.array([
        odom_pose.orientation.x,
        odom_pose.orientation.y,
        odom_pose.orientation.z,
        odom_pose.orientation.w
    ])

    # 构建从 base_link 到 odom 的变换矩阵
    T_base_to_odom = tft.quaternion_matrix(car_orientation)
    T_base_to_odom[0:3, 3] = car_position

    # 如果传感器相对于 base_link 有已知的固定变换，定义如下
    # 例如，传感器在 x 方向前方 0.5 米，无旋转
    sensor_translation = np.array([0.5, 0.0, 0.0])  # 根据实际情况修改
    sensor_rotation = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数，表示无旋转

    # 构建从 sensor_frame 到 base_link 的变换矩阵
    T_sensor_to_base = tft.quaternion_matrix(sensor_rotation)
    T_sensor_to_base[0:3, 3] = sensor_translation

    # 构建从 sensor_frame 到 odom 的总变换矩阵
    T_sensor_to_odom = np.dot(T_base_to_odom, T_sensor_to_base)

    # 初始化结果数组
    positions_global = []

    # 遍历所有检测目标
    for detection in detections_with_odom.detections:
        # 提取目标在传感器坐标系下的位置
        local_pos = np.array([
            detection.position.x,
            detection.position.y,
            detection.position.z,
            1.0  # 齐次坐标
        ])
        # 将位置转换到全局坐标系
        global_pos_homogeneous = np.dot(T_sensor_to_odom, local_pos)
        global_pos = global_pos_homogeneous[0:3]
        positions_global.append(global_pos)

    return np.array(positions_global)

def callback(data):
    positions = transform_detections_to_global(data)
    print("全局坐标系下的目标位置：")
    print(positions)

def listener():
    rospy.init_node('detection_listener', anonymous=True)
    rospy.Subscriber('/detection', DetectionsWithOdom, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()