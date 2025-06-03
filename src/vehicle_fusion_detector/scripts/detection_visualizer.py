#!/usr/bin/env python

import os
from datetime import datetime

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from demoros2.msg import Detections, DetectionsWithOdom

# 尝试导入cv_bridge，如果失败则使用替代方案
try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except ImportError:
    rospy.logwarn("cv_bridge not available, using manual conversion")
    CV_BRIDGE_AVAILABLE = False


class DetectionVisualizer:
    def __init__(self):
        rospy.init_node("detection_visualizer", anonymous=True)

        # 参数配置
        self.image_width = rospy.get_param("~image_width", 1200)
        self.image_height = rospy.get_param("~image_height", 800)
        self.scale_factor = rospy.get_param("~scale_factor", 15.0)  # 坐标缩放因子

        # 新增参数：是否保存图像文件
        self.save_images = rospy.get_param("~save_images", True)
        self.save_path = rospy.get_param(
            "~save_path", "/home/huyaowen/catkin_ws/visualization_output"
        )
        self.save_interval = rospy.get_param("~save_interval", 1.0)  # 保存间隔（秒）

        # 创建保存目录
        if self.save_images:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            rospy.loginfo(f"Visualization images will be saved to: {self.save_path}")

        # CV Bridge (如果可用)
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

        # 发布器
        self.vis_pub = rospy.Publisher("detection_visualization", Image, queue_size=10)

        # 数据存储
        self.own_detections = []
        self.other_detections = {}  # {vehicle_id: detections}
        self.fused_detections = []
        self.current_matches = {}  # {vehicle_id: [(own_idx, other_idx), ...]}

        # 图像保存计数器
        self.image_counter = 0
        self.last_save_time = rospy.Time.now()

        # 订阅器
        rospy.Subscriber(
            "own_sensor_detections", DetectionsWithOdom, self.own_detections_callback
        )
        rospy.Subscriber(
            "other_vehicle_detections",
            DetectionsWithOdom,
            self.other_detections_callback,
        )
        rospy.Subscriber(
            "fused_detections", DetectionsWithOdom, self.fused_detections_callback
        )

        # 颜色定义
        self.colors = {
            "own": (0, 255, 0),  # 绿色 - 自车检测
            "other": (0, 0, 255),  # 红色 - 他车检测
            "matched": (255, 0, 255),  # 紫色 - 匹配的检测
            "fused": (0, 255, 255),  # 青色 - 融合后的检测
            "text": (255, 255, 255),  # 白色 - 文字
            "background": (50, 50, 50),  # 深灰色 - 背景
        }

        # 定时器 - 定期生成可视化
        self.vis_timer = rospy.Timer(rospy.Duration(0.2), self.generate_visualization)

        rospy.loginfo("Detection Visualizer initialized")
        rospy.loginfo(f"Save images: {self.save_images}")
        rospy.loginfo(f"Image size: {self.image_width}x{self.image_height}")

    def cv2_to_imgmsg_manual(self, cv_image, encoding="bgr8"):
        """手动将OpenCV图像转换为ROS Image消息（当cv_bridge不可用时）"""
        img_msg = Image()
        img_msg.header = Header(stamp=rospy.Time.now(), frame_id="visualization")
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        img_msg.data = cv_image.tobytes()
        return img_msg

    def own_detections_callback(self, msg):
        """接收自车检测结果"""
        self.own_detections = msg.detections
        rospy.logdebug(f"Received {len(self.own_detections)} own detections")

    def other_detections_callback(self, msg):
        """接收他车检测结果"""
        vehicle_id = msg.car_id
        self.other_detections[vehicle_id] = msg.detections
        rospy.logdebug(f"Received {len(msg.detections)} detections from {vehicle_id}")

    def fused_detections_callback(self, msg):
        """接收融合后的检测结果"""
        self.fused_detections = msg.detections
        rospy.logdebug(f"Received {len(self.fused_detections)} fused detections")

    def update_matches(self, vehicle_id, matches):
        """更新匹配结果（由fusion_node调用）"""
        self.current_matches[vehicle_id] = matches

    def world_to_image_coords(self, x, y):
        """将世界坐标转换为图像坐标"""
        img_x = int(x * self.scale_factor + self.image_width // 2)
        img_y = int(
            self.image_height - (y * self.scale_factor + self.image_height // 2)
        )
        return max(0, min(img_x, self.image_width - 1)), max(
            0, min(img_y, self.image_height - 1)
        )

    def draw_detection_box(self, img, detection, color, label_prefix=""):
        """绘制检测框"""
        # 使用3D位置作为中心点
        center_x, center_y = self.world_to_image_coords(
            detection.position.x, detection.position.y
        )

        # 使用2D边界框大小计算显示框大小
        if len(detection.box_2d) >= 4:
            box_width = max(20, abs(detection.box_2d[2] - detection.box_2d[0]) // 8)
            box_height = max(15, abs(detection.box_2d[3] - detection.box_2d[1]) // 8)
        else:
            box_width, box_height = 30, 20

        # 绘制矩形框
        top_left = (center_x - box_width // 2, center_y - box_height // 2)
        bottom_right = (center_x + box_width // 2, center_y + box_height // 2)
        cv2.rectangle(img, top_left, bottom_right, color, 2)

        # 绘制中心点
        cv2.circle(img, (center_x, center_y), 4, color, -1)

        # 绘制标签
        label = f"{label_prefix}{detection.object_id}({detection.type}:{detection.confidence:.2f})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]

        # 确保标签在图像范围内
        label_x = max(0, min(top_left[0], self.image_width - label_size[0] - 5))
        label_y = max(label_size[1] + 5, top_left[1])

        cv2.rectangle(
            img,
            (label_x, label_y - label_size[1] - 5),
            (label_x + label_size[0] + 5, label_y),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (label_x + 2, label_y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
        )

    def save_visualization_image(self, img):
        """保存可视化图像到文件"""
        if not self.save_images:
            return

        current_time = rospy.Time.now()
        if (current_time - self.last_save_time).to_sec() < self.save_interval:
            return

        self.last_save_time = current_time

        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_vis_{timestamp}_{self.image_counter:04d}.jpg"
        filepath = os.path.join(self.save_path, filename)

        # 保存图像
        try:
            cv2.imwrite(filepath, img)
            self.image_counter += 1
            rospy.loginfo_throttle(5, f"Saved visualization to: {filename}")
        except Exception as e:
            rospy.logwarn(f"Failed to save image: {e}")

    def draw_match_line(self, img, own_det, other_det, vehicle_id):
        """绘制匹配连线"""
        own_x, own_y = self.world_to_image_coords(
            own_det.position.x, own_det.position.y
        )
        other_x, other_y = self.world_to_image_coords(
            other_det.position.x, other_det.position.y
        )

        # 绘制连线
        cv2.line(img, (own_x, own_y), (other_x, other_y), self.colors["matched"], 2)

        # 在连线中点绘制匹配标识
        mid_x, mid_y = (own_x + other_x) // 2, (own_y + other_y) // 2
        cv2.circle(img, (mid_x, mid_y), 6, self.colors["matched"], -1)
        cv2.putText(
            img,
            f"M",
            (mid_x + 8, mid_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            self.colors["text"],
            1,
        )

    def generate_visualization(self, event=None):
        """生成可视化图像"""
        # 创建空白图像
        img = np.full(
            (self.image_height, self.image_width, 3),
            self.colors["background"],
            dtype=np.uint8,
        )

        # 绘制坐标轴和网格
        self.draw_grid(img)

        # 绘制自车检测结果
        for i, detection in enumerate(self.own_detections):
            self.draw_detection_box(img, detection, self.colors["own"], "Own_")

        # 绘制他车检测结果
        for vehicle_id, detections in self.other_detections.items():
            for detection in detections:
                self.draw_detection_box(
                    img, detection, self.colors["other"], f"{vehicle_id}_"
                )

        # 绘制匹配连线
        for vehicle_id, matches in self.current_matches.items():
            if vehicle_id in self.other_detections:
                other_dets = self.other_detections[vehicle_id]
                for own_idx, other_idx in matches:
                    if own_idx < len(self.own_detections) and other_idx < len(
                        other_dets
                    ):
                        own_det = self.own_detections[own_idx]
                        other_det = other_dets[other_idx]
                        self.draw_match_line(img, own_det, other_det, vehicle_id)

        # 绘制融合后的检测结果
        for fused_det in self.fused_detections:
            self.draw_detection_box(img, fused_det, self.colors["fused"], "Fused_")

        # 添加图例
        self.draw_legend(img)

        # 添加统计信息
        self.draw_statistics(img)

        # 添加时间戳
        self.draw_timestamp(img)

        # 保存图像文件
        self.save_visualization_image(img)

        # 发布可视化图像
        try:
            if CV_BRIDGE_AVAILABLE and self.bridge:
                img_msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
            else:
                img_msg = self.cv2_to_imgmsg_manual(img, "bgr8")

            img_msg.header = Header(stamp=rospy.Time.now(), frame_id="visualization")
            self.vis_pub.publish(img_msg)
        except Exception as e:
            rospy.logwarn(f"Failed to publish visualization: {e}")

    def draw_timestamp(self, img):
        """绘制时间戳"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            img,
            timestamp,
            (self.image_width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.colors["text"],
            1,
        )

    def draw_grid(self, img):
        """绘制坐标轴和网格"""
        center_x, center_y = self.image_width // 2, self.image_height // 2

        # 绘制主坐标轴
        cv2.line(img, (0, center_y), (self.image_width, center_y), (120, 120, 120), 2)
        cv2.line(img, (center_x, 0), (center_x, self.image_height), (120, 120, 120), 2)

        # 绘制网格
        grid_spacing = int(self.scale_factor * 2)  # 每2米一个网格
        for x in range(center_x, self.image_width, grid_spacing):
            cv2.line(img, (x, 0), (x, self.image_height), (80, 80, 80), 1)
        for x in range(center_x, 0, -grid_spacing):
            cv2.line(img, (x, 0), (x, self.image_height), (80, 80, 80), 1)
        for y in range(center_y, self.image_height, grid_spacing):
            cv2.line(img, (0, y), (self.image_width, y), (80, 80, 80), 1)
        for y in range(center_y, 0, -grid_spacing):
            cv2.line(img, (0, y), (self.image_width, y), (80, 80, 80), 1)

        # 添加坐标标签
        cv2.putText(
            img,
            "X",
            (self.image_width - 30, center_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors["text"],
            2,
        )
        cv2.putText(
            img,
            "Y",
            (center_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.colors["text"],
            2,
        )

    def draw_legend(self, img):
        """绘制图例"""
        legend_items = [
            ("Own Detections", self.colors["own"]),
            ("Other Detections", self.colors["other"]),
            ("Match Lines", self.colors["matched"]),
            ("Fused Results", self.colors["fused"]),
        ]

        legend_x = 20
        legend_y = 40
        box_size = 20

        # 绘制图例背景
        legend_bg_width = 200
        legend_bg_height = len(legend_items) * 30 + 20
        cv2.rectangle(
            img,
            (legend_x - 10, legend_y - 20),
            (legend_x + legend_bg_width, legend_y + legend_bg_height),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            img,
            (legend_x - 10, legend_y - 20),
            (legend_x + legend_bg_width, legend_y + legend_bg_height),
            (150, 150, 150),
            2,
        )

        for i, (label, color) in enumerate(legend_items):
            y = legend_y + i * 30
            cv2.rectangle(
                img, (legend_x, y), (legend_x + box_size, y + box_size), color, -1
            )
            cv2.rectangle(
                img,
                (legend_x, y),
                (legend_x + box_size, y + box_size),
                (255, 255, 255),
                1,
            )
            cv2.putText(
                img,
                label,
                (legend_x + box_size + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["text"],
                1,
            )

    def draw_statistics(self, img):
        """绘制统计信息"""
        total_matches = sum(len(matches) for matches in self.current_matches.values())
        total_other = sum(len(dets) for dets in self.other_detections.values())

        stats = [
            f"Own Detections: {len(self.own_detections)}",
            f"Other Detections: {total_other}",
            f"Total Matches: {total_matches}",
            f"Fused Detections: {len(self.fused_detections)}",
            f"Other Vehicles: {len(self.other_detections)}",
        ]

        stats_x = 20
        stats_y = self.image_height - 120

        # 绘制统计信息背景
        stats_bg_width = 250
        stats_bg_height = len(stats) * 20 + 20
        cv2.rectangle(
            img,
            (stats_x - 10, stats_y - 20),
            (stats_x + stats_bg_width, stats_y + stats_bg_height),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            img,
            (stats_x - 10, stats_y - 20),
            (stats_x + stats_bg_width, stats_y + stats_bg_height),
            (150, 150, 150),
            2,
        )

        for i, stat in enumerate(stats):
            y = stats_y + i * 20
            cv2.putText(
                img,
                stat,
                (stats_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.colors["text"],
                1,
            )


# 全局可视化器实例
visualizer = None


def get_visualizer():
    """获取全局可视化器实例"""
    global visualizer
    if visualizer is None:
        visualizer = DetectionVisualizer()
    return visualizer


if __name__ == "__main__":
    try:
        visualizer = DetectionVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection Visualizer shutdown.")
