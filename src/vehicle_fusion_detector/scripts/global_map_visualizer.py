#!/usr/bin/env python

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import threading

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rospy
from matplotlib.patches import Rectangle
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from demoros2.msg import FusionResult

# 尝试导入cv_bridge，如果失败则使用替代方案
try:
    from cv_bridge import CvBridge

    CV_BRIDGE_AVAILABLE = True
except ImportError:
    rospy.logwarn("cv_bridge not available, using manual conversion")
    CV_BRIDGE_AVAILABLE = False


class GlobalMapVisualizer:
    def __init__(self):
        rospy.init_node("global_map_visualizer", anonymous=True)

        # 参数配置
        self.figure_width = rospy.get_param("~figure_width", 20)
        self.figure_height = rospy.get_param("~figure_height", 16)
        self.global_map_range = rospy.get_param(
            "~global_map_range", 100.0
        )  # 全局地图范围（米）
        self.dpi = rospy.get_param("~dpi", 100)

        # 保存配置
        self.save_images = rospy.get_param("~save_images", True)
        self.save_path = rospy.get_param(
            "~save_path", "/home/huyaowen/catkin_ws/global_visualization_output"
        )
        self.save_interval = rospy.get_param("~save_interval", 2.0)  # 保存间隔（秒）

        # 同步配置
        self.sync_timeout = rospy.get_param("~sync_timeout", 5.0)  # 同步超时时间（秒）
        self.max_time_diff = rospy.get_param("~max_time_diff", 2.0)  # 最大时间差（秒）

        # 创建保存目录
        if self.save_images:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            rospy.loginfo(
                f"Global visualization images will be saved to: {self.save_path}"
            )

        # CV Bridge (如果可用)
        if CV_BRIDGE_AVAILABLE:
            self.bridge = CvBridge()
        else:
            self.bridge = None

        # 发布器
        self.vis_pub = rospy.Publisher("global_map_visualization", Image, queue_size=10)

        # 数据存储：4个象限的融合结果
        self.quadrant_data = {
            "Q1": None,  # 第一象限 (x>0, y>0)
            "Q2": None,  # 第二象限 (x<0, y>0)
            "Q3": None,  # 第三象限 (x<0, y<0)
            "Q4": None,  # 第四象限 (x>0, y<0)
        }

        # 数据接收时间戳
        self.quadrant_timestamps = {
            "Q1": rospy.Time(0),
            "Q2": rospy.Time(0),
            "Q3": rospy.Time(0),
            "Q4": rospy.Time(0),
        }

        # 线程锁
        self.data_lock = threading.Lock()

        # 图像保存计数器
        self.image_counter = 0
        self.last_save_time = rospy.Time.now()

        # 订阅器 - 订阅4个象限的融合结果
        rospy.Subscriber(
            "quadrant_1/fusion_results",
            FusionResult,
            lambda msg: self.fusion_result_callback(msg, "Q1"),
        )
        rospy.Subscriber(
            "quadrant_2/fusion_results",
            FusionResult,
            lambda msg: self.fusion_result_callback(msg, "Q2"),
        )
        rospy.Subscriber(
            "quadrant_3/fusion_results",
            FusionResult,
            lambda msg: self.fusion_result_callback(msg, "Q3"),
        )
        rospy.Subscriber(
            "quadrant_4/fusion_results",
            FusionResult,
            lambda msg: self.fusion_result_callback(msg, "Q4"),
        )

        # 颜色定义
        self.colors = {
            "Q1": "#FF4444",  # 红色 - 第一象限
            "Q2": "#44FF44",  # 绿色 - 第二象限
            "Q3": "#4444FF",  # 蓝色 - 第三象限
            "Q4": "#FFAA00",  # 橙色 - 第四象限
            "background": "#F8F8F8",  # 浅灰色背景
        }

        # 定时器 - 定期生成可视化
        self.vis_timer = rospy.Timer(rospy.Duration(1.0), self.generate_visualization)

        rospy.loginfo("Global Map Visualizer initialized")
        rospy.loginfo(f"Global map range: {self.global_map_range}m")
        rospy.loginfo(f"Sync timeout: {self.sync_timeout}s")

    def fusion_result_callback(self, msg, quadrant):
        """接收象限融合结果消息"""
        with self.data_lock:
            self.quadrant_data[quadrant] = msg
            self.quadrant_timestamps[quadrant] = rospy.Time.now()

            rospy.logdebug(
                f"Received fusion result from {quadrant}: "
                f"{len(msg.own_detections_with_odom.detections)} own, "
                f"{len(msg.other_detections_list)} other vehicles, "
                f"{len(msg.fused_detections.detections)} fused"
            )

    def get_synchronized_data(self):
        """获取同步的数据"""
        with self.data_lock:
            current_time = rospy.Time.now()
            valid_data = {}

            # 检查每个象限的数据是否在同步时间窗口内
            for quadrant in ["Q1", "Q2", "Q3", "Q4"]:
                if self.quadrant_data[quadrant] is not None:
                    time_diff = (
                        current_time - self.quadrant_timestamps[quadrant]
                    ).to_sec()
                    if time_diff <= self.sync_timeout:
                        valid_data[quadrant] = self.quadrant_data[quadrant]
                    else:
                        rospy.logdebug(f"{quadrant} data too old ({time_diff:.1f}s)")

            return valid_data

    def transform_coordinates_to_global(self, detections, quadrant):
        """将象限坐标转换为全局坐标"""
        transformed_detections = []

        for detection in detections:
            # 复制检测对象
            transformed_det = type(detection)()
            transformed_det.object_id = detection.object_id
            transformed_det.type = detection.type
            transformed_det.confidence = detection.confidence
            transformed_det.box_2d = list(detection.box_2d)

            # 转换3D位置坐标
            x, y, z = detection.position.x, detection.position.y, detection.position.z

            # 根据象限进行坐标变换
            if quadrant == "Q1":  # 第一象限 (x>0, y>0)
                global_x = x + self.global_map_range / 4
                global_y = y + self.global_map_range / 4
            elif quadrant == "Q2":  # 第二象限 (x<0, y>0)
                global_x = x - self.global_map_range / 4
                global_y = y + self.global_map_range / 4
            elif quadrant == "Q3":  # 第三象限 (x<0, y<0)
                global_x = x - self.global_map_range / 4
                global_y = y - self.global_map_range / 4
            elif quadrant == "Q4":  # 第四象限 (x>0, y<0)
                global_x = x + self.global_map_range / 4
                global_y = y - self.global_map_range / 4
            else:
                global_x, global_y = x, y

            from geometry_msgs.msg import Point

            transformed_det.position = Point(x=global_x, y=global_y, z=z)

            # 转换3D边界框坐标
            if len(detection.bbox_3d) >= 6:
                xmin, ymin, zmin, xmax, ymax, zmax = detection.bbox_3d

                # 变换边界框坐标
                if quadrant == "Q1":
                    global_xmin = xmin + self.global_map_range / 4
                    global_ymin = ymin + self.global_map_range / 4
                    global_xmax = xmax + self.global_map_range / 4
                    global_ymax = ymax + self.global_map_range / 4
                elif quadrant == "Q2":
                    global_xmin = xmin - self.global_map_range / 4
                    global_ymin = ymin + self.global_map_range / 4
                    global_xmax = xmax - self.global_map_range / 4
                    global_ymax = ymax + self.global_map_range / 4
                elif quadrant == "Q3":
                    global_xmin = xmin - self.global_map_range / 4
                    global_ymin = ymin - self.global_map_range / 4
                    global_xmax = xmax - self.global_map_range / 4
                    global_ymax = ymax - self.global_map_range / 4
                elif quadrant == "Q4":
                    global_xmin = xmin + self.global_map_range / 4
                    global_ymin = ymin - self.global_map_range / 4
                    global_xmax = xmax + self.global_map_range / 4
                    global_ymax = ymax - self.global_map_range / 4
                else:
                    global_xmin, global_ymin = xmin, ymin
                    global_xmax, global_ymax = xmax, ymax

                transformed_det.bbox_3d = [
                    global_xmin,
                    global_ymin,
                    zmin,
                    global_xmax,
                    global_ymax,
                    zmax,
                ]
            else:
                transformed_det.bbox_3d = list(detection.bbox_3d)

            transformed_detections.append(transformed_det)

        return transformed_detections

    def get_unmatched_detections_for_quadrant(self, fusion_result):
        """获取某个象限未匹配的检测"""
        # 构建匹配关系映射
        match_results = {}
        for match_result in fusion_result.match_results:
            vehicle_id = match_result.other_vehicle_id
            for i in range(len(match_result.own_indices)):
                own_idx = match_result.own_indices[i]
                other_idx = match_result.other_indices[i]
                match_results[(own_idx, vehicle_id)] = other_idx

        # 获取未匹配的own检测
        own_detections = fusion_result.own_detections_with_odom.detections
        matched_own_indices = set(own_idx for (own_idx, _), _ in match_results.items())
        unmatched_own = [
            det for i, det in enumerate(own_detections) if i not in matched_own_indices
        ]

        # 获取未匹配的other检测
        unmatched_other = []
        for other_detection_msg in fusion_result.other_detections_list:
            vehicle_id = other_detection_msg.car_id
            detections = other_detection_msg.detections

            # 收集该车辆所有被匹配的索引
            matched_indices = set()
            for (own_idx, v_id), other_idx in match_results.items():
                if v_id == vehicle_id:
                    matched_indices.add(other_idx)

            # 找出未匹配的检测
            for i, detection in enumerate(detections):
                if i not in matched_indices:
                    unmatched_other.append(detection)

        return unmatched_own, unmatched_other

    def draw_detection_3d_box(self, ax, detection, color, label_prefix="", alpha=0.7):
        """绘制3D检测框"""
        if len(detection.bbox_3d) < 6:
            return

        xmin, ymin, zmin, xmax, ymax, zmax = detection.bbox_3d
        width = xmax - xmin
        height = ymax - ymin

        # 绘制矩形框
        rect = Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
        )
        ax.add_patch(rect)

        # 绘制中心点
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ax.plot(
            center_x,
            center_y,
            "o",
            color=color,
            markersize=4,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # 添加标签
        label = f"{label_prefix}{detection.object_id}"
        ax.text(
            center_x,
            center_y + height / 2 + 2,
            label,
            fontsize=6,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.1", facecolor=color, alpha=0.8),
        )

    def setup_global_map_plot(self, ax):
        """设置全局地图绘图属性"""
        ax.set_xlim(-self.global_map_range / 2, self.global_map_range / 2)
        ax.set_ylim(-self.global_map_range / 2, self.global_map_range / 2)
        ax.set_xlabel("X (meters)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y (meters)", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(
            "Global Map - Multi-Quadrant Fusion Results",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # 绘制坐标轴
        ax.axhline(y=0, color="black", linewidth=2, alpha=0.8)
        ax.axvline(x=0, color="black", linewidth=2, alpha=0.8)

        # 绘制象限分界线
        ax.axhline(y=0, color="gray", linewidth=1, alpha=0.5, linestyle="--")
        ax.axvline(x=0, color="gray", linewidth=1, alpha=0.5, linestyle="--")

        # 标注象限
        ax.text(
            self.global_map_range / 4,
            self.global_map_range / 4,
            "Q1",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors["Q1"], alpha=0.3),
        )
        ax.text(
            -self.global_map_range / 4,
            self.global_map_range / 4,
            "Q2",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors["Q2"], alpha=0.3),
        )
        ax.text(
            -self.global_map_range / 4,
            -self.global_map_range / 4,
            "Q3",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors["Q3"], alpha=0.3),
        )
        ax.text(
            self.global_map_range / 4,
            -self.global_map_range / 4,
            "Q4",
            fontsize=16,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors["Q4"], alpha=0.3),
        )

    def generate_visualization(self, event=None):
        """生成全局地图可视化"""
        # 获取同步数据
        valid_data = self.get_synchronized_data()

        if not valid_data:
            rospy.logdebug("No valid synchronized data available")
            return

        rospy.logdebug(
            f"Generating global visualization with data from {list(valid_data.keys())}"
        )

        # 创建图形
        fig, ax = plt.subplots(
            figsize=(self.figure_width, self.figure_height), dpi=self.dpi
        )
        fig.patch.set_facecolor(self.colors["background"])

        # 设置全局地图
        self.setup_global_map_plot(ax)

        # 统计信息
        total_fused = 0
        total_unmatched_own = 0
        total_unmatched_other = 0
        legend_elements = []

        # 处理每个象限的数据
        for quadrant, fusion_result in valid_data.items():
            quadrant_color = self.colors[quadrant]

            # 获取未匹配的检测和融合检测
            unmatched_own, unmatched_other = self.get_unmatched_detections_for_quadrant(
                fusion_result
            )
            fused_detections = fusion_result.fused_detections.detections

            # 转换坐标到全局坐标系
            global_unmatched_own = self.transform_coordinates_to_global(
                unmatched_own, quadrant
            )
            global_unmatched_other = self.transform_coordinates_to_global(
                unmatched_other, quadrant
            )
            global_fused = self.transform_coordinates_to_global(
                fused_detections, quadrant
            )

            # 绘制未匹配的own检测
            for detection in global_unmatched_own:
                self.draw_detection_3d_box(
                    ax, detection, quadrant_color, f"{quadrant}_Own_", alpha=0.5
                )

            # 绘制未匹配的other检测
            for detection in global_unmatched_other:
                self.draw_detection_3d_box(
                    ax, detection, quadrant_color, f"{quadrant}_Other_", alpha=0.4
                )

            # 绘制融合检测
            for detection in global_fused:
                self.draw_detection_3d_box(
                    ax, detection, quadrant_color, f"{quadrant}_Fused_", alpha=0.8
                )

            # 更新统计
            total_fused += len(global_fused)
            total_unmatched_own += len(global_unmatched_own)
            total_unmatched_other += len(global_unmatched_other)

            # 添加图例
            quadrant_total = (
                len(global_fused)
                + len(global_unmatched_own)
                + len(global_unmatched_other)
            )
            if quadrant_total > 0:
                legend_elements.append(
                    patches.Patch(
                        color=quadrant_color,
                        alpha=0.7,
                        label=f"{quadrant}: {quadrant_total} detections "
                        f"(F:{len(global_fused)}, O:{len(global_unmatched_own)}, Ot:{len(global_unmatched_other)})",
                    )
                )

        # 添加图例
        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                fontsize=10,
                bbox_to_anchor=(0.02, 0.98),
            )

        # 添加总体统计信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_detections = total_fused + total_unmatched_own + total_unmatched_other

        stats_text = (
            f"Global Statistics [{timestamp}]\n"
            f"Active Quadrants: {len(valid_data)} | "
            f"Total Detections: {total_detections}\n"
            f"Fused: {total_fused} | "
            f"Unmatched Own: {total_unmatched_own} | "
            f"Unmatched Others: {total_unmatched_other}"
        )

        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # 调整布局
        plt.tight_layout()

        # 转换为OpenCV图像格式
        cv_image = self.matplotlib_to_cv2(fig)

        # 保存图像文件
        self.save_visualization_image(cv_image)

        # 发布可视化图像
        try:
            if CV_BRIDGE_AVAILABLE and self.bridge:
                img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            else:
                img_msg = self.cv2_to_imgmsg_manual(cv_image, "bgr8")

            img_msg.header = Header(stamp=rospy.Time.now(), frame_id="global_map")
            self.vis_pub.publish(img_msg)

            rospy.loginfo_throttle(
                5,
                f"Published global visualization: "
                f"{len(valid_data)} quadrants, {total_detections} detections",
            )
        except Exception as e:
            rospy.logwarn(f"Failed to publish global visualization: {e}")

        # 清理matplotlib资源
        plt.close(fig)

    def matplotlib_to_cv2(self, fig):
        """将matplotlib图形转换为OpenCV图像"""
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_bgr = buf[:, :, ::-1].copy()
        return img_bgr

    def cv2_to_imgmsg_manual(self, cv_image, encoding="bgr8"):
        """手动将OpenCV图像转换为ROS Image消息"""
        img_msg = Image()
        img_msg.header = Header(stamp=rospy.Time.now(), frame_id="global_map")
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        img_msg.data = cv_image.tobytes()
        return img_msg

    def save_visualization_image(self, img):
        """保存可视化图像到文件"""
        if not self.save_images:
            return

        current_time = rospy.Time.now()
        if (current_time - self.last_save_time).to_sec() < self.save_interval:
            return

        self.last_save_time = current_time

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"global_map_vis_{timestamp}_{self.image_counter:04d}.jpg"
        filepath = os.path.join(self.save_path, filename)

        try:
            import cv2

            cv2.imwrite(filepath, img)
            self.image_counter += 1
            rospy.loginfo_throttle(10, f"Saved global visualization to: {filename}")
        except Exception as e:
            rospy.logwarn(f"Failed to save global image: {e}")


if __name__ == "__main__":
    try:
        visualizer = GlobalMapVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Global Map Visualizer shutdown.")
