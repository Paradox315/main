#!/usr/bin/env python

import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
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


class DetectionVisualizer:
    def __init__(self):
        rospy.init_node("detection_visualizer", anonymous=True)

        # 参数配置
        self.figure_width = rospy.get_param("~figure_width", 16)
        self.figure_height = rospy.get_param("~figure_height", 12)
        self.plot_range = rospy.get_param("~plot_range", 50.0)
        self.dpi = rospy.get_param("~dpi", 100)

        # 新增参数：是否保存图像文件
        self.save_images = rospy.get_param("~save_images", True)
        self.save_path = rospy.get_param(
            "~save_path", "/home/huyaowen/catkin_ws/visualization_output"
        )
        self.save_interval = rospy.get_param("~save_interval", 1.0)  # 保存间隔（秒）

        # 新增参数：融合结果检查
        self.require_fused_results = rospy.get_param(
            "~require_fused_results", True
        )  # 是否要求融合结果不为空
        self.wait_for_fusion_timeout = rospy.get_param(
            "~wait_for_fusion_timeout", 5.0
        )  # 等待融合结果的超时时间（秒）

        # 新增参数：子图3显示模式配置
        self.subplot3_mode = rospy.get_param(
            "~subplot3_mode", "union_filtered"
        )  # "fused_only" 或 "union_filtered"

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

        # 数据存储 - 修改为从融合结果获取数据
        self.current_fusion_result = None
        self.own_detections = []
        self.other_detections = {}  # {vehicle_id: detections}
        self.fused_detections = []
        self.match_results = {}  # {(own_idx, vehicle_id): (other_idx, score)}

        # 融合结果状态跟踪
        self.last_fused_update_time = rospy.Time.now()
        self.has_received_fused_data = False

        # 图像保存计数器
        self.image_counter = 0
        self.last_save_time = rospy.Time.now()

        # 订阅器 - 只订阅融合结果
        rospy.Subscriber("fusion_results", FusionResult, self.fusion_result_callback)

        # 颜色定义 - 使用matplotlib颜色格式
        self.colors = {
            "own": "#00AA00",  # 深绿色 - 自车检测
            "fused": "#0080FF",  # 蓝色 - 融合后的检测
            "background": "#F8F8F8",  # 浅灰色背景
        }

        # 为不同车辆ID分配不同颜色 - 更丰富的颜色搭配
        self.vehicle_colors = [
            "#FF4444",  # 红色
            "#FF8800",  # 橙色
            "#8800FF",  # 紫色
            "#FF4488",  # 粉红色
            "#BB4400",  # 棕色
            "#44FF44",  # 亮绿色
            "#4444FF",  # 蓝色
            "#FFAA00",  # 黄橙色
        ]

        # 定时器 - 定期生成可视化
        self.vis_timer = rospy.Timer(rospy.Duration(0.2), self.generate_visualization)

        rospy.loginfo("Detection Visualizer initialized with 2x2 layout")
        rospy.loginfo(f"Save images: {self.save_images}")
        rospy.loginfo(f"Figure size: {self.figure_width}x{self.figure_height}")
        rospy.loginfo(f"Subplot3 mode: {self.subplot3_mode}")

    def fusion_result_callback(self, msg):
        """接收融合结果消息"""
        self.current_fusion_result = msg

        # 更新数据 - 从DetectionsWithOdom中提取检测数据
        self.own_detections = list(msg.own_detections_with_odom.detections)

        # 重建other_detections字典
        self.other_detections = {}
        for other_detection_msg in msg.other_detections_list:
            vehicle_id = other_detection_msg.car_id
            self.other_detections[vehicle_id] = list(other_detection_msg.detections)

        self.fused_detections = list(msg.fused_detections.detections)

        # 重建匹配结果信息
        self.match_results = {}
        for match_result in msg.match_results:
            vehicle_id = match_result.other_vehicle_id
            for i in range(len(match_result.own_indices)):
                own_idx = match_result.own_indices[i]
                other_idx = match_result.other_indices[i]
                score = match_result.match_scores[i]
                self.match_results[(own_idx, vehicle_id)] = (other_idx, score)

        # 更新融合状态
        self.last_fused_update_time = rospy.Time.now()
        self.has_received_fused_data = True

        rospy.logdebug(
            f"Received fusion result: {len(self.own_detections)} own, "
            f"{len(self.other_detections)} other vehicles, "
            f"{len(self.match_results)} match results, "
            f"{len(self.fused_detections)} fused"
        )

    def setup_subplot(self, ax, title, plot_range=None):
        """设置子图的基本属性"""
        if plot_range is None:
            plot_range = self.plot_range

        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)
        ax.set_xlabel("X (meters)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Y (meters)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # 绘制坐标轴
        ax.axhline(y=0, color="black", linewidth=1, alpha=0.5)
        ax.axvline(x=0, color="black", linewidth=1, alpha=0.5)

    def draw_detection_3d_box(
        self,
        ax,
        detection,
        color,
        label_prefix="",
        alpha=0.7,
        show_label=True,
        linewidth=2,
    ):
        """
        从俯视角度绘制3D检测框
        bbox_3d格式: [xmin, ymin, zmin, xmax, ymax, zmax]
        """
        if len(detection.bbox_3d) < 6:
            rospy.logwarn(f"Invalid bbox_3d format for detection {detection.object_id}")
            return

        # 提取3D边界框坐标
        xmin, ymin, zmin, xmax, ymax, zmax = detection.bbox_3d

        # 计算长宽（俯视图只关心x和y）
        width = xmax - xmin
        height = ymax - ymin

        # 绘制矩形框（俯视图）
        rect = Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=linewidth,  # 使用传入的线宽
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
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=1,
        )

        # 添加标签（根据需要）
        if show_label:
            label = f"{label_prefix}{detection.object_id}\n{detection.type}\n{detection.confidence:.2f}"
            ax.text(
                center_x,
                center_y + height / 2 + 2,
                label,
                fontsize=7,
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7),
            )

    def get_vehicle_color(self, vehicle_id):
        """为不同车辆分配颜色"""
        vehicle_list = sorted(self.other_detections.keys())
        if vehicle_id in vehicle_list:
            idx = vehicle_list.index(vehicle_id)
            return self.vehicle_colors[idx % len(self.vehicle_colors)]
        return self.vehicle_colors[0]

    def draw_match_lines_between_subplots(self, fig, ax1, ax2):
        """在子图1和子图2之间绘制匹配连线"""
        for (own_idx, vehicle_id), (other_idx, score) in self.match_results.items():
            if (
                own_idx < len(self.own_detections)
                and vehicle_id in self.other_detections
                and other_idx < len(self.other_detections[vehicle_id])
            ):
                own_det = self.own_detections[own_idx]
                other_det = self.other_detections[vehicle_id][other_idx]

                # 计算在各自子图中的坐标
                own_x, own_y = self._get_detection_center_in_subplot(own_det)
                other_x, other_y = self._get_detection_center_in_subplot(other_det)

                # 转换为图形坐标系
                own_fig_x, own_fig_y = self._subplot_to_figure_coords(ax1, own_x, own_y)
                other_fig_x, other_fig_y = self._subplot_to_figure_coords(
                    ax2, other_x, other_y
                )

                # 获取车辆颜色
                vehicle_color = self.get_vehicle_color(vehicle_id)

                # 根据得分调整线条透明度
                alpha = min(0.8, max(0.3, score))

                # 在整个图形上绘制连线
                line = plt.Line2D(
                    [own_fig_x, other_fig_x],
                    [own_fig_y, other_fig_y],
                    color=vehicle_color,
                    linewidth=2,
                    alpha=alpha,
                    linestyle="--",
                    transform=fig.transFigure,
                )
                fig.lines.append(line)

    def _get_detection_center_in_subplot(self, detection):
        """获取检测框在子图坐标系中的中心点"""
        if len(detection.bbox_3d) >= 6:
            center_x = (detection.bbox_3d[0] + detection.bbox_3d[3]) / 2
            center_y = (detection.bbox_3d[1] + detection.bbox_3d[4]) / 2
            return center_x, center_y
        else:
            # 备选方案：使用position
            return detection.position.x, detection.position.y

    def _subplot_to_figure_coords(self, ax, x, y):
        """将子图坐标转换为图形坐标"""
        # 将数据坐标转换为轴坐标
        ax_x, ax_y = ax.transData.transform((x, y))
        # 将轴坐标转换为图形坐标
        fig_x, fig_y = ax.figure.transFigure.inverted().transform((ax_x, ax_y))
        return fig_x, fig_y

    def draw_own_detections_subplot(self, ax):
        """绘制子图1：Own车检测结果"""
        self.setup_subplot(ax, f"Own Vehicle Detections ({len(self.own_detections)})")

        # 绘制自车检测结果
        for i, detection in enumerate(self.own_detections):
            # 检查这个检测是否被匹配了
            is_matched = any(
                own_idx == i for (own_idx, _), _ in self.match_results.items()
            )

            # 根据是否匹配使用不同的透明度和边框
            alpha = 0.8 if is_matched else 0.6
            linewidth = 3 if is_matched else 2

            self.draw_detection_3d_box(
                ax,
                detection,
                self.colors["own"],
                "Own_",
                alpha=alpha,
                show_label=True,
                linewidth=linewidth,
            )

        # 添加图例
        matched_count = len(
            set(own_idx for (own_idx, _), _ in self.match_results.items())
        )
        legend_elements = [
            patches.Patch(
                color=self.colors["own"],
                alpha=0.7,
                label=f"Own Detections ({len(self.own_detections)}, {matched_count} matched)",
            )
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    def draw_other_detections_subplot(self, ax):
        """绘制子图2：Other车检测结果"""
        total_other = sum(len(dets) for dets in self.other_detections.values())
        self.setup_subplot(ax, f"Other Vehicles Detections ({total_other})")

        # 绘制其他车辆检测结果
        legend_elements = []
        for vehicle_id, detections in self.other_detections.items():
            vehicle_color = self.get_vehicle_color(vehicle_id)

            # 计算该车辆被匹配的检测数量
            matched_other_indices = set()
            for (own_idx, v_id), (other_idx, score) in self.match_results.items():
                if v_id == vehicle_id:
                    matched_other_indices.add(other_idx)

            for i, detection in enumerate(detections):
                # 检查这个检测是否被匹配了
                is_matched = i in matched_other_indices

                # 根据是否匹配使用不同的透明度和边框
                alpha = 0.8 if is_matched else 0.5
                linewidth = 3 if is_matched else 2

                self.draw_detection_3d_box(
                    ax,
                    detection,
                    vehicle_color,
                    f"{vehicle_id}_",
                    alpha=alpha,
                    show_label=True,
                    linewidth=linewidth,
                )

            # 为每个车辆添加图例项
            matched_count = len(matched_other_indices)
            legend_elements.append(
                patches.Patch(
                    color=vehicle_color,
                    alpha=0.6,
                    label=f"{vehicle_id} ({len(detections)}, {matched_count} matched)",
                )
            )

        # 添加图例
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    def draw_fused_detections_subplot(self, ax):
        """绘制子图3：根据配置选择显示内容"""
        if self.subplot3_mode == "fused_only":
            self._draw_fused_only_mode(ax)
        elif self.subplot3_mode == "union_filtered":
            self._draw_union_filtered_mode(ax)
        else:
            rospy.logwarn(
                f"Unknown subplot3_mode: {self.subplot3_mode}, using fused_only"
            )
            self._draw_fused_only_mode(ax)

    def _draw_fused_only_mode(self, ax):
        """模式1：仅显示融合后的框"""
        title = f"Fused Detections Only ({len(self.fused_detections)})"

        # 如果融合结果为空，在标题中标注
        if len(self.fused_detections) == 0:
            title += " - No Fusion Results"

        self.setup_subplot(ax, title)

        # 绘制融合后的检测结果
        for fused_det in self.fused_detections:
            self.draw_detection_3d_box(
                ax,
                fused_det,
                self.colors["fused"],
                "Fused_",
                alpha=0.8,
                show_label=True,
            )

        # 添加图例
        if len(self.fused_detections) > 0:
            legend_elements = [
                patches.Patch(
                    color=self.colors["fused"],
                    alpha=0.8,
                    label=f"Fused Detections ({len(self.fused_detections)})",
                )
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        else:
            # 如果没有融合结果，显示等待信息
            ax.text(
                0.5,
                0.5,
                "Waiting for\nFusion Results...",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
            )

    def _draw_union_filtered_mode(self, ax):
        """模式2：显示所有框的并集（删除已融合的原始框，保留融合框和未匹配框）"""
        # 获取未匹配的检测
        unmatched_own, unmatched_other = self.get_unmatched_detections()

        # 计算总数
        total_unmatched_own = len(unmatched_own)
        total_unmatched_other = sum(len(dets) for dets in unmatched_other.values())
        total_fused = len(self.fused_detections)
        total_displayed = total_unmatched_own + total_unmatched_other + total_fused

        title = f"Union View: {total_displayed} Detections (Own:{total_unmatched_own} + Others:{total_unmatched_other} + Fused:{total_fused})"
        self.setup_subplot(ax, title)

        # 1. 绘制未匹配的own检测
        for detection in unmatched_own:
            self.draw_detection_3d_box(
                ax, detection, self.colors["own"], "Own_", alpha=0.6, show_label=True
            )

        # 2. 绘制未匹配的other检测
        for vehicle_id, detections in unmatched_other.items():
            vehicle_color = self.get_vehicle_color(vehicle_id)
            for detection in detections:
                self.draw_detection_3d_box(
                    ax,
                    detection,
                    vehicle_color,
                    f"{vehicle_id}_",
                    alpha=0.5,
                    show_label=True,
                )

        # 3. 绘制融合后的检测结果
        for fused_det in self.fused_detections:
            self.draw_detection_3d_box(
                ax,
                fused_det,
                self.colors["fused"],
                "Fused_",
                alpha=0.8,
                show_label=True,
            )

        # 添加图例
        legend_elements = []

        # 未匹配own检测图例
        if total_unmatched_own > 0:
            legend_elements.append(
                patches.Patch(
                    color=self.colors["own"],
                    alpha=0.6,
                    label=f"Unmatched Own ({total_unmatched_own})",
                )
            )

        # 未匹配other检测图例
        for vehicle_id, detections in unmatched_other.items():
            if detections:
                vehicle_color = self.get_vehicle_color(vehicle_id)
                legend_elements.append(
                    patches.Patch(
                        color=vehicle_color,
                        alpha=0.5,
                        label=f"Unmatched {vehicle_id} ({len(detections)})",
                    )
                )

        # 融合检测图例
        if total_fused > 0:
            legend_elements.append(
                patches.Patch(
                    color=self.colors["fused"],
                    alpha=0.8,
                    label=f"Fused ({total_fused})",
                )
            )

        # 显示图例
        if legend_elements:
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        # 如果没有任何检测结果，显示提示信息
        if total_displayed == 0:
            ax.text(
                0.5,
                0.5,
                "No Detections\nAvailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="gray",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

    def matplotlib_to_cv2(self, fig):
        """将matplotlib图形转换为OpenCV图像"""
        # 将图形保存到内存中的字节流
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 转换RGB到BGR（OpenCV格式）
        img_bgr = buf[:, :, ::-1].copy()
        return img_bgr

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
            import cv2

            cv2.imwrite(filepath, img)
            self.image_counter += 1
            rospy.loginfo_throttle(5, f"Saved visualization to: {filename}")
        except Exception as e:
            rospy.logwarn(f"Failed to save image: {e}")

    def should_generate_visualization(self):
        """检查是否应该生成可视化"""
        # 如果不要求融合结果，直接允许绘制
        if not self.require_fused_results:
            return True, "Fusion results not required"

        # 检查是否收到融合结果
        if self.current_fusion_result is None:
            return False, "No fusion result received"

        # 检查是否有基本数据
        if not self.own_detections and not any(self.other_detections.values()):
            return False, "No detection data available"

        # 如果要求融合结果不为空
        if len(self.fused_detections) == 0:
            return False, "Fusion results list is empty"

        return True, "All conditions met"

    def generate_visualization(self, event=None):
        """生成可视化图像 - 2x2布局：上行(Own, Other)，下行(Fused跨列)"""
        # 检查是否应该生成可视化
        should_generate, reason = self.should_generate_visualization()

        if not should_generate:
            rospy.logdebug_throttle(2, f"Skipping visualization: {reason}")
            return

        rospy.logdebug(f"Generating visualization: {reason}")

        # 创建2x2子图布局
        fig = plt.figure(figsize=(self.figure_width, self.figure_height), dpi=self.dpi)

        # 创建子图：上行两个，下行一个跨列
        ax1 = plt.subplot2grid((2, 2), (0, 0))  # Own detections
        ax2 = plt.subplot2grid((2, 2), (0, 1))  # Other detections
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)  # Fused detections (跨两列)

        # 设置整体背景色
        fig.patch.set_facecolor(self.colors["background"])

        # 绘制三个子图
        self.draw_own_detections_subplot(ax1)
        self.draw_other_detections_subplot(ax2)
        self.draw_fused_detections_subplot(ax3)

        # 在子图1和子图2之间绘制匹配连线
        self.draw_match_lines_between_subplots(fig, ax1, ax2)

        # 添加整体标题
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_text = (
            "Fused Only" if self.subplot3_mode == "fused_only" else "Union Filtered"
        )
        fig.suptitle(
            f"Vehicle Detection Fusion System - Bird's Eye View [{mode_text}]\n{timestamp}",
            fontsize=14,
            fontweight="bold",
            y=0.95,
        )

        # 添加统计信息
        total_other = sum(len(dets) for dets in self.other_detections.values())
        total_match_results = len(self.match_results)

        # 添加融合状态信息
        fusion_status = "Active" if len(self.fused_detections) > 0 else "Inactive"
        last_update = (rospy.Time.now() - self.last_fused_update_time).to_sec()

        # 为union_filtered模式添加额外统计信息
        extra_stats = ""
        if self.subplot3_mode == "union_filtered":
            unmatched_own, unmatched_other = self.get_unmatched_detections()
            total_unmatched = len(unmatched_own) + sum(
                len(dets) for dets in unmatched_other.values()
            )
            extra_stats = f" | Unmatched: {total_unmatched}"

        stats_text = (
            f"System Statistics: Own: {len(self.own_detections)} | "
            f"Others: {total_other} | "
            f"Fused: {len(self.fused_detections)} | "
            f"Match Results: {total_match_results} | "
            f"Vehicles: {len(self.other_detections)} | "
            f"Fusion: {fusion_status} (Updated {last_update:.1f}s ago)"
            f"{extra_stats}"
        )

        fig.text(
            0.5,
            0.02,
            stats_text,
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # 调整子图间距
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])

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

            img_msg.header = Header(stamp=rospy.Time.now(), frame_id="visualization")
            self.vis_pub.publish(img_msg)

            rospy.loginfo_throttle(
                5,
                f"Published visualization with {len(self.fused_detections)} fused detections, {total_match_results} match results (mode: {self.subplot3_mode})",
            )
        except Exception as e:
            rospy.logwarn(f"Failed to publish visualization: {e}")

        # 清理matplotlib资源
        plt.close(fig)

    def get_matched_detection_indices(self):
        """获取所有已匹配的检测索引"""
        matched_own_indices = set()
        matched_other_indices = {}  # {vehicle_id: set(indices)}

        for vehicle_id, matches in self.current_matches.items():
            if vehicle_id not in matched_other_indices:
                matched_other_indices[vehicle_id] = set()

            for own_idx, other_idx in matches:
                matched_own_indices.add(own_idx)
                matched_other_indices[vehicle_id].add(other_idx)

        return matched_own_indices, matched_other_indices

    def get_unmatched_detections(self):
        """获取所有未匹配的检测"""
        # 获取未匹配的own检测
        matched_own_indices = set(
            own_idx for (own_idx, _), _ in self.match_results.items()
        )
        unmatched_own = []
        for i, detection in enumerate(self.own_detections):
            if i not in matched_own_indices:
                unmatched_own.append(detection)

        # 获取未匹配的other检测
        unmatched_other = {}
        for vehicle_id, detections in self.other_detections.items():
            # 收集该车辆所有被匹配的索引
            matched_indices = set()
            for (own_idx, v_id), (other_idx, score) in self.match_results.items():
                if v_id == vehicle_id:
                    matched_indices.add(other_idx)

            # 找出未匹配的检测
            unmatched_detections = []
            for i, detection in enumerate(detections):
                if i not in matched_indices:
                    unmatched_detections.append(detection)

            if unmatched_detections:
                unmatched_other[vehicle_id] = unmatched_detections

        return unmatched_own, unmatched_other


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
