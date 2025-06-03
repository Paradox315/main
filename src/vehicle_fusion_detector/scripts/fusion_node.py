#!/usr/bin/env python
import collections  # 用于 deque
import json
import os
import sys  # 用于动态添加路径
from typing import Any, Dict, List, Optional

import message_filters  # 用于同步消息
import numpy as np
import requests
import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from demoros2.msg import Detections, DetectionsWithOdom  # 使用demoros2包中的消息

# 添加当前脚本目录到Python路径，以便导入graph_matching模块
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
GRAPH_MATCHING_AVAILABLE = True


# --- 配置参数 ---
IOU_THRESHOLD = 0.3
DISTANCE_THRESHOLD = 2.0  # 米
USE_IOU_FOR_COST = True
USE_GRAPH_MATCHING = True  # 新增：是否使用图匹配算法
GRAPH_MATCHING_THRESHOLD = 0.1  # 图匹配的最小亲和力阈值

# RPC 服务配置
RPC_SERVER_HOST = rospy.get_param("~rpc_host", "localhost")
RPC_SERVER_PORT = rospy.get_param("~rpc_port", 8000)
RPC_TIMEOUT = rospy.get_param("~rpc_timeout", 5.0)  # 秒

# ApproximateTimeSynchronizer slop: 允许消息之间最大时间差，以秒为单位
APPROXIMATE_SYNC_SLOP = 0.1  # 例如，0.1秒
# MAX_TIME_DIFFERENCE_FOR_MATCHING: 在deque中为特定车辆选择消息时，与own_msg的最大时间差
MAX_TIME_DIFFERENCE_FOR_MATCHING = rospy.Duration(0.2)

OTHER_DETECTIONS_BUFFER_SIZE = 5
BUFFER_CLEANUP_INTERVAL = rospy.Duration(5.0)

# 可选：预筛选门限
PREFILTER_MAX_DISTANCE_GATE = 10.0  # 米，粗略筛选检测对的最大距离


class DetectionFuser:
    def __init__(self):
        rospy.init_node("detection_fuser_node", anonymous=True)

        self.current_vehicle_id = rospy.get_param("~vehicle_id", "vehicle_A")

        # RPC 客户端配置
        self.rpc_base_url = f"http://{RPC_SERVER_HOST}:{RPC_SERVER_PORT}"
        self.rpc_session = requests.Session()
        self.rpc_session.timeout = RPC_TIMEOUT

        # 发布器 - 添加融合检测结果发布器
        self.fused_pub = rospy.Publisher(
            "fused_detections", DetectionsWithOdom, queue_size=10
        )

        self.other_detections_buffer = collections.defaultdict(
            lambda: collections.deque(maxlen=OTHER_DETECTIONS_BUFFER_SIZE)
        )

        # --- 使用 ApproximateTimeSynchronizer 同步消息 ---
        own_sensor_sub = message_filters.Subscriber(
            "own_sensor_detections", DetectionsWithOdom
        )
        other_vehicle_sub = message_filters.Subscriber(
            "other_vehicle_detections", DetectionsWithOdom
        )

        # 时间同步器，队列大小可以根据消息频率调整
        # slop 参数定义了消息时间戳之间可以接受的最大差异（秒）
        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [own_sensor_sub, other_vehicle_sub],
            queue_size=10,
            slop=APPROXIMATE_SYNC_SLOP,
        )
        # 注册同步回调
        self.time_synchronizer.registerCallback(self.synchronized_callback)

        self.cleanup_timer = rospy.Timer(
            BUFFER_CLEANUP_INTERVAL, self.cleanup_old_detections
        )

        rospy.loginfo(
            f"Detection Fuser Node Initialized for {self.current_vehicle_id}."
        )
        rospy.loginfo(f"RPC Server: {self.rpc_base_url}")
        rospy.loginfo(
            f"ApproximateTimeSynchronizer with slop: {APPROXIMATE_SYNC_SLOP}s"
        )

    def _detection_to_dict(self, detection) -> Dict[str, Any]:
        """将ROS检测消息转换为字典格式"""
        return {
            "object_id": detection.object_id,
            "type": detection.type,
            "confidence": detection.confidence,
            "box_2d": list(detection.box_2d),  # int32[4] -> List[int]
            "position": {
                "x": detection.position.x,
                "y": detection.position.y,
                "z": detection.position.z,
            },
            "bbox_3d": list(detection.bbox_3d),  # float32[6] -> List[float]
        }

    def _call_rpc_matching(
        self, endpoint: str, own_dets_list, other_dets_list, other_vehicle_id
    ) -> Optional[Dict[str, Any]]:
        """调用RPC匹配服务"""
        try:
            request_data = {
                "own_detections": [
                    self._detection_to_dict(det) for det in own_dets_list
                ],
                "other_detections": [
                    self._detection_to_dict(det) for det in other_dets_list
                ],
                "other_vehicle_id": other_vehicle_id,
                "use_iou_for_cost": USE_IOU_FOR_COST,
                "iou_threshold": IOU_THRESHOLD,
                "distance_threshold": DISTANCE_THRESHOLD,
                "prefilter_max_distance": PREFILTER_MAX_DISTANCE_GATE,
            }

            url = f"{self.rpc_base_url}/{endpoint}"
            response = self.rpc_session.post(
                url, json=request_data, headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                return response.json()
            else:
                rospy.logwarn(
                    f"RPC call to {endpoint} failed with status {response.status_code}: {response.text}"
                )
                return None

        except requests.exceptions.Timeout:
            rospy.logwarn(f"RPC call to {endpoint} timed out")
            return None
        except requests.exceptions.ConnectionError:
            rospy.logwarn(f"Failed to connect to RPC server at {self.rpc_base_url}")
            return None
        except Exception as e:
            rospy.logwarn(f"RPC call to {endpoint} failed: {str(e)}")
            return None

    def synchronized_callback(
        self, own_detections_msg, other_vehicle_single_source_msg
    ):
        """
        当 own_sensor_detections 和 other_vehicle_detections 消息近似同步到达时被调用。
        other_vehicle_single_source_msg 是来自某一个其他车辆的 DetectionsWithOdom。
        """
        rospy.logdebug(
            f"Synchronized callback triggered at {rospy.Time.now().to_sec()}: "
            f"Own Detections (stamp: {own_detections_msg.header.stamp.to_sec()}), "
            f"Other Vehicle Detections (stamp: {other_vehicle_single_source_msg.header.stamp.to_sec()})"
        )

        # 1. 处理和缓冲接收到的其他车辆检测数据
        if not other_vehicle_single_source_msg.detections:
            rospy.logdebug(
                "Received empty detections from another vehicle in synchronized callback."
            )
        else:
            source_id = other_vehicle_single_source_msg.car_id
            if not source_id or source_id == self.current_vehicle_id:
                rospy.logwarn_throttle(
                    10,
                    "Ignoring other_vehicle_detections from self or with no source_id in sync callback.",
                )
            else:
                rospy.logdebug(
                    f"Buffering {len(other_vehicle_single_source_msg.detections)} detections from {source_id} (sync)."
                )
                self.other_detections_buffer[source_id].append(
                    other_vehicle_single_source_msg
                )

        # 2. 尝试进行匹配
        if own_detections_msg.detections:  # 确保本车有检测结果
            self.try_match_detections(own_detections_msg)
        else:
            rospy.logdebug(
                "Own detections empty in synchronized callback, skipping matching."
            )

    def cleanup_old_detections(self, event=None):
        now = rospy.Time.now()
        cleanup_threshold_duration = MAX_TIME_DIFFERENCE_FOR_MATCHING + rospy.Duration(
            1.0
        )
        keys_to_delete_from_buffer = []
        for vehicle_id, deq in self.other_detections_buffer.items():
            while deq and (now - deq[0].header.stamp > cleanup_threshold_duration):
                rospy.logdebug(
                    f"Cleaning up old detection from {vehicle_id} (timestamp: {deq[0].header.stamp.to_sec()})"
                )
                deq.popleft()
            if not deq:
                keys_to_delete_from_buffer.append(vehicle_id)
        for key in keys_to_delete_from_buffer:
            if (
                key in self.other_detections_buffer
            ):  # 再次检查，防止并发问题（虽然Python GIL使其不太可能）
                del self.other_detections_buffer[key]
                rospy.loginfo(
                    f"Removed vehicle {key} from buffer due to inactivity or all data timed out."
                )

    def _calculate_iou(self, boxA_det, boxB_det):
        # 使用新的消息格式中的box_2d字段
        xA = max(boxA_det.box_2d[0], boxB_det.box_2d[0])  # x_min
        yA = max(boxA_det.box_2d[1], boxB_det.box_2d[1])  # y_min
        xB = min(boxA_det.box_2d[2], boxB_det.box_2d[2])  # x_max
        yB = min(boxA_det.box_2d[3], boxB_det.box_2d[3])  # y_max
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA_det.box_2d[2] - boxA_det.box_2d[0]) * (
            boxA_det.box_2d[3] - boxA_det.box_2d[1]
        )
        boxBArea = (boxB_det.box_2d[2] - boxB_det.box_2d[0]) * (
            boxB_det.box_2d[3] - boxB_det.box_2d[1]
        )
        if boxAArea <= 0 or boxBArea <= 0:
            return 0.0
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _calculate_center_distance(self, boxA_det, boxB_det):
        # 使用新的消息格式中的box_2d字段
        centerAx = (boxA_det.box_2d[0] + boxA_det.box_2d[2]) / 2.0
        centerAy = (boxA_det.box_2d[1] + boxA_det.box_2d[3]) / 2.0
        centerBx = (boxB_det.box_2d[0] + boxB_det.box_2d[2]) / 2.0
        centerBy = (boxB_det.box_2d[1] + boxB_det.box_2d[3]) / 2.0
        return np.sqrt((centerAx - centerBx) ** 2 + (centerAy - centerBy) ** 2)

    def try_match_detections(self, current_own_detections_msg):
        """
        使用传入的本车检测结果，与缓冲区中各其他车辆的检测结果进行匹配。
        """
        # 从DetectionsWithOdom中提取所有Detection
        own_dets_list = current_own_detections_msg.detections

        own_time = current_own_detections_msg.header.stamp
        num_own = len(own_dets_list)

        if num_own == 0:
            rospy.logdebug("No own detections to match against.")
            return

        rospy.logdebug(
            f"Attempting to match {num_own} own detections from time {own_time.to_sec()}."
        )

        for other_vehicle_id, other_msgs_deque in list(
            self.other_detections_buffer.items()
        ):
            best_other_msg_for_match = None
            min_time_diff_to_own = rospy.Duration(
                MAX_TIME_DIFFERENCE_FOR_MATCHING.to_sec() + 1.0
            )

            for other_det_array_msg in reversed(other_msgs_deque):
                time_diff = abs(own_time - other_det_array_msg.header.stamp)
                if time_diff <= MAX_TIME_DIFFERENCE_FOR_MATCHING:
                    if time_diff < min_time_diff_to_own:
                        min_time_diff_to_own = time_diff
                        best_other_msg_for_match = other_det_array_msg
                elif other_det_array_msg.header.stamp < own_time:
                    break

            if (
                best_other_msg_for_match is None
                or not best_other_msg_for_match.detections
            ):
                rospy.logdebug(
                    f"No suitable time-synced detections from {other_vehicle_id} for own detections at {own_time.to_sec()}."
                )
                continue

            # 从其他车辆的DetectionsWithOdom中提取所有Detection
            other_dets_list = best_other_msg_for_match.detections

            other_time = best_other_msg_for_match.header.stamp
            num_other = len(other_dets_list)

            rospy.loginfo(
                f"Selected {num_other} detections from {other_vehicle_id} (time: {other_time.to_sec()}, diff: {min_time_diff_to_own.to_sec():.3f}s) for matching."
            )

            # 选择匹配方法
            if USE_GRAPH_MATCHING and num_own > 1 and num_other > 1:
                # 如果启用了图匹配，并且两组检测都大于1，则使用图匹配方法
                self._graph_matching_approach(
                    own_dets_list, other_dets_list, other_vehicle_id
                )
                rospy.loginfo(
                    f"Graph matching attempted for {other_vehicle_id} with {num_own} own detections and {num_other} other detections."
                )
            else:
                self._traditional_matching_approach(
                    own_dets_list, other_dets_list, other_vehicle_id
                )
                rospy.loginfo(
                    f"Traditional matching attempted for {other_vehicle_id} with {num_own} own detections and {num_other} other detections."
                )

    def fuse_detections(self, own_det, other_det):
        """根据置信度加权融合两个检测框"""
        own_conf = own_det.confidence
        other_conf = other_det.confidence
        total_conf = own_conf + other_conf

        if total_conf == 0:
            weight_own = 0.5
            weight_other = 0.5
        else:
            weight_own = own_conf / total_conf
            weight_other = other_conf / total_conf

        # 创建融合后的检测
        fused_det = Detections()
        fused_det.object_id = f"fused_{own_det.object_id}_{other_det.object_id}"
        fused_det.type = own_det.type  # 假设类型相同
        fused_det.confidence = max(own_conf, other_conf)  # 使用更高的置信度

        # 融合3D位置
        fused_det.position = Point(
            x=weight_own * own_det.position.x + weight_other * other_det.position.x,
            y=weight_own * own_det.position.y + weight_other * other_det.position.y,
            z=weight_own * own_det.position.z + weight_other * other_det.position.z,
        )

        # 融合2D边界框
        fused_det.box_2d = [
            int(weight_own * own_det.box_2d[0] + weight_other * other_det.box_2d[0]),
            int(weight_own * own_det.box_2d[1] + weight_other * other_det.box_2d[1]),
            int(weight_own * own_det.box_2d[2] + weight_other * other_det.box_2d[2]),
            int(weight_own * own_det.box_2d[3] + weight_other * other_det.box_2d[3]),
        ]

        # 融合3D边界框
        fused_det.bbox_3d = [
            weight_own * own_det.bbox_3d[i] + weight_other * other_det.bbox_3d[i]
            for i in range(6)
        ]

        return fused_det

    def publish_fused_detections(self, fused_detections, timestamp):
        """发布融合后的检测结果"""
        if not fused_detections:
            return

        fused_msg = DetectionsWithOdom()
        fused_msg.header = Header(stamp=timestamp, frame_id="world")
        fused_msg.car_id = f"{self.current_vehicle_id}_fused"

        # 创建默认里程计数据
        from geometry_msgs.msg import Pose, Quaternion, Twist, Vector3

        odom = Odometry()
        odom.header = Header(stamp=timestamp, frame_id="world")
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(x=0, y=0, z=0),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        )
        odom.twist.twist = Twist()
        fused_msg.odom = odom

        fused_msg.detections = fused_detections
        self.fused_pub.publish(fused_msg)

    def notify_visualizer_matches(self, vehicle_id, matches):
        """通知可视化器匹配结果"""
        try:
            # 尝试导入并获取可视化器实例
            from detection_visualizer import get_visualizer

            visualizer = get_visualizer()
            visualizer.update_matches(vehicle_id, matches)
        except Exception as e:
            rospy.logdebug(f"Could not update visualizer matches: {e}")

    def _traditional_matching_approach(
        self, own_dets_list, other_dets_list, other_vehicle_id
    ):
        """
        使用传统的IoU/距离匹配方法 - 通过RPC调用
        """
        result = self._call_rpc_matching(
            "traditional_matching", own_dets_list, other_dets_list, other_vehicle_id
        )

        if result is None:
            rospy.logwarn(
                f"Traditional matching RPC failed for {other_vehicle_id}. Skipping matching."
            )
            return

        if not result.get("success", False):
            error_msg = result.get("error_message", "Unknown error")
            rospy.logwarn(
                f"Traditional matching failed for {other_vehicle_id}: {error_msg}"
            )
            return

        matches = result.get("matches", [])

        rospy.loginfo(
            f"--- Traditional Matching Results for Own vs {other_vehicle_id} ---"
        )

        # 生成融合检测结果
        fused_detections = []
        for original_own_idx, original_other_idx in matches:
            if original_own_idx < len(own_dets_list) and original_other_idx < len(
                other_dets_list
            ):
                own_det_orig = own_dets_list[original_own_idx]
                other_det_orig = other_dets_list[original_other_idx]

                # 计算度量值用于日志显示
                if USE_IOU_FOR_COST:
                    metric_value = self._calculate_iou(own_det_orig, other_det_orig)
                else:
                    metric_value = self._calculate_center_distance(
                        own_det_orig, other_det_orig
                    )

                rospy.loginfo(
                    f"  TRADITIONAL MATCH: OwnDet_Orig[{original_own_idx}] (ID:{own_det_orig.object_id}, {own_det_orig.type}@{own_det_orig.confidence:.2f}) "
                    f"<-> {other_vehicle_id} Det_Orig[{original_other_idx}] (ID:{other_det_orig.object_id}, {other_det_orig.type}@{other_det_orig.confidence:.2f}). "
                    f"Metric Value: {metric_value:.2f}"
                )

                # 生成融合检测
                fused_det = self.fuse_detections(own_det_orig, other_det_orig)
                fused_detections.append(fused_det)

        # 发布融合结果
        if fused_detections:
            self.publish_fused_detections(fused_detections, rospy.Time.now())
            rospy.loginfo(
                f"Published {len(fused_detections)} fused detections for {other_vehicle_id}"
            )

        # 通知可视化器匹配结果
        self.notify_visualizer_matches(other_vehicle_id, matches)

        rospy.loginfo("----------------------------------------------------")

    def _graph_matching_approach(
        self, own_dets_list, other_dets_list, other_vehicle_id
    ):
        """
        使用图匹配方法进行检测匹配 - 通过RPC调用
        """
        result = self._call_rpc_matching(
            "graph_matching", own_dets_list, other_dets_list, other_vehicle_id
        )

        if result is None:
            rospy.logwarn(
                f"Graph matching RPC failed for {other_vehicle_id}. Falling back to traditional method."
            )
            self._traditional_matching_approach(
                own_dets_list, other_dets_list, other_vehicle_id
            )
            return

        if not result.get("success", False):
            error_msg = result.get("error_message", "Unknown error")
            rospy.logwarn(
                f"Graph matching failed for {other_vehicle_id}: {error_msg}. Falling back to traditional method."
            )
            self._traditional_matching_approach(
                own_dets_list, other_dets_list, other_vehicle_id
            )
            return

        matches = result.get("matches", [])

        rospy.loginfo(f"--- Graph Matching Results for Own vs {other_vehicle_id} ---")

        # 生成融合检测结果
        fused_detections = []
        for own_idx, other_idx in matches:
            if own_idx < len(own_dets_list) and other_idx < len(other_dets_list):
                own_det = own_dets_list[own_idx]
                other_det = other_dets_list[other_idx]
                rospy.loginfo(
                    f"  GRAPH MATCH: OwnDet[{own_idx}] (ID:{own_det.object_id}, {own_det.type}@{own_det.confidence:.2f}) "
                    f"<-> {other_vehicle_id} Det[{other_idx}] (ID:{other_det.object_id}, {other_det.type}@{other_det.confidence:.2f}). "
                )

                # 生成融合检测
                fused_det = self.fuse_detections(own_det, other_det)
                fused_detections.append(fused_det)

        # 发布融合结果
        if fused_detections:
            self.publish_fused_detections(fused_detections, rospy.Time.now())
            rospy.loginfo(
                f"Published {len(fused_detections)} fused detections for {other_vehicle_id}"
            )

        # 通知可视化器匹配结果
        self.notify_visualizer_matches(other_vehicle_id, matches)

        rospy.loginfo("----------------------------------------------------")


if __name__ == "__main__":
    try:
        DetectionFuser()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection Fuser Node Shutdown.")
