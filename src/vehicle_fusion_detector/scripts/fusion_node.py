#!/usr/bin/env python
import collections  # 用于 deque
import json
import os
import sys  # 用于动态添加路径
from typing import Any, Dict, List, Optional, Tuple

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
        修改后的逻辑：先与所有车辆进行匹配，然后统一融合所有关联的检测
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

        # 收集所有有效的其他车辆检测数据
        all_other_vehicles_data = {}  # {vehicle_id: (detections_list, timestamp)}

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
                best_other_msg_for_match is not None
                and best_other_msg_for_match.detections
            ):
                other_dets_list = best_other_msg_for_match.detections
                other_time = best_other_msg_for_match.header.stamp
                all_other_vehicles_data[other_vehicle_id] = (
                    other_dets_list,
                    other_time,
                    min_time_diff_to_own,
                )

                rospy.loginfo(
                    f"Collected {len(other_dets_list)} detections from {other_vehicle_id} "
                    f"(time: {other_time.to_sec()}, diff: {min_time_diff_to_own.to_sec():.3f}s) for matching."
                )

        if not all_other_vehicles_data:
            rospy.logdebug("No suitable other vehicle detections found for matching.")
            return

        # 执行统一的多车辆匹配和融合
        self._multi_vehicle_matching_and_fusion(own_dets_list, all_other_vehicles_data)

    def publish_fused_detections(self, fused_detections, timestamp):
        """发布融合后的检测结果"""
        if not fused_detections:
            rospy.logdebug("No fused detections to publish, skipping...")
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

        # 确认融合结果不为空再发布
        if len(fused_detections) > 0:
            self.fused_pub.publish(fused_msg)
            rospy.loginfo(
                f"Successfully published {len(fused_detections)} fused detections"
            )
        else:
            rospy.logwarn("Attempted to publish empty fused detections list")

    def _multi_vehicle_matching_and_fusion(
        self, own_dets_list, all_other_vehicles_data
    ):
        """
        与多个车辆进行匹配，然后统一融合所有关联的检测
        """
        # 存储每个own检测的所有匹配结果
        # own_matches[own_idx] = [(vehicle_id, other_idx, match_score), ...]
        own_matches = {i: [] for i in range(len(own_dets_list))}

        rospy.loginfo("=== Starting Multi-Vehicle Matching ===")

        # 步骤1：与每个其他车辆进行匹配
        for vehicle_id, (
            other_dets_list,
            other_time,
            time_diff,
        ) in all_other_vehicles_data.items():
            rospy.loginfo(f"Matching with vehicle {vehicle_id}")

            # 选择匹配方法
            if (
                USE_GRAPH_MATCHING
                and len(own_dets_list) > 1
                and len(other_dets_list) > 1
            ):
                matches = self._perform_graph_matching(
                    own_dets_list, other_dets_list, vehicle_id
                )
            else:
                matches = self._perform_traditional_matching(
                    own_dets_list, other_dets_list, vehicle_id
                )

            # 计算匹配得分并存储
            for own_idx, other_idx in matches:
                if own_idx < len(own_dets_list) and other_idx < len(other_dets_list):
                    own_det = own_dets_list[own_idx]
                    other_det = other_dets_list[other_idx]

                    # 计算匹配得分
                    if USE_IOU_FOR_COST:
                        match_score = self._calculate_iou(own_det, other_det)
                    else:
                        # 距离越小得分越高，转换为0-1之间的得分
                        distance = self._calculate_center_distance(own_det, other_det)
                        match_score = max(
                            0, 1.0 - distance / PREFILTER_MAX_DISTANCE_GATE
                        )

                    own_matches[own_idx].append(
                        (vehicle_id, other_idx, match_score, other_det)
                    )

                    rospy.loginfo(
                        f"  MATCH: Own[{own_idx}] <-> {vehicle_id}[{other_idx}] "
                        f"(Score: {match_score:.3f})"
                    )

        # 步骤2：为每个own检测融合所有关联的other检测
        fused_detections = []

        rospy.loginfo("=== Starting Multi-Detection Fusion ===")

        for own_idx, own_det in enumerate(own_dets_list):
            associated_detections = own_matches[own_idx]

            if not associated_detections:
                # 没有关联的检测，保留原始own检测
                fused_det = self._copy_detection(own_det)
                fused_det.object_id = f"own_only_{own_det.object_id}"
                fused_detections.append(fused_det)
                rospy.loginfo(
                    f"  SOLO: Own[{own_idx}] -> No associations, kept as own_only"
                )
            else:
                # 有关联的检测，进行多检测融合
                fused_det = self._fuse_multiple_detections(
                    own_det, associated_detections, own_idx
                )
                fused_detections.append(fused_det)

                vehicle_list = [item[0] for item in associated_detections]
                rospy.loginfo(
                    f"  FUSION: Own[{own_idx}] + {len(associated_detections)} others "
                    f"from {vehicle_list} -> Fused[{fused_det.object_id}]"
                )

        # 步骤3：发布融合结果（一次性发布所有融合后的检测）
        if fused_detections and len(fused_detections) > 0:
            self.publish_fused_detections(fused_detections, rospy.Time.now())
            rospy.loginfo(f"=== Published {len(fused_detections)} fused detections ===")

            # 通知可视化器（需要重新整理匹配关系）
            self._notify_visualizer_multi_matches(own_matches, all_other_vehicles_data)
        else:
            rospy.logwarn("No fused detections generated - fusion list is empty")
            # 发布空的融合结果消息（让可视化器知道融合过程完成了，但没有结果）
            empty_fused_msg = DetectionsWithOdom()
            empty_fused_msg.header = Header(stamp=rospy.Time.now(), frame_id="world")
            empty_fused_msg.car_id = f"{self.current_vehicle_id}_fused"

            from geometry_msgs.msg import Pose, Quaternion, Twist
            from nav_msgs.msg import Odometry

            odom = Odometry()
            odom.header = Header(stamp=rospy.Time.now(), frame_id="world")
            odom.child_frame_id = "base_link"
            odom.pose.pose = Pose(
                position=Point(x=0, y=0, z=0),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            )
            odom.twist.twist = Twist()
            empty_fused_msg.odom = odom
            empty_fused_msg.detections = []  # 空列表

            # 不发布空结果，让可视化器等待非空结果

    def _perform_graph_matching(self, own_dets_list, other_dets_list, vehicle_id):
        """执行图匹配并返回匹配结果"""
        result = self._call_rpc_matching(
            "graph_matching", own_dets_list, other_dets_list, vehicle_id
        )

        if result and result.get("success", False):
            return result.get("matches", [])
        else:
            rospy.logwarn(
                f"Graph matching failed for {vehicle_id}, using traditional matching"
            )
            return self._perform_traditional_matching(
                own_dets_list, other_dets_list, vehicle_id
            )

    def _perform_traditional_matching(self, own_dets_list, other_dets_list, vehicle_id):
        """执行传统匹配并返回匹配结果"""
        result = self._call_rpc_matching(
            "traditional_matching", own_dets_list, other_dets_list, vehicle_id
        )

        if result and result.get("success", False):
            return result.get("matches", [])
        else:
            rospy.logwarn(f"Traditional matching failed for {vehicle_id}")
            return []

    def _copy_detection(self, detection):
        """复制一个检测对象"""
        copied_det = Detections()
        copied_det.object_id = detection.object_id
        copied_det.type = detection.type
        copied_det.confidence = detection.confidence
        copied_det.box_2d = list(detection.box_2d)
        copied_det.position = Point(
            x=detection.position.x, y=detection.position.y, z=detection.position.z
        )
        copied_det.bbox_3d = list(detection.bbox_3d)
        return copied_det

    def _fuse_multiple_detections(self, own_det, associated_detections, own_idx):
        """
        融合一个own检测与多个other检测
        associated_detections: [(vehicle_id, other_idx, match_score, other_det), ...]
        """
        # 计算总权重（包含own检测的权重）
        own_weight = own_det.confidence
        total_weight = own_weight

        weighted_detections = [(own_det, own_weight)]
        vehicle_ids = ["own"]

        for vehicle_id, other_idx, match_score, other_det in associated_detections:
            # 其他检测的权重 = 置信度 * 匹配得分
            other_weight = other_det.confidence * match_score
            total_weight += other_weight
            weighted_detections.append((other_det, other_weight))
            vehicle_ids.append(f"{vehicle_id}[{other_idx}]")

        if total_weight == 0:
            # 避免除零错误
            normalized_weights = [1.0 / len(weighted_detections)] * len(
                weighted_detections
            )
        else:
            normalized_weights = [
                weight / total_weight for _, weight in weighted_detections
            ]

        # 创建融合后的检测
        fused_det = Detections()
        fused_det.object_id = f"fused_{own_idx}_{'_'.join(vehicle_ids[1:])}"
        fused_det.type = own_det.type  # 使用own检测的类型

        # 融合置信度：使用加权平均，但不超过1.0
        fused_confidence = sum(
            det.confidence * weight
            for (det, _), weight in zip(weighted_detections, normalized_weights)
        )
        fused_det.confidence = min(1.0, fused_confidence)

        # 融合3D位置
        fused_x = sum(
            det.position.x * weight
            for (det, _), weight in zip(weighted_detections, normalized_weights)
        )
        fused_y = sum(
            det.position.y * weight
            for (det, _), weight in zip(weighted_detections, normalized_weights)
        )
        fused_z = sum(
            det.position.z * weight
            for (det, _), weight in zip(weighted_detections, normalized_weights)
        )
        fused_det.position = Point(x=fused_x, y=fused_y, z=fused_z)

        # 融合2D边界框
        fused_box_2d = []
        for i in range(4):  # x_min, y_min, x_max, y_max
            fused_coord = sum(
                det.box_2d[i] * weight
                for (det, _), weight in zip(weighted_detections, normalized_weights)
            )
            fused_box_2d.append(int(fused_coord))
        fused_det.box_2d = fused_box_2d

        # 融合3D边界框
        fused_bbox_3d = []
        for i in range(6):
            fused_coord = sum(
                det.bbox_3d[i] * weight
                for (det, _), weight in zip(weighted_detections, normalized_weights)
            )
            fused_bbox_3d.append(fused_coord)
        fused_det.bbox_3d = fused_bbox_3d

        return fused_det

    def _notify_visualizer_multi_matches(self, own_matches, all_other_vehicles_data):
        """通知可视化器多车匹配结果"""
        try:
            from detection_visualizer import get_visualizer

            visualizer = get_visualizer()

            # 为每个车辆整理匹配关系
            for vehicle_id in all_other_vehicles_data.keys():
                vehicle_matches = []
                for own_idx, associations in own_matches.items():
                    for v_id, other_idx, score, _ in associations:
                        if v_id == vehicle_id:
                            vehicle_matches.append((own_idx, other_idx))

                if vehicle_matches:
                    visualizer.update_matches(vehicle_id, vehicle_matches)

        except Exception as e:
            rospy.logdebug(f"Could not update visualizer matches: {e}")

    # 移除原来的单车匹配方法，保留RPC调用的核心逻辑
    def _traditional_matching_approach(
        self, own_dets_list, other_dets_list, other_vehicle_id
    ):
        """已弃用：使用新的多车匹配逻辑替代"""
        rospy.logwarn(
            "_traditional_matching_approach is deprecated, use _multi_vehicle_matching_and_fusion instead"
        )

    def _graph_matching_approach(
        self, own_dets_list, other_dets_list, other_vehicle_id
    ):
        """已弃用：使用新的多车匹配逻辑替代"""
        rospy.logwarn(
            "_graph_matching_approach is deprecated, use _multi_vehicle_matching_and_fusion instead"
        )


if __name__ == "__main__":
    try:
        DetectionFuser()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Detection Fuser Node Shutdown.")
