#!/usr/bin/env python

import random

import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from demoros2.msg import Detections, DetectionsWithOdom, FusionResult, MatchResult


class QuadrantFusionSimulator:
    def __init__(self, quadrant_id):
        self.quadrant_id = quadrant_id
        self.node_name = f"quadrant_{quadrant_id}_fusion_simulator"

        rospy.init_node(self.node_name, anonymous=True)

        # 参数配置
        self.publish_rate = rospy.get_param(
            f"~publish_rate", random.uniform(0.5, 2.0)
        )  # 随机发布频率
        self.quadrant_range = rospy.get_param("~quadrant_range", 50.0)  # 象限范围
        self.max_detections = rospy.get_param("~max_detections", 3)  # 最大检测数量

        # 发布器
        self.fusion_pub = rospy.Publisher(
            f"quadrant_{quadrant_id}/fusion_results", FusionResult, queue_size=10
        )

        # 设置发布频率
        self.rate = rospy.Rate(self.publish_rate)

        # 象限坐标偏移
        self.coordinate_offset = self._get_coordinate_offset()

        rospy.loginfo(f"Quadrant {quadrant_id} Fusion Simulator initialized")
        rospy.loginfo(f"Publish rate: {self.publish_rate:.1f} Hz")
        rospy.loginfo(f"Coordinate offset: {self.coordinate_offset}")

    def _get_coordinate_offset(self):
        """获取象限坐标偏移"""
        # 象限坐标偏移：相对于全局坐标系的偏移
        if self.quadrant_id == 1:  # 第一象限 (x>0, y>0)
            return (self.quadrant_range / 4, self.quadrant_range / 4)
        elif self.quadrant_id == 2:  # 第二象限 (x<0, y>0)
            return (-self.quadrant_range / 4, self.quadrant_range / 4)
        elif self.quadrant_id == 3:  # 第三象限 (x<0, y<0)
            return (-self.quadrant_range / 4, -self.quadrant_range / 4)
        elif self.quadrant_id == 4:  # 第四象限 (x>0, y<0)
            return (self.quadrant_range / 4, -self.quadrant_range / 4)
        else:
            return (0, 0)

    def create_random_detection(self, prefix, detection_id):
        """创建随机检测"""
        detection = Detections()
        detection.object_id = f"{prefix}_{detection_id}"
        detection.type = random.choice(["car", "truck", "bus", "pedestrian"])
        detection.confidence = random.uniform(0.6, 0.95)

        # 在象限范围内生成随机坐标
        local_x = random.uniform(-self.quadrant_range / 4, self.quadrant_range / 4)
        local_y = random.uniform(-self.quadrant_range / 4, self.quadrant_range / 4)
        local_z = random.uniform(0, 2)

        detection.position = Point(x=local_x, y=local_y, z=local_z)

        # 生成2D边界框
        box_size = random.uniform(2, 8)
        detection.box_2d = [
            int(local_x - box_size / 2),
            int(local_y - box_size / 2),
            int(local_x + box_size / 2),
            int(local_y + box_size / 2),
        ]

        # 生成3D边界框
        detection.bbox_3d = [
            local_x - box_size / 2,
            local_y - box_size / 2,
            local_z,
            local_x + box_size / 2,
            local_y + box_size / 2,
            local_z + 2,
        ]

        return detection

    def create_detections_with_odom(self, car_id, num_detections):
        """创建带里程计的检测消息"""
        current_time = rospy.Time.now()

        msg = DetectionsWithOdom()
        msg.header = Header(stamp=current_time, frame_id="world")
        msg.car_id = car_id

        # 创建里程计数据
        odom = Odometry()
        odom.header = Header(stamp=current_time, frame_id="world")
        odom.child_frame_id = "base_link"

        # 在象限内随机位置
        offset_x, offset_y = self.coordinate_offset
        odom.pose.pose = Pose(
            position=Point(
                x=offset_x + random.uniform(-10, 10),
                y=offset_y + random.uniform(-10, 10),
                z=0,
            ),
            orientation=Quaternion(x=0, y=0, z=0, w=1),
        )
        odom.twist.twist = Twist()
        msg.odom = odom

        # 创建检测结果
        for i in range(num_detections):
            detection = self.create_random_detection(car_id, i)
            msg.detections.append(detection)

        return msg

    def create_match_results(self, own_count, other_counts):
        """创建匹配结果"""
        match_results = []
        current_time = rospy.Time.now()

        # 模拟匹配关系
        for vehicle_idx, other_count in enumerate(other_counts):
            if other_count > 0 and own_count > 0:
                # 随机创建一些匹配
                num_matches = min(own_count, other_count, random.randint(0, 2))

                if num_matches > 0:
                    match_result = MatchResult()
                    match_result.header = Header(stamp=current_time, frame_id="world")
                    match_result.own_vehicle_id = f"Q{self.quadrant_id}_own"
                    match_result.other_vehicle_id = (
                        f"Q{self.quadrant_id}_other_{vehicle_idx}"
                    )

                    # 随机选择匹配的索引
                    own_indices = random.sample(range(own_count), num_matches)
                    other_indices = random.sample(range(other_count), num_matches)
                    match_scores = [
                        random.uniform(0.5, 0.9) for _ in range(num_matches)
                    ]

                    match_result.own_indices = own_indices
                    match_result.other_indices = other_indices
                    match_result.match_scores = match_scores

                    match_results.append(match_result)

        return match_results

    def create_fused_detections(
        self, own_detections, other_detections_list, match_results
    ):
        """创建融合后的检测结果"""
        current_time = rospy.Time.now()

        # 找出被匹配的检测索引
        matched_own_indices = set()
        for match_result in match_results:
            matched_own_indices.update(match_result.own_indices)

        fused_msg = DetectionsWithOdom()
        fused_msg.header = Header(stamp=current_time, frame_id="world")
        fused_msg.car_id = f"Q{self.quadrant_id}_fused"

        # 创建默认里程计
        odom = Odometry()
        odom.header = Header(stamp=current_time, frame_id="world")
        odom.child_frame_id = "base_link"
        odom.pose.pose = Pose(
            position=Point(x=0, y=0, z=0), orientation=Quaternion(x=0, y=0, z=0, w=1)
        )
        odom.twist.twist = Twist()
        fused_msg.odom = odom

        # 创建融合检测：为每个匹配的own检测创建一个融合检测
        for own_idx in matched_own_indices:
            if own_idx < len(own_detections):
                original_det = own_detections[own_idx]

                # 创建融合检测（稍微调整位置和置信度）
                fused_det = Detections()
                fused_det.object_id = f"fused_{self.quadrant_id}_{own_idx}"
                fused_det.type = original_det.type
                fused_det.confidence = min(
                    0.95, original_det.confidence + 0.1
                )  # 提高置信度

                # 稍微调整位置（模拟融合效果）
                fused_x = original_det.position.x + random.uniform(-0.5, 0.5)
                fused_y = original_det.position.y + random.uniform(-0.5, 0.5)
                fused_det.position = Point(
                    x=fused_x, y=fused_y, z=original_det.position.z
                )

                # 调整边界框
                fused_det.box_2d = [
                    int(original_det.box_2d[0] + random.uniform(-1, 1)),
                    int(original_det.box_2d[1] + random.uniform(-1, 1)),
                    int(original_det.box_2d[2] + random.uniform(-1, 1)),
                    int(original_det.box_2d[3] + random.uniform(-1, 1)),
                ]

                fused_det.bbox_3d = [
                    original_det.bbox_3d[0] + random.uniform(-0.5, 0.5),
                    original_det.bbox_3d[1] + random.uniform(-0.5, 0.5),
                    original_det.bbox_3d[2],
                    original_det.bbox_3d[3] + random.uniform(-0.5, 0.5),
                    original_det.bbox_3d[4] + random.uniform(-0.5, 0.5),
                    original_det.bbox_3d[5],
                ]

                fused_msg.detections.append(fused_det)

        return fused_msg

    def generate_fusion_result(self):
        """生成融合结果"""
        current_time = rospy.Time.now()

        # 创建融合结果消息
        fusion_result = FusionResult()
        fusion_result.header = Header(stamp=current_time, frame_id="world")
        fusion_result.own_vehicle_id = f"Q{self.quadrant_id}_own"

        # 创建own检测结果
        own_count = random.randint(1, self.max_detections)
        fusion_result.own_detections_with_odom = self.create_detections_with_odom(
            f"Q{self.quadrant_id}_own", own_count
        )

        # 创建other车辆检测结果（1-2个其他车辆）
        num_other_vehicles = random.randint(1, 2)
        other_counts = []

        for i in range(num_other_vehicles):
            other_count = random.randint(0, self.max_detections)
            other_counts.append(other_count)

            if other_count > 0:
                other_msg = self.create_detections_with_odom(
                    f"Q{self.quadrant_id}_other_{i}", other_count
                )
                fusion_result.other_detections_list.append(other_msg)

        # 创建匹配结果
        fusion_result.match_results = self.create_match_results(own_count, other_counts)

        # 创建融合检测结果
        fusion_result.fused_detections = self.create_fused_detections(
            fusion_result.own_detections_with_odom.detections,
            fusion_result.other_detections_list,
            fusion_result.match_results,
        )

        return fusion_result

    def run(self):
        """运行模拟器"""
        rospy.loginfo(f"Starting Quadrant {self.quadrant_id} fusion simulation...")

        while not rospy.is_shutdown():
            try:
                # 生成并发布融合结果
                fusion_result = self.generate_fusion_result()
                self.fusion_pub.publish(fusion_result)

                rospy.loginfo_throttle(
                    10,
                    f"Q{self.quadrant_id}: Published fusion result - "
                    f"Own: {len(fusion_result.own_detections_with_odom.detections)}, "
                    f"Others: {len(fusion_result.other_detections_list)}, "
                    f"Matches: {len(fusion_result.match_results)}, "
                    f"Fused: {len(fusion_result.fused_detections.detections)}",
                )

                self.rate.sleep()

            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Error in Quadrant {self.quadrant_id} simulation: {e}")
                self.rate.sleep()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        rospy.logerr("Usage: quadrant_fusion_simulator.py <quadrant_id>")
        sys.exit(1)

    try:
        quadrant_id = int(sys.argv[1])
        if quadrant_id not in [1, 2, 3, 4]:
            rospy.logerr("Quadrant ID must be 1, 2, 3, or 4")
            sys.exit(1)

        simulator = QuadrantFusionSimulator(quadrant_id)
        simulator.run()

    except ValueError:
        rospy.logerr("Quadrant ID must be an integer")
        sys.exit(1)
    except rospy.ROSInterruptException:
        rospy.loginfo("Quadrant Fusion Simulator shutdown.")
