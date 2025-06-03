#!/usr/bin/env python

import rosbag
import rospy
from std_msgs.msg import Header

from demoros2.msg import Detections, DetectionsWithOdom


class OtherVehicleDetector:
    def __init__(self):
        rospy.init_node("other_vehicle_detector", anonymous=True)

        # 获取参数
        self.bag_file = rospy.get_param("~bag_file", "/home/huyaowen/output2.bag")
        self.input_topic = rospy.get_param("~input_topic", "/detection")
        self.vehicle_id = rospy.get_param("~vehicle_id", "vehicle_B")
        self.publish_rate = rospy.get_param("~publish_rate", 1.0)
        self.loop_playback = rospy.get_param("~loop_playback", True)

        # 发布器
        self.pub = rospy.Publisher(
            "other_vehicle_detections", DetectionsWithOdom, queue_size=10
        )

        # 存储bag中的消息
        self.messages = []
        self.current_index = 0

        # 加载rosbag数据
        self.load_bag_data()

        # 设置发布频率
        self.rate = rospy.Rate(self.publish_rate)

        rospy.loginfo(f"Other Vehicle Detector initialized for {self.vehicle_id}")
        rospy.loginfo(f"Loaded {len(self.messages)} messages from {self.bag_file}")

    def load_bag_data(self):
        """从rosbag加载数据"""
        try:
            with rosbag.Bag(self.bag_file, "r") as bag:
                for topic, msg, t in bag.read_messages(topics=[self.input_topic]):
                    # 如果bag中已经是DetectionsWithOdom格式，直接使用
                    if isinstance(msg, DetectionsWithOdom):
                        self.messages.append(msg)
                    # 如果是其他格式，需要转换
                    elif hasattr(msg, "detections"):
                        detection_with_odom = self.convert_to_detections_with_odom(
                            msg, t
                        )
                        self.messages.append(detection_with_odom)
                    else:
                        rospy.logwarn(f"Unknown message type: {type(msg)}")

        except Exception as e:
            rospy.logerr(f"Failed to load bag file {self.bag_file}: {e}")
            # 如果加载失败，创建一些默认数据
            self.create_default_data()

    def create_default_data(self):
        """创建默认测试数据（当bag文件加载失败时使用）"""
        import random

        from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
        from nav_msgs.msg import Odometry

        rospy.logwarn("Creating default test data due to bag loading failure")

        for i in range(10):  # 创建10条测试数据
            detections_with_odom = DetectionsWithOdom()
            detections_with_odom.header = Header(
                stamp=rospy.Time.now(), frame_id="world"
            )
            detections_with_odom.car_id = self.vehicle_id

            # 模拟里程计数据
            odom = Odometry()
            odom.header = Header(stamp=rospy.Time.now(), frame_id="world")
            odom.child_frame_id = "base_link"
            odom.pose.pose = Pose(
                position=Point(x=random.uniform(0, 50), y=random.uniform(0, 50), z=0),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            )
            odom.twist.twist = Twist(
                linear=Vector3(x=random.uniform(0, 10), y=0, z=0),
                angular=Vector3(x=0, y=0, z=random.uniform(-0.5, 0.5)),
            )
            detections_with_odom.odom = odom

            # 模拟检测结果
            num_detections = random.randint(1, 3)
            for j in range(num_detections):
                detection = Detections()
                detection.object_id = f"obj_{j}"
                detection.type = "car"
                detection.confidence = random.uniform(0.7, 0.99)
                detection.box_2d = [
                    int(random.uniform(1, 10)),
                    int(random.uniform(1, 10)),
                    int(random.uniform(12, 20)),
                    int(random.uniform(12, 20)),
                ]
                detection.position = Point(
                    x=random.uniform(1, 20),
                    y=random.uniform(1, 20),
                    z=random.uniform(0, 2),
                )
                detection.bbox_3d = [
                    random.uniform(1.5, 9.5),
                    random.uniform(1.5, 9.5),
                    random.uniform(1.5, 9.5),
                    random.uniform(11, 18),
                    random.uniform(11, 18),
                    random.uniform(11, 18),
                ]
                detections_with_odom.detections.append(detection)

            self.messages.append(detections_with_odom)

    def convert_to_detections_with_odom(self, msg, timestamp):
        """将其他格式的消息转换为DetectionsWithOdom"""
        # 这里需要根据实际的bag数据格式进行转换
        detections_with_odom = DetectionsWithOdom()
        detections_with_odom.header = Header(stamp=timestamp, frame_id="world")
        detections_with_odom.car_id = self.vehicle_id

        # 如果原始消息包含检测数据，复制过来
        if hasattr(msg, "detections"):
            detections_with_odom.detections = msg.detections

        # 如果原始消息包含里程计数据，复制过来
        if hasattr(msg, "odom"):
            detections_with_odom.odom = msg.odom
        else:
            # 创建默认里程计数据
            from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
            from nav_msgs.msg import Odometry

            odom = Odometry()
            odom.header = Header(stamp=timestamp, frame_id="world")
            odom.child_frame_id = "base_link"
            odom.pose.pose = Pose(
                position=Point(x=0, y=0, z=0),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            )
            odom.twist.twist = Twist()
            detections_with_odom.odom = odom

        return detections_with_odom

    def publish_next_message(self):
        """发布下一条消息"""
        if not self.messages:
            rospy.logwarn("No messages to publish")
            return False

        # 获取当前消息
        msg = self.messages[self.current_index]

        # 更新时间戳为当前时间
        msg.header.stamp = rospy.Time.now()
        msg.odom.header.stamp = rospy.Time.now()

        # 确保car_id正确
        msg.car_id = self.vehicle_id

        # 发布消息
        self.pub.publish(msg)

        rospy.loginfo(
            f"Vehicle {self.vehicle_id} publishing {len(msg.detections)} detections."
        )

        # 更新索引
        self.current_index += 1
        if self.current_index >= len(self.messages):
            if self.loop_playback:
                self.current_index = 0
                rospy.loginfo("Restarting bag playback (loop mode)")
            else:
                rospy.loginfo("Finished playing all messages")
                return False

        return True

    def run(self):
        """主运行循环"""
        while not rospy.is_shutdown():
            if not self.publish_next_message():
                if not self.loop_playback:
                    break
            self.rate.sleep()


if __name__ == "__main__":
    try:
        detector = OtherVehicleDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Other Vehicle Detector shutdown.")
