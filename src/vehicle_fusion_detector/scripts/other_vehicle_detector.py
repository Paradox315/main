#!/usr/bin/env python

import random

import rospy
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

from demoros2.msg import Detections, DetectionsWithOdom


def talker():
    pub = rospy.Publisher("other_vehicle_detections", DetectionsWithOdom, queue_size=10)
    rospy.init_node("other_vehicle_detector", anonymous=True)
    rate = rospy.Rate(1)  # 1hz

    vehicle_id = "vehicle_B"  # 其他车辆的ID

    while not rospy.is_shutdown():
        detections_with_odom = DetectionsWithOdom()
        detections_with_odom.header = Header(stamp=rospy.Time.now(), frame_id="world")
        detections_with_odom.car_id = vehicle_id

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

        # 模拟产生一些检测结果
        num_detections = random.randint(1, 3)
        for i in range(num_detections):
            detection = Detections()
            detection.object_id = f"obj_{i}"
            detection.type = "car"
            detection.confidence = random.uniform(0.7, 0.99)
            detection.box_2d = [
                int(random.uniform(1, 10)),  # x_min
                int(random.uniform(1, 10)),  # y_min
                int(random.uniform(12, 20)),  # x_max
                int(random.uniform(12, 20)),  # y_max
            ]
            detection.position = Point(
                x=random.uniform(1, 20), y=random.uniform(1, 20), z=random.uniform(0, 2)
            )
            detection.bbox_3d = [
                int(random.uniform(1.5, 9.5)),  # x_min
                int(random.uniform(1.5, 9.5)),  # y_min
                int(random.uniform(1.5, 9.5)),  # z_min
                int(random.uniform(11, 18)),  # x_max
                int(random.uniform(11, 18)),  # y_max
                int(random.uniform(11, 18)),  # z_max
            ]
            detections_with_odom.detections.append(detection)

        if detections_with_odom.detections:
            rospy.loginfo(
                f"Vehicle {vehicle_id} publishing {len(detections_with_odom.detections)} detections."
            )
            pub.publish(detections_with_odom)
        rate.sleep()


if __name__ == "__main__":
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
