#!/usr/bin/env python

import os
import subprocess
import time

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class SimpleImageViewer:
    def __init__(self):
        rospy.init_node("simple_image_viewer", anonymous=True)

        # 参数
        self.image_dir = rospy.get_param(
            "~image_dir", "/home/huyaowen/visualization_output"
        )
        self.check_interval = rospy.get_param("~check_interval", 2.0)
        self.use_web_viewer = rospy.get_param("~use_web_viewer", True)
        self.web_port = rospy.get_param("~web_port", 8080)

        # 订阅可视化图像
        rospy.Subscriber("detection_visualization", Image, self.image_callback)

        # 启动web查看器
        if self.use_web_viewer:
            self.start_web_viewer()

        # 定时检查新图像
        rospy.Timer(rospy.Duration(self.check_interval), self.check_new_images)

        rospy.loginfo(f"Simple Image Viewer started, monitoring: {self.image_dir}")
        if self.use_web_viewer:
            rospy.loginfo(f"Web viewer available at: http://localhost:{self.web_port}")

    def image_callback(self, msg):
        """接收图像消息并转换为文件"""
        rospy.logdebug("Received visualization image message")

    def start_web_viewer(self):
        """启动简单的web图像查看器"""
        try:
            # 创建简单的HTML文件
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Detection Visualization</title>
    <meta http-equiv="refresh" content="2">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ccc; }}
        .info {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>Vehicle Detection Fusion Visualization</h1>
    <div class="info">
        <p>Latest visualization image:</p>
        <p>Auto-refresh every 2 seconds</p>
    </div>
    <img src="latest_visualization.jpg" alt="Latest Visualization" id="mainImage">
    <script>
        // 每2秒刷新图像
        setInterval(function(){{
            document.getElementById('mainImage').src = 'latest_visualization.jpg?' + new Date().getTime();
        }}, 2000);
    </script>
</body>
</html>
"""
            html_file = os.path.join(self.image_dir, "viewer.html")
            with open(html_file, "w") as f:
                f.write(html_content)

            # 启动简单的HTTP服务器
            cmd = f"cd {self.image_dir} && python3 -m http.server {self.web_port}"
            subprocess.Popen(cmd, shell=True)

        except Exception as e:
            rospy.logwarn(f"Failed to start web viewer: {e}")

    def check_new_images(self, event=None):
        """检查新的图像文件"""
        if not os.path.exists(self.image_dir):
            return

        try:
            # 找到最新的图像文件
            image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
            if image_files:
                latest_file = max(
                    image_files,
                    key=lambda x: os.path.getctime(os.path.join(self.image_dir, x)),
                )
                latest_path = os.path.join(self.image_dir, latest_file)

                # 创建或更新latest_visualization.jpg链接
                latest_link = os.path.join(self.image_dir, "latest_visualization.jpg")
                if os.path.exists(latest_link):
                    os.remove(latest_link)
                os.symlink(latest_file, latest_link)

                rospy.logdebug(f"Latest image: {latest_file}")

        except Exception as e:
            rospy.logwarn(f"Error checking images: {e}")


if __name__ == "__main__":
    try:
        viewer = SimpleImageViewer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Simple Image Viewer shutdown.")
