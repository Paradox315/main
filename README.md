# 1 基本环境
操作系统：Ubuntu 20.04
ROS版本：Noetic
Python版本：3.8
# 2 安装依赖
```bash
sudo apt install python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo apt install ros-noetic-tf
```
# 3 运行代码
```bash
cd ~/catkin_ws
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
# 然后 source ~/.bashrc 或者打开新终端
# 运行节点
roslaunch vehicle_fusion_detector run_detection_system.launch
```
# 4 节点文件内容

1. own_sensor_detector.py
模拟当前车辆目标检测，结果发布到 `own_sensor_detections` 话题。
2. other_vehicle_detector.py
模拟其他车辆目标检测，结果发布到 `other_vehicle_detections` 话题。
3. fusion_node.py
融合节点，订阅 `own_sensor_detections` 和 `other_vehicle_detections` 话题，进行目标融合，并发布到 `fused_detections` 话题。