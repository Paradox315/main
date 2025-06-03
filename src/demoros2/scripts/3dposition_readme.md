rknn_test: PC上使用RKNN版本
rknnlite_test: RK3588上使用RKNN_lite版本

先修改最前面的参数地址（rknn_test要ONNX_MODEL+RKNN_MODEL，rknnlite_test只需要RKNN_MODEL，CLASSES_PATH对应目标检测类别文件）
修改PUB_TOPIC：发布的topic
修改CAR_ID：车ID
MAX_QUEUE_SIZE：获取数据的size
NUM_WORKER_THREADS = 4：工作线程数量

全局坐标转换：
CAR_MAP -> GLOBAL_MAP
transform = self.tf_buffer.lookup_transform(GLOBAL_MAP, CAR_MAP, pc_msg.header.stamp, rospy.Duration(0.1))

投影矩阵：
projection_matrix -> P_rect
rectification_matrix -> R_rect
外参（顺序x,y,z,qx,qy,qz,qw）-> RT
外参RT转换：
网站  https://staff.aist.go.jp/k.koide/workspace/matrix_converter/matrix_converter.html
数据粘贴进去后，先点击“TUM[tx ty tz qx qy qz qw]”按钮，再点击“Inverse”按钮


env:
torch==1.10.1
opencv-python==4.11.0.86
fast-histogram==0.13
numpy==1.24.4
onnx==1.17.0
onnxruntime==1.16.3
protobuf==4.25.4
psutil==7.0.0
ruamel.yaml==0.18.10
scipy==1.10.1
tqdm==4.67.1
scikit-learn==1.3.0
pyyaml==5.3.1