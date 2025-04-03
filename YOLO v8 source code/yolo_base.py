#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetector:
    def __init__(self):
        rospy.init_node('yolo_v8_detector', anonymous=True)
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # 또는 yolov8s.pt 등

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        rospy.loginfo("YOLOv8 ROS 노드 시작됨")
        rospy.spin()

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge 변환 실패: %s", e)
            return

        # YOLOv8 객체 탐지 수행
        results = self.model(frame)[0]

        # 결과 시각화
        annotated_frame = results.plot()

        cv2.imshow("YOLOv8 Detection", annotated_frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        YoloDetector()
    except rospy.ROSInterruptException:
        pass
