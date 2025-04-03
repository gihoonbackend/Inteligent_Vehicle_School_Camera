#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloVestDetector:
    def __init__(self):
        rospy.init_node('yolo_vest_detector', anonymous=True)
        self.bridge = CvBridge()

        # ✅ 학습된 모델 경로로 변경
        self.model = YOLO('yolov8n.pt')

        # ✅ Vest 관련 클래스 이름 (대소문자 구분 주의)
        self.target_classes = ['car', 'people']
        self.conf_threshold = 0.1 #(신뢰도값)

        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        rospy.loginfo("YOLOv8 Vest Detector 노드 시작됨")
        rospy.spin()

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge 변환 실패: %s", e)
            return

        results = self.model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.model.names[cls_id]

            if class_name in self.target_classes and conf >= self.conf_threshold:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = f'{class_name} {conf:.2f}'
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 - Vest Detection", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        YoloVestDetector()
    except rospy.ROSInterruptException:
        pass
