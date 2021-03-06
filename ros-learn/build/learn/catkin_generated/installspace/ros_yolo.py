#!/usr/bin/env python3
# license removed for brevity
import rospy
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import yolov5
classesFile = "/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/yolo-file/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
model = yolov5.load('/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/yolo-file/yolov5n6.pt')
#
# # or load custom model
# model = yolov5.load('train/best.pt')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

def detector(img):
    classIds = []
    results = model(img, size = 1280 ,augment=True)
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    length = boxes.shape[0]
    # print(scores)
    for i in range(length):
        x1 = int(boxes[i][0])
        y1 = int(boxes[i][1])
        x2 = int(boxes[i][2])
        y2 = int(boxes[i][3])
        cv2.rectangle(img, (x1, y1), (x2,y2), (255, 0 , 255), 2)
        classIds.append(int(categories[i]))
        # print(classNames[classIds[i]])
        cv2.putText(img,f'{classNames[classIds[i]]} {int(scores[i]*100)}%',
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)




def callback(data):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    detector(img)
    cv2.imshow('images',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow('image')

def listener():
    rospy.init_node('yolo_listener', anonymous=True)
    rospy.Subscriber("image", Image, callback)
    rospy.loginfo("open succeed")
    rospy.spin()


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass


