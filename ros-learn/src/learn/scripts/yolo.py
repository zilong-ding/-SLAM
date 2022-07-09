#!/usr/bin/env python
# license removed for brevity
import cv2
import yolov5
# load pretrained model
classesFile = "/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/yolo-file/coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
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

# set image
# img = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

cap = cv2.VideoCapture("/home/dzl/CLionProjects/-SLAM/ros-learn/src/learn/200862413-1-64.flv")
while(cap.isOpened()):
    ret, img = cap.read()
    classIds = []
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame',gray)
    # perform inference
#     results = model(img)
# #
# # # inference with larger input size
#     results = model(img, size=1280)

# inference with test time augmentation
    results = model(img, size = 1280 ,augment=True)

# parse results
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
    cv2.imshow('images',img)
    cv2.waitKey(1)


# show detection bounding boxes on image
#     results.show()

# save results into "results/" folder
# results.save(save_dir='results/')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()