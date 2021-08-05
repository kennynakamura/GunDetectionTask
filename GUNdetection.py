import numpy as np
import time
import cv2
import os

labelsPath = os.path.join("C:/Users/kenny.ksn/Desktop/DetectionGun/yolo.names")
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.join("C:/Users/kenny.ksn/Desktop/DetectionGun/yolov3_900.weights")
configPath = os.path.join("C:/Users/kenny.ksn/Desktop/DetectionGun/yolov3_custom_test.cfg")
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)

def predict(image):
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]
    
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.2
    
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)  
    if len(idxs) > 0:
      flag = "S"
    else:
      flag = 'N'    
    return flag
    
img = cv2.imread("C:/Users/kenny.ksn/Desktop/DetectionGun/1.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
if predict(img) == 'N':
  print("Não possui arma")
else:
  print("Possível presença de arma")
