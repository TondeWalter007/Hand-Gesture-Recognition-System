import cv2
import numpy as np
import math
from ColorConstancy import CLAHE
from DominantColorExtraction import *

print("RECOGNITION SYSTEM START...")


def load_weights():
    """"Loads the Yolov3 trained hand detection model"""
    net = cv2.dnn.readNet("yolov3_hand_model.weights", "yolov3_hand_model.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = f.read().splitlines() # extracts the names in the file

    return net, classes


def hand_detection(CLAHE_img, net):
    """"Detects hands from the given input"""
    blob = cv2.dnn.blobFromImage(CLAHE_img, 1 / 255, (416, 416), (0, 0, 0), 
    	swapRB=True, crop=False) # Rescales, Resizes the image and swap BGR to RGB
    net.setInput(blob) # Sets blob as the input to the model
    output_layers_names = net.getUnconnectedOutLayersNames() # gets info from the trained model
    layers_outputs = net.forward(output_layers_names) # makes the prediction of the hand

    return layers_outputs


def bounding_boxes(height, width, layers_outputs):
    boxes = []
    confidences = []
    class_ids = []

    for output in layers_outputs: # extracts info from layer_outputs
        for detection in output: 
            scores = detection[5:] 
            class_id = np.argmax(scores) # gets the highest score's location 
            confidence = scores[class_id] # extracts the highest score
            
            if confidence > 0.3: # Threshold value of the confidence
            	# centre x and y coordinates of detected hand bounding box
                centre_x = int(detection[0] * width) 
                centre_y = int(detection[1] * height)
                # size (width and height) of the bounding box
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # location of the upper left corner of bounding box
                x = int(centre_x - w / 2)
                y = int(centre_y - h / 2)

                # append the acquired info to their variables
                boxes.append([x, y, w, h]) 
                confidences.append((float(confidence)))
                class_ids.append(class_id)

	# suppress redundant bounding boxes and keep highest score box
    bounding_box = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4) 

    return scores, boxes, confidences, bounding_box, class_ids


def hand_isolation(bounding_box, boxes, classes, class_ids, confidences, img):
    
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in bounding_box.flatten():
        x, y, w, h = boxes[i] # extract the location and size of bounding box
        label = str(classes[class_ids[i]]) # extract the corresponding class (human hand)
        confidence = str(round(confidences[i], 2)) # extract the confidence of the detection
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) # create the bounding box

    hand_box = img[y:(y + h), x:(x + w)] # isolate the hand region in the video input

    return hand_box


def hand_recognition(thresh, hand_box):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(hand_box, contours, 0, (255, 255, 0), 3)

    if len(contours) > 0:
        cnt = contours[0]
        hull = cv2.convexHull(cnt, returnPoints=False)
        # finding convexity defects
        defects = cv2.convexityDefects(cnt, hull)
        count_defects = 0
        # apply Cosine Rule to find angle for all defects (between fingers)

        if defects is not None:
            for i in range(defects.shape[0]):
                p, q, r, s = defects[i, 0]
                finger1 = tuple(cnt[p][0])
                finger2 = tuple(cnt[q][0])
                dip = tuple(cnt[r][0])
                # find length of all sides of triangle
                a = math.sqrt((finger2[0] - finger1[0]) ** 2 + (finger2[1] - finger1[1]) ** 2)
                b = math.sqrt((dip[0] - finger1[0]) ** 2 + (dip[1] - finger1[1]) ** 2)
                c = math.sqrt((finger2[0] - dip[0]) ** 2 + (finger2[1] - dip[1]) ** 2)
                # apply cosine rule 
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57.29
                # ignore angles > 90 and highlight angles < 90 with dots
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(hand_box, far, 3, [255, 0, 0], -1)

                cv2.line(hand_box, start, end, [0, 255, 0], 2)
        count += 1

    return count_defects



def actions(img, count_defects, area_cnt, area_ratio):
    # define actions required
    font = cv2.FONT_HERSHEY_SIMPLEX
    if count_defects == 1:
        if area_cnt < 2000:
            cv2.putText(img, "Recognition Failed", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if area_ratio < 12:
                cv2.putText(img, "0", (0, 50), font, 2, (0, 255, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(img, "1", (0, 50), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

    elif count_defects == 2:
        cv2.putText(img, "2", (0, 50), font, 2, (255, 255, 0), 3, cv2.LINE_AA)

    elif count_defects == 3:
        cv2.putText(img, "3", (0, 50), font, 2, (0, 255, 0), 3, cv2.LINE_AA)

    elif count_defects == 4:
        cv2.putText(img, "4", (0, 50), font, 2, (255, 0, 255), 3, cv2.LINE_AA)

    elif count_defects == 5:
        cv2.putText(img, "Recognition Failed", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    else:
        cv2.putText(img, "Recognition Failed", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)


def hand_recognition_system(net, classes):
    # Loading image
    cap = cv2.VideoCapture(0)
    # img = cv2.imread("images/test30.jpg")

    while True:
        _, img = cap.read()
        height, width, _ = img.shape
        CLAHE_img = CLAHE(img)

        layersOutputs = hand_detection(CLAHE_img, net)

        scores, boxes, confidences, indexes, class_ids = bounding_boxes(height, width, layersOutputs)
   
        try:
            hand_box = hand_isolation(indexes, boxes, classes, class_ids, confidences, img)
            cv2.imshow("Detected Hand Thresh", thresh)

            thresh = get_hsv_threshold(hand_box)
            cv2.imshow("Detected Hand Threshold", thresh)

            count_defects, area_cnt, area_ratio = hand_recognition(thresh, hand_box)

            actions(img, count_defects, area_cnt, area_ratio)

        except Exception as e:
            pass

        cv2.imshow("Image", img)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


net, classes = load_weights()
hand_recognition_system(net, classes)

