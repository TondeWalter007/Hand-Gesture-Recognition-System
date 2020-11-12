# Hand-Gesture-Recognition-System
Final Year Project

The project is on Skin Colour based hand gesture recognition. The proposed recognition system uses YOLOv3 algorithm to detect the hand and separate it from the input image/video stream. The skin colour of the hand is then extractred using the k-means clustering algorithm which returns the dominant colour from the isolated hand region. This proposed method of skin colour segmentation is used to attempt to reduce the effect of the background lighting conditions. The shape of the hand is then extracted by converting the image into the HSV colour space and determining the range of the skin colour using the extracted dominant colour (which is the skin colour). This is used to create a mask that will sepate the hand from the remaining background. The openCV functions, convex hull and convexity defect algorithm, are used to distinguish the different hand gestures, which are finger counting from zero to five.


ColorConstancy.py contains a single method that pre-processes the input by enhancing the local contrast of an input and results in better quality for detection.  
DominantColorExtraction.py contains multiple methods including the method to extract the dominant colour from the isolated hand region and the method that returns the masked hand image for recognition.

HandGestureRecognitionSystem.py contains the main methods for the hand detection and hand recognition algorithm. I am unable to upload the trained model as the file was too large, therefore, the model needs to be trained first before the recognition system can be tested

The yolov3_hand_model.cfg file contains the configuration settings to train the hand detection algorithm using YOLOv3 and the obj.names file contains a single "Human hand" class to be detected by the algorithm.
The image dataset used to train the model was obtained from Google’s OpenImagesV6 dataset (https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0k65p) and converted into the required format using the OID Toolkit obtained from https://github.com/pythonlessons/OIDv4_ToolKit.git.

The model was trained up to 4000 iterations using 2000 images consisting of hands in a variety of backgrounds. The model was trained using Darknet on Google Collaboratory because of its free access to a GPU accelarator which speeds up the training process. The trained model was saved on my google drive.

