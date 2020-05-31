from mtcnn import MTCNN
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import functools
import pandas as pd

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

detector = MTCNN()


FaceColor = {
    'dark skin tone': [(45, 34, 30), (75, 57, 50)],
    'medium-dark skin tone': [(75, 57, 50), (120, 92, 80)],
    'medium skin tone': [(120, 92, 80), (180, 138, 120)],
    'medium-light skin tone': [(180, 138, 120), (240, 184, 160)],
    'light skin tone': [(240, 184, 160), (255, 229, 200)],
}


def highlightFace(net, image, conf_threshold=0.7):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * image.shape[1])
            y1 = int(detections[0, 0, i, 4] * image.shape[0])
            x2 = int(detections[0, 0, i, 5] * image.shape[1])
            y2 = int(detections[0, 0, i, 6] * image.shape[0])
            faceBoxes.append([x1, y1, x2, y2])

    return faceBoxes


def genderAndAgeDetection(image):
    img = image.copy()
    faceBoxes = highlightFace(faceNet, img)
    padding = 20

    for faceBox in faceBoxes:
        face = img[max(0, faceBox[1] - padding): min(faceBox[3] + padding, image.shape[0] - 1),
                   max(0, faceBox[0] - padding): min(faceBox[2] + padding, image.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        return gender, age


def extractSkin(image):
    img = image.copy()
    # BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV Thresholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # channel mask
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # cleaning mask by Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # extracting skin color from mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # return skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):
    # total number of occurrences for each color
    occurrence_counter = Counter(estimator_labels)

    # Loop through the most common occurring color
    for x in occurrence_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        if Counter(color) == Counter([0, 0, 0]):
            # delete the occurrence
            del occurrence_counter[x[0]]
            # remove the cluster
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return occurrence_counter, estimator_cluster


def extractDominantColor(image, number_of_colors=2):
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    occurrence, cluster = removeBlack(estimator.labels_, estimator.cluster_centers_)
    dominantColors = [int(c) for c in cluster[0]]

    def compare(test_list1, test_list2):
        return functools.reduce(lambda i, j: i and j, map(lambda m, k: m >= k, test_list1, test_list2), True)

    for skin in FaceColor:
        if compare(dominantColors, FaceColor[skin][0]) and compare(FaceColor[skin][1], dominantColors):
            return dominantColors, skin

    return 'dominant color error'


if __name__ == '__main__':
    image = cv2.imread("selfie2.jpg")

    gender, age = genderAndAgeDetection(image)
    print(f'Gender: {gender}, Age: {age[1:-1]} years')

    image = imutils.resize(image, width=250)
    skin = extractSkin(image)

    dominantColors = extractDominantColor(skin)

    if age[1:-1] == '0-2':
        description = 'baby: ' + dominantColors[1]
    elif gender == 'Male' and (age[1:-1] == '4-6' or age[1:-1] == '8-12'):
        description = 'boy: ' + dominantColors[1]
    elif gender == 'Female' and (age[1:-1] == '4-6' or age[1:-1] == '8-12'):
        description = 'girl: ' + dominantColors[1]
    elif gender == 'Male' and (age[1:-1] == '8-12' or age[1:-1] == '15-20' or age[1:-1] == '25-32'):
        description = 'man: ' + dominantColors[1]
    elif gender == 'Female' and (age[1:-1] == '8-12' or age[1:-1] == '15-20' or age[1:-1] == '25-32'):
        description = 'woman: ' + dominantColors[1]
    elif gender == 'Male' and (age[1:-1] == '38-43' or age[1:-1] == '48-53' or age[1:-1] == '60-100'):
        description = 'man: ' + dominantColors[1] + ', white hair'
    elif gender == 'Female' and (age[1:-1] == '38-43' or age[1:-1] == '48-53' or age[1:-1] == '60-100'):
        description = 'woman: ' + dominantColors[1] + ', white hair'

    file_path = 'emoji_df.csv'
    data = pd.read_csv(file_path)
    my_emoji = data.loc[(data.name == description)].emoji
    print(my_emoji)

    cv2.imshow('img', image)
    cv2.imshow('skin', skin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


