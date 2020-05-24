from mtcnn import MTCNN
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint

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
class_name = ("Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black", "Green", "Green Gray", "Other")
EyeColor = {
    class_name[0]: ((166, 21, 50), (240, 100, 85)),
    class_name[1]: ((166, 2, 25), (300, 20, 75)),
    class_name[2]: ((2, 20, 20), (40, 100, 60)),
    class_name[3]: ((20, 3, 30), (65, 60, 60)),
    class_name[4]: ((0, 10, 5), (40, 40, 25)),
    class_name[5]: ((60, 21, 50), (165, 100, 85)),
    class_name[6]: ((60, 2, 25), (165, 20, 65))
}


def getFaceDetails(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(gray_image)
    getEyesColor(gray_image, result)

    faceBox = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image, (faceBox[0], faceBox[1]),
                  (faceBox[0] + faceBox[2], faceBox[1] + faceBox[3]),
                  (0, 155, 255), 2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    cv2.imwrite("image_draw.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print('\n')
    return result


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
    faceBoxes = highlightFace(faceNet, image)
    padding = 20
    for faceBox in faceBoxes:
        face = image[max(0, faceBox[1] - padding): min(faceBox[3] + padding, image.shape[0] - 1),
                     max(0, faceBox[0] - padding): min(faceBox[2] + padding, image.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')


def check_color(hsv, color):
    if (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and (hsv[1] >= color[0][1]) and \
            hsv[1] <= color[1][1] and (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2]):
        return True
    return False


def find_class(hsv):
    color_id = 7
    for i in range(len(class_name) - 1):
        if check_color(hsv, EyeColor[class_name[i]]):
            color_id = i
    return color_id


def getEyesColor(image, result):
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    h, w = image.shape[0:2]

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = eye_distance / 15  # approximate

    cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    eye_class = np.zeros(len(class_name), np.float)
    for x in range(h):
        for y in range(w):
            if imgMask[x, y] != 0:
                eye_class[find_class(image[x, y])] += 1

    main_color_index = np.argmax(eye_class[:len(eye_class) - 1])
    total_vote = eye_class.sum()

    print("\n\nDominant Eye Color: ", class_name[main_color_index])
    print("\n **Eyes Color Percentage **")
    for i in range(len(class_name)):
        print(class_name[i], ": ", round(eye_class[i] / total_vote * 100, 2), "%")

    #label = 'Dominant Eye Color: %s' % class_name[main_color_index]
    #cv2.putText(image, label, (left_eye[0] - 10, left_eye[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 255, 0))



def getFilteredImage(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):
    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y):
        return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return occurance_counter, estimator_cluster, hasBlack


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
    colorInformation = []
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding:
        occurance, cluster, black = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index - 1) if ((hasThresholding & hasBlack) & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1] / totalOccurance)

        # make the dictionary of the information
        colorInfo = {"cluster_index": index, "color": color, "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(img, number_of_colors=5, hasThresholding=False):
    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding:
        number_of_colors += 1

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0] * img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()


if __name__ == '__main__':
    image = cv2.imread("kid1.jpg")

    getFaceDetails(image)
    image = imutils.resize(image, width=250)
    skin = getFilteredImage(image)

    dominantColors = extractDominantColor(skin, hasThresholding=True)

    # Show in the dominant color information
    print("Color Information")
    prety_print_data(dominantColors)

    genderAndAgeDetection(image)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

