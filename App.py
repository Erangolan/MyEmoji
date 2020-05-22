from mtcnn import MTCNN
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np


detector = MTCNN()
# define HSV color ranges for eyes colors
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

# define HSV color ranges for faces colors
name = ("1", "2", "3", "4", "5", "6", "7", "8", "other")
FaceColor = {
    name[0]: ((45, 34, 30), (60, 46, 40)),
    name[1]: ((75, 57, 50), (90, 69, 60)),
    name[2]: ((105, 80, 70), (120, 92, 80)),
    name[3]: ((135, 103, 90), (150, 114, 100)),
    name[4]: ((165, 126, 110), (180, 138, 120)),
    name[5]: ((195, 149, 130), (210, 161, 140)),
    name[6]: ((225, 172, 150), (240, 184, 160)),
    name[7]: ((255, 195, 170), (255, 206, 180)),
    name[8]: ((255, 218, 190), (255, 229, 200))
}


def getFaceDetails(image):
    result = detector.detect_faces(image)
    print(result)
    getEyesColor(image, result)
    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
                         (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                         (0, 155, 255), 2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    cv2.imwrite("image_draw.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    getFaceColor(image, result)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(result)


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


def find_class2(hsv):
    color_id = 8
    for i in range(len(name) - 1):
        if check_color(hsv, FaceColor[name[i]]):
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

    label = 'Dominant Eye Color: %s' % class_name[main_color_index]
    cv2.putText(image, label, (left_eye[0] - 10, left_eye[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (155, 255, 0))


def getFaceColor(image, result):
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    h, w = image.shape[0:2]

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

    startPoint = (left_eye[0], left_eye[1] + 20) if left_eye[1] > right_eye[1] else (right_eye[0], right_eye[1] + 20)
    endPoint = (result[0]['keypoints']['nose'][0] - int(eye_distance / 2), result[0]['keypoints']['nose'][1]) if \
        startPoint[0] != left_eye[0] else (result[0]['keypoints']['nose'][0] + int(eye_distance / 2), result[0]['keypoints']['nose'][1])

    cv2.rectangle(imgMask, startPoint, endPoint, (255, 255, 255), -1)
    cv2.rectangle(image, startPoint, endPoint, (255, 255, 255), 1)

    face_class = np.zeros(len(name), np.float)
    for x in range(h):
        for y in range(w):
            if imgMask[x, y] != 0:
                face_class[find_class2(image[x, y])] += 1

    main_color_index = np.argmax(face_class[:len(face_class) - 1])
    total_vote = face_class.sum()

    print("\n\nDominant Face Color: ", name[main_color_index])
    print("\n **Face Color Percentage **")
    for i in range(len(name)):
        print(name[i], ": ", round(face_class[i] / total_vote * 100, 2), "%")


if __name__ == '__main__':
    image = cv2.cvtColor(cv2.imread("selfie.jpg"), cv2.COLOR_BGR2RGB)
    getFaceDetails(image)



