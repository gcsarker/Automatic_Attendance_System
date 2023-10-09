from scripts.utility import findCosineDistance, findEuclideanDistance
from PIL import Image
import numpy as np
import math
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing import image

#import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision

def alignment_procedure(img, left_eye, right_eye):
    # this function aligns given face in img based on left and right eye coordinates

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation
        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    # -----------------------

    return img  # return img anyway



def build_detector_model():
    import mediapipe as mp  # this is not a must dependency. do not import it in the global level.

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
    return face_detection



def detect_faces(face_detector, img, align=True):
    resp = []

    img_width = img.shape[1]
    img_height = img.shape[0]

    results = face_detector.process(img)

    # If no face has been detected, return an empty list
    if results.detections is None:
        return resp

    # Extract the bounding box, the landmarks and the confidence score
    for detection in results.detections:
        (confidence,) = detection.score

        bounding_box = detection.location_data.relative_bounding_box
        landmarks = detection.location_data.relative_keypoints

        x = int(bounding_box.xmin * img_width)
        w = int(bounding_box.width * img_width)
        y = int(bounding_box.ymin * img_height)
        h = int(bounding_box.height * img_height)

        # Extract landmarks
        left_eye = (int(landmarks[0].x * img_width), int(landmarks[0].y * img_height))
        right_eye = (int(landmarks[1].x * img_width), int(landmarks[1].y * img_height))
        # nose = (int(landmarks[2].x * img_width), int(landmarks[2].y * img_height))
        # mouth = (int(landmarks[3].x * img_width), int(landmarks[3].y * img_height))
        # right_ear = (int(landmarks[4].x * img_width), int(landmarks[4].y * img_height))
        # left_ear = (int(landmarks[5].x * img_width), int(landmarks[5].y * img_height))

        if x > 0 and y > 0:
            detected_face = img[y : y + h, x : x + w]
            img_region = [x, y, w, h]

            if align:
                detected_face = alignment_procedure(detected_face, left_eye, right_eye)

            resp.append((detected_face, img_region, confidence))

    return resp



def extract_faces(
    img,
    target_size=(160, 160),
    detector_backend="mediapipe",
    grayscale=False,
    enforce_detection=True,
    align=True,
):
    """Extract faces from an image.

    Args:
        img: a path, url, base64 or numpy array.
        target_size (tuple, optional): the target size of the extracted faces.
        Defaults to (224, 224).
        detector_backend (str, optional): the face detector backend. Defaults to "opencv".
        grayscale (bool, optional): whether to convert the extracted faces to grayscale.
        Defaults to False.
        enforce_detection (bool, optional): whether to enforce face detection. Defaults to True.
        align (bool, optional): whether to align the extracted faces. Defaults to True.

    Raises:
        ValueError: if face could not be detected and enforce_detection is True.

    Returns:
        list: a list of extracted faces.
    """

    # this is going to store a list of img itself (numpy), it region and confidence
    extracted_faces = []

    # img might be path, base64 or numpy array. Convert it to numpy whatever it is.
    #img = load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    #if detector_backend == "skip":
    #    face_objs = [(img, img_region, 0)]
    #else:
    #    face_detector = FaceDetector.build_model(detector_backend)
    #    face_objs = FaceDetector.detect_faces(face_detector, detector_backend, img, align)
    face_detector = build_detector_model()
    face_objs = detect_faces(face_detector, img, align)

    
    
    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        raise ValueError(
            "Face could not be detected. Please confirm that the picture is a face photo "
            + "or consider to set enforce_detection param to False."
        )

    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:
            if grayscale is True:
                current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            # resize and padding
            if current_img.shape[0] > 0 and current_img.shape[1] > 0:
                factor_0 = target_size[0] / current_img.shape[0]
                factor_1 = target_size[1] / current_img.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (
                    int(current_img.shape[1] * factor),
                    int(current_img.shape[0] * factor),
                )
                current_img = cv2.resize(current_img, dsize)

                diff_0 = target_size[0] - current_img.shape[0]
                diff_1 = target_size[1] - current_img.shape[1]
                if grayscale is False:
                    # Put the base image in the middle of the padded image
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                            (0, 0),
                        ),
                        "constant",
                    )
                else:
                    current_img = np.pad(
                        current_img,
                        (
                            (diff_0 // 2, diff_0 - diff_0 // 2),
                            (diff_1 // 2, diff_1 - diff_1 // 2),
                        ),
                        "constant",
                    )

            # double check: if target image is not still the same size with target.
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # normalizing the image pixels
            # what this line doing? must?
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255  # normalize input in [0, 1]

            # int cast is for the exception - object of type 'float32' is not JSON serializable
            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    if len(extracted_faces) == 0 and enforce_detection == True:
        raise ValueError(
            f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False."
        )

    return extracted_faces