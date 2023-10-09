# Give facial image after extraction
# face_obj = extract_faces(img, target_size = target_frame_size, enforce_detection = False)
from scripts.utility import findCosineDistance
from scripts.face_detection import extract_faces
from scripts.facenet_model import Facenet512
import numpy as np
import pickle
import cv2
import os

facenet_path = 'models/feature_extractors/facenet512_weights.h5'


IMG_SIZE = 160

feature_extractor = Facenet512(dimension = 512, weights_path = facenet_path)

def get_embedding(img):
    img = np.reshape(img, (1, img.shape[0],img.shape[1],img.shape[2]))
    emb = np.array(feature_extractor([img]))
    return emb


def find_target_identity(img, embeddings):
    target_emb = np.squeeze(get_embedding(img))
    prev_dist = 100
    identity = ''
    for key in embeddings.keys():
        dist = findCosineDistance(embeddings[key], target_emb)
        if dist < prev_dist:
            prev_dist = dist
            identity = key
        else:
            continue
    
    if prev_dist > 0.4:
        identity = 'Not Found!!'
        
    return identity, prev_dist


def make_representations(database_path, representations_path):
    embeddings = {}
    target_frame_size = (IMG_SIZE, IMG_SIZE)
    students = os.listdir(database_path)
    
    for student in students:
        dir_path = os.path.join(database_path, student)
        student_imgs = os.listdir(dir_path)
        img_path = os.path.join(dir_path, student_imgs[0])
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
        target_frame_size = (IMG_SIZE,IMG_SIZE)
        face_obj = extract_faces(img, target_size = target_frame_size, enforce_detection = False)
        emb  = np.squeeze(get_embedding(face_obj[0][0][0]))
        embeddings[student] = emb

    # save dictionary 
    with open(representations_path + '/representations.pkl', 'wb') as fp:
        pickle.dump(embeddings, fp)
        print('Face embeddings saved successfully to file')
        fp.close()


def load_representations(representations_path):
    try:
        with open(representations_path + '/representations.pkl', 'rb') as fp:
            embeddings = pickle.load(fp)
            fp.close()
    except:
        embeddings = None
    return embeddings


def add_representation(identity, new_embedding, representations_path = 'representations'):
    embeddings = load_representations(representations_path)
    
    embeddings[identity] = new_embedding
    
    # save dictionary 
    with open(representations_path + '/representations.pkl', 'wb') as fp:
        pickle.dump(embeddings, fp)
        print('Face embeddings saved successfully to file')
        fp.close()
        
    return embeddings