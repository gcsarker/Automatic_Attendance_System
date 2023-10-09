from scripts.face_vectors import load_representations, find_target_identity, get_embedding, add_representation
from scripts.face_detection import extract_faces
from datetime import datetime
import cv2
import os
import numpy as np
import pandas as pd

IMG_SIZE = 160
student_df = pd.read_excel('Students_information.xlsx').set_index('ID')

def new_entry_to_database(video_path, database_path = 'database', representations_path = 'representations'):
    
    #Confirmation: 
    while True:
        identity = int(input('Enter Student ID : '))
        name = input('Enter Student Name: ')
        print(f'Student name {identity} . Press \'y\' to confirm')
        key = input()
        if key == 'y':
            break
        else:
            continue
    
    student_df.loc[identity] = name
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else: 
        cap = cv2.VideoCapture(0)
    
    #embeddings = load_representations(representations_path)
    
    try:
      while True:
        ret, frame = cap.read()

        cv2.imshow("Playing",frame)
        #key = cv2.waitKey(1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    save_path = database_path + f'/{str(identity)}/'
    if os.path.isdir(save_path):
        cv2.imwrite(save_path+f'ID_{str(identity)}_{name}.jpg', frame)
    else:
        os.mkdir(save_path)
        cv2.imwrite(save_path+f'ID_{str(identity)}_{name}.jpg', frame)
    
    frame = frame[:, :, [2, 1, 0]]
    target_frame_size = (IMG_SIZE,IMG_SIZE)
    face_obj = extract_faces(frame, target_size = target_frame_size, enforce_detection = False)
    emb  = np.squeeze(get_embedding(face_obj[0][0][0]))
    
    add_representation(identity, emb)
    student_df.to_excel('Students_information.xlsx')