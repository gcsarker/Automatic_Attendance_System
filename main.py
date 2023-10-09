from scripts.face_vectors import load_representations, find_target_identity, get_embedding, add_representation
from scripts.face_detection import extract_faces
from datetime import datetime
import cv2
import os
import numpy as np
import pandas as pd

IMG_SIZE = 160
student_df = pd.read_excel('Students_information.xlsx').set_index('ID')

def inference(video_path, database_path = 'database', representations_path = 'representations'):
    attendance_df = student_df.copy()
    attendance_df['Attendance'] = ['Absent' for i in range(student_df.shape[0])]
    
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else: 
        cap = cv2.VideoCapture(0)
    
    embeddings = load_representations(representations_path)
    
    try:
      while True:
        ret, frame = cap.read()
        identity = 'Editing'
           
        frame = frame[:, :, [2, 1, 0]]
        target_frame_size = (IMG_SIZE,IMG_SIZE)
        face_obj = extract_faces(frame, target_size = target_frame_size, enforce_detection = False)
        bbox = face_obj[0][1]
        x = int(bbox['x'])
        y = int(bbox['y'])
        w = int(bbox['w'])
        h = int(bbox['h'])
        
        identity, distance = find_target_identity(face_obj[0][0][0], embeddings)
        if identity == 'Not Found!!':
            name = ''
            continue
        else:
            identity  = int(identity)
            attendance_df.loc[identity, 'Attendance'] = 'Present'
            name = attendance_df.loc[identity, 'Name']
            #print(f'Student ID : {identity}, Name : \'{name}\' Present')
        
        frame = frame[:, :, [2, 1, 0]]
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        cv2.rectangle(frame, (x,y), (x+w, y + h),(0,255,0),3)
        cv2.putText(frame, f'ID : {str(identity)}', (x,y-40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        cv2.putText(frame, f'Name : {str(name)}', (x,y-20), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        cv2.imshow("Frame",frame)
        #key = cv2.waitKey(1)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    
    now = datetime.now()
    # dd/mm/YY H:M:S
    
    now = now.strftime("%d-%b-%Y_%H-%M-%S")
    attendance_df.to_excel('Attendances/'+now+'_attendance.xlsx')


if __name__ == 'main':
	inference(None)