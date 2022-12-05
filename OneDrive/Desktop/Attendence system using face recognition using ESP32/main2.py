import cv2
import urllib.request
import face_recognition
import numpy as np 
import csv
import os
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

def detect_eye(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Applying filter to remove impurities
    gray = cv2.bilateralFilter(gray,5,1,1)

    #Detecting the face for region of image to be fed to eye classifier
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            #roi_face is face which is input to eye classifier
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))

            #Examining the length of eyes object for eyes
            if(len(eyes)==2):
                return True
            else:
                return False
                
def time_cur():
    now=datetime.now()
    current_time=now.strftime("%H:%M:%S")
    return current_time

def st():
    now=datetime.now()
    min=now.strftime("%M")
    min=int(min)
    return min

def load_image(path):
    return face_recognition.load_image_file(path)

def encoding(x):
    return face_recognition.face_encodings(x)[0]

pt="C:\\Users\\hp\\OneDrive\\Desktop\\Attendence system using face recognition using ESP32\\Photos"
known_face_encoding=[]
i=0
list=os.listdir(pt)
for pic in list:
    path=pt+"\\"+pic
    x=load_image(path)
    known_face_encoding.append(encoding(x))
known_face_names=["Aadarsh Ranjan","MS Dhoni","Elon Musk","Harsh Soni"]
known_roll_no=["20001017001","20001017023","20001017025","20001017021"]

students=known_face_names.copy()
face_locations=[]
face_encodings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

f=open(current_date+'.csv','w+',newline='')
lnwrite= csv.writer(f)
lnwrite.writerow(["Name of the student","Time of entry","Roll number","Status"])
url='http://192.168.60.252/cam-hi.jpg'
while True:
    imgResponse=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
    small_frame= cv2.resize(img,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    cv2.imshow("attendence system",img)
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame)
        face_names=[]
        for i in face_encodings:
            matches= face_recognition.compare_faces(known_face_encoding,i)
            name=""
            roll=""
            face_distance=face_recognition.face_distance(known_face_encoding,i)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
                roll=known_roll_no[best_match_index]
            face_names.append(name)
            min=int(now.strftime("%M"))+2
            if name in known_face_names:
                if detect_eye(img):
                    if name in students:
                        if min-st()>0:
                            students.remove(name)
                            print(students)
                            lnwrite.writerow([name,time_cur(),roll,"On time present"])
                        else:
                            students.remove(name)
                            print(students)
                            lnwrite.writerow([name,time_cur(),roll,"Late entry"])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows
f.close()