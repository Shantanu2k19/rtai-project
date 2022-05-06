import os
import cv2
import shutil
import face_recognition
from PIL import Image, ImageDraw

import urllib.request
from werkzeug.utils import secure_filename

from numpy import asarray
from numpy import save
from numpy import load

from picamera import PiCamera
from time import sleep
import time

facesEnc = []
facesNam = []

print('Starting System')


    
    
    

def trainAgain():
    print("training again")
    path = '/home/pi/Desktop/pi-api/trainingData'
    #path = os.path.join(root_directory, path)
    
    global facesEnc
    global facesNam
    
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        name = filename.split('.jpg')
        name = name[0]
        
        #read image
        img = face_recognition.load_image_file(f)

        #make encoding
        enc = face_recognition.face_encodings(img)[0]
        
        #save
        facesEnc.append(enc)
        facesNam.append(name)
    
    print("updated for :",facesNam)

    #move files to another directory
    dest = "/home/pi/Desktop/pi-api/deletedData"
    #dest = os.path.join(root_directory, dest)

    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        dest2 = os.path.join(dest, filename)
        # print(filename,"->",f)
        shutil.move(f,dest2)
    
    file1 = open("names.txt","w")
    for x in facesNam:
        file1.write(x+",")
    file1.close()
    
    save('encodings.npy', facesEnc)
    print("saved")
    return
                
def updateAttendance(name):
    print("updating for : ",name)
    #api call
    import requests

    url = "http://shantanu2k17.pythonanywhere.com/mark_present?rno="+name

    payload={}
    headers = {}
    print(url)
    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)
    return
    
        
def findPerson(img,tol=0.4):
    test_image=img
    #cv2.imwrite("/home/pi/Desktop/pi-api/deletedData/lol.jpg",img)
    face_locations = face_recognition.face_locations(test_image)
    all_faces = []
    
    if(len(face_locations)==0):
        print("NO FACE FOUND")
        return all_faces
    
    face_encodings = face_recognition.face_encodings(test_image, face_locations)
    
    # Loop through faces in test image
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(facesEnc, face_encoding, tolerance=tol)
        name = "Unknown Person"
        # If match
        if True in matches:
            first_match_index = matches.index(True)
            name = facesNam[first_match_index]
            all_faces.append(name)
        print(name)
    return all_faces

    
def captureAttendance():
    print("capturing")
    

    while True:
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        people = findPerson(frame)
        if(len(people)!=0):
            updateAttendance(people[0])
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()
        initialize_app()

def initialize_app():
    global facesEnc
    global facesNam
    
    print("initialized")
    facesEnc = load('encodings.npy')
    facesEnc = list(facesEnc)
    
    file = open("names.txt","r+")
    names = file.read()
    file.close()
    names = names.split(',')
    names.remove('')
    facesNam = names
    
        
#trainAgain()
initialize_app()
captureAttendance()

if __name__ == '__main__':
    app.run()



#/Desktop/pi-api $ USE_NGROK=True FLASK_ENV=development FLASK_APP=application.py flask run
