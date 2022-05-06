from flask import Flask, jsonify,request, render_template, redirect
import sys
from flask_ngrok import run_with_ngrok

import os
import cv2
import shutil
import face_recognition
from PIL import Image, ImageDraw

import urllib.request
from werkzeug.utils import secure_filename

import matplotlib.pyplot as plt

from numpy import asarray
from numpy import save
from numpy import load

from picamera import PiCamera
from time import sleep
import time

facesEnc = []
facesNam = []

newImage = 0
print('Starting System')

def init_webhooks(base_url):
    # Update inbound traffic via APIs to use the public-facing ngrok URL
    pass

def create_app():
    app = Flask(__name__)
    
    #testing purposes
    @app.route("/",methods=['GET'])
    def home():
        print("test successful")
        return render_template("home.html")
    
    
    #APIs
    root_directory = "/home/pi/Desktop"
    
    #update model with new images
    @app.route("/updateDB",methods=['GET', 'POST'])
    def updatedb():
        print("updating DB")
        trainAgain()
        return "ok"

    #receive images from server
    @app.route("/test",methods=['GET','POST'])
    def test():
        print("got image")
        rollNo = request.headers['rno']
        rollNo = rollNo+".jpg"
        savePath = "pi-api/trainingData"
        savePath = os.path.join(root_directory,savePath)

        file = request.files['img']
        # filename = secure_filename(file.filename)
        filename = rollNo.replace('/','_')
        file.save(os.path.join(savePath, filename))
        return "ok"

    
    
    
    
    
    #functions

    def trainAgain():
        initialize_app()
        print("training again")
        path = 'pi-api/trainingData'
        path = os.path.join(root_directory, path)
        
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
        dest = "pi-api/deletedData"
        dest = os.path.join(root_directory, dest)

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
            
            global newImage
            if(newImage==1):
                trainAgain()
                continue
        

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
        #print(names)
        #print(facesEnc)
        
  
    
    
    
    
    
    
    
    
    #END

    app.config.from_mapping(
        BASE_URL="http://localhost:5000",
        USE_NGROK=os.environ.get("USE_NGROK", "False") == "True" and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    )

    if app.config.get("ENV") == "development" and app.config["USE_NGROK"]:
        # pyngrok will only be installed, and should only ever be initialized, in a dev environment
        from pyngrok import ngrok

        # Get the dev server port (defaults to 5000 for Flask, can be overridden with `--port`
        # when starting the server
        port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 5000

        # Open a ngrok tunnel to the dev server
        public_url = ngrok.connect(port).public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

        # Update any base URLs or webhooks to use the public ngrok URL
        app.config["BASE_URL"] = public_url
        init_webhooks(public_url)
    return app

 




if __name__ == '__main__':
    app.run()
