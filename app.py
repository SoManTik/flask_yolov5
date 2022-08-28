import argparse
import io
import os
from PIL import Image
import json
import torch
import glob
from flask import Flask, render_template, request, redirect, send_file
import numpy as np
from numpy import asarray
import cv2
import random
from PIL import Image as im




def get_Center_point_on_imageFrame(json_object,img):
    numpyImg = asarray(img)
    for obj in range(len(json_object)):
        x = json_object[obj]['xmax']- (json_object[obj]['xmax']-json_object[obj]['xmin'])/2
        y = json_object[obj]['ymax']- (json_object[obj]['ymax']-json_object[obj]['ymin'])/2

        randomColour=random.randrange(100,255)
        numpyImg = cv2.circle(numpyImg,(int(x),int(y)), 3, (randomColour), -1)
        cv2.putText(numpyImg, json_object[obj]['name'], (int(x-10), int(y-10)), cv2.FONT_HERSHEY_DUPLEX,0.5, (0, 240, 0), 1, cv2.LINE_AA)
    img = im.fromarray(numpyImg)
    return img




# using torch.hub.load to get custom mode of yolov 5
model = torch.hub.load('C:/Users/so14085/yolov5', 'custom', path='C:/Users/so14085/yolov5/runs/train/exp5/weights/best.pt', source='local',force_reload=True)

# get path of static/images
images_folder = os.path.join('static', 'images')
    
# initialize file_name
file_name =0




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = images_folder
@app.route("/", methods=["GET","POST"])    

def home():

    if request.method == "POST":
        #receive picture to detect obj using model
        img_bytes = request.files['file'].read()
        img = Image.open(io.BytesIO(img_bytes))
        results =model(img, size=640) 
        
        # get json of label picture
        data = results.pandas().xyxy[0].to_json(orient="records")
        json_object = json.loads(data)
        print(json_object)  
        
        # get object_names from json
        obj_detect=[]
        for obj in json_object:
            obj_detect.append(obj['name'])
        # classify room type
        allobj_toilet=['Toilet','Bathtub','Exhaust Fan', 'Toilet Sink', 'Water Heater' ,'Washing Machine' ,'Water Cubicle', 'Thermo Ventilator' ,'Faucet']
        allobj_kitchen = ['Microwave','Cupboard','Oven','Exhaust Hood','Stove','Dish Washer','Sink',"Washing Machine",'Fridge','Faucet','Exhaust Fan']
        
        
        check_kitchen= filter(lambda a : a in obj_detect, allobj_kitchen)
        check_toilet= filter(lambda a : a in obj_detect, allobj_toilet)

        list_check_kitchen=list(check_kitchen)
        list_check_toilet=list(check_toilet)


        if len(list_check_kitchen)> len(list_check_toilet) :
            typ_of_room="Kitchen"
        else:
            typ_of_room="Toilet"
            
            
        # set name of result
        results = results.render()
        path='C:/Users/so14085/yolov5-flask-master/yolov5-flask-master/static/images'
        os.chdir(path)
        my_files = glob.glob('*.jpg')
        global file_name
        while str(file_name)  in str(my_files)   : 
            file_name=file_name+1    
            
        # save  result of model     
        path='C:/Users/so14085/yolov5-flask-master/yolov5-flask-master'
        os.chdir(path)
        img_base64 = get_Center_point_on_imageFrame(json_object,img)
        for img in results:
            img_base64.save("static/images/"+str(file_name)+".jpg", format="JPEG")
        print(file_name)    
        
    
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], str(file_name)+'.jpg')
        return render_template("image.html",file_name=full_filename,obj_name =obj_detect,typ_of_room=typ_of_room)
        
    return render_template("index.html")



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


