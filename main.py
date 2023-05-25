from fastapi import FastAPI, File, UploadFile
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import pickle
import os
from rembg import remove
from sklearn.ensemble import RandomForestClassifier


app = FastAPI()

def imageProcess(input_data):
    
    with open('RFA.pkl', 'rb') as file:
        clf=pickle.load(file)

    #input1 = cv2.imread(input_path.file.read())
    #input1 = cv2.imdecode(np.frombuffer(input_path.read(), np.uint8), cv2.IMREAD_COLOR)
    input1 = cv2.imdecode(np.frombuffer(input_data, np.uint8), cv2.IMREAD_COLOR)    
    #resized_image = cv2.resize(input1, (1600, 1200))
    
    #background removal
    output = remove(input1)
    #RGB to gray
    grayscale_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    #median filtering
    medianfilter_image=cv2.medianBlur(grayscale_image,3)    #3 is the kernel size

    #OTSU thresholding
    _ , binary_image=cv2.threshold(medianfilter_image,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Inverted Image
    inverted_image = cv2.bitwise_not(binary_image)
    
    #Canny Edge Detection
    edges = cv2.Canny(medianfilter_image,100,200)

    contours, _ = cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour,True)

        x,y,w,h=cv2.boundingRect(contour)
        length=h
        width=w

        rectangularity=w/h
        if(area != 0):
            rectangularity = w*h/area
        else:
            rectangularity = 0

        if(perimeter!= 0):
            circularity=(4*np.pi*area)/(perimeter**2)
        else:
            circularity=0
        
    #steps to create features data frame
    vector = [area,perimeter,w,h,rectangularity,circularity]
    df_temp = pd.DataFrame([vector])
    df=df_temp
    df.to_csv("features_100.csv", index=False)
    X_test = pd.read_csv("features_100.csv", header=None, names=['feat1', 'feat2', 'feat3', 'feat4', 'feat5', 'feat6'])
    predicted_label=clf.predict(X_test)

    #for testing purpose only !!!!!!!!!!!!!!!!!!!!!!!!
    
    return predicted_label[0]
    

@app.post("/predict")
async def predict_plant(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return {"status": "error", "message": "Image must be jpg or png format!"}
    
    contents = await file.read()
    #image = Image.open(file.file)
    #image = image.convert("RGB")
    #img_path = f"{file.filename}"
    #with open(img_path, "wb") as f:
    #    f.write(contents)
    
    #label = imageProcess(file.file)
    label = imageProcess(contents)
    #os.remove(img_path)
    return {label}
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
