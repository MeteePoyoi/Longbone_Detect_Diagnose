from asyncio.windows_events import NULL
from email import message
import cv2
from cv2 import threshold
import numpy as np
import pickle
import argparse
import io
from PIL import Image
from numpy import asarray, empty
import torch
from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import IntegerField,SubmitField,TextAreaField
import datetime
from wtforms.validators import NumberRange
app = Flask(__name__)
app.config['SECRET_KEY']='mykey'
model_length = pickle.load(open('Model_Length.sav', 'rb'))
model_alingment = pickle.load(open('Model_Alingment.sav', 'rb'))
model_apposition = pickle.load(open('Model_Apposition.sav', 'rb'))

class MyForm(FlaskForm):
    number_length = IntegerField("ป้อนค่า length ด้านล่าง ไม่เกิน 1-100"
                                 ,validators=[NumberRange(min=0, max=100, message="โปรดระบุค่า length ให้อยู่ระหว่าง 1-100")])
    number_alingment = IntegerField("ป้อนค่า alingment ด้านล่าง ไม่เกิน 1-100"
                                    ,validators=[NumberRange(min=0, max=100, message="โปรดระบุค่า alingment ให้อยู่ระหว่าง 1-100")])
    number_apposition = IntegerField("ป้อนค่า apposition ด้านล่าง ไม่เกิน 1-100"
                                     ,validators=[NumberRange(min=0, max=100, message="โปรดระบุค่า apposition ให้อยู่ระหว่าง 1-100")])
    submit = SubmitField("ตกลง")
    
@app.route("/" ,methods=["GET"])
def home():
    return render_template("home_page.html")

@app.route("/help_1" ,methods=["GET"])
def help1():
    return render_template("help_1.html")
@app.route("/help_2" ,methods=["GET"])
def help2():
    return render_template("help_2.html")
@app.route("/help_3" ,methods=["GET"])
def help3():
    return render_template("help_3.html")
@app.route("/help_4" ,methods=["GET"])
def help4():
    return render_template("help_4.html")
@app.route("/page", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)           
        file = request.files["file"]
        if not file:
            return render_template("page_1.html")
     
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model(img, size=640)
        if len(results.xyxy[0]) == 0:
             for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save("static/image0.jpg", format="JPEG")
             return render_template("page_3.html")
        else:
            class_item = int(results.xyxy[0][0][5].item())
            
            x1 = results.xyxy[0][0][0].item()
            y1 = results.xyxy[0][0][1].item()
            x2 = results.xyxy[0][0][2].item()
            y2 = results.xyxy[0][0][3].item()
            img_crop = img.crop((x1,y1,x2,y2))
            img_crop.save("static/image1.jpg", format="JPEG")

            results.render()
            for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save("static/image0.jpg", format="JPEG")

            if class_item == 1:
                return render_template("page_2.html")
            elif class_item == 0:
                return render_template("page_3.html")
            else:
                return render_template("page_1.html")
         
    return render_template("page_1.html")


@app.route('/predict',methods=['POST', 'GET'])
def predicts():
    def get_pixel(img, center, x, y):
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1
        except:
            pass
        return new_value

    def lbp_calculated_pixel(img, x, y):
        center = img[x][y]
        val_ar = []
        val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
        val_ar.append(get_pixel(img, center, x, y+1))       # right
        val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
        val_ar.append(get_pixel(img, center, x+1, y))       # bottom
        val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
        val_ar.append(get_pixel(img, center, x, y-1))       # left
        val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
        val_ar.append(get_pixel(img, center, x-1, y))       # top

        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
        val = 0
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
        return val
    image_file = 'static/image1.jpg'
    # image_resize = cv2.resize(image_file,(100,100))
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    
    pixels = asarray(img_lbp)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    hist_lbp2 = cv2.calcHist([pixels], [0], None, [256], [0, 1])
    nparray = np.array(hist_lbp2)
    np.savetxt("Long_Bone.csv", nparray, delimiter=",")
    np.genfromtxt("Long_Bone.csv", delimiter=",")
    OutputTest = np.genfromtxt("Long_Bone.csv", delimiter=",")
    OutputTest = OutputTest.reshape((1,256))
    prediction_length = model_length.predict(OutputTest)
    prediction_alingment = model_alingment.predict(OutputTest)
    prediction_apposition = model_apposition.predict(OutputTest)
    
    outputLength = round(prediction_length[0])
    outputAlingment = round(prediction_alingment[0])
    outputApposition = round(prediction_apposition[0])
    
    loop = 1
    x = 0
    y= 5
    while loop < 20:
        if outputLength > x and outputLength <= y:
            xLengths = x
            yLengths = y
        if outputAlingment > x and outputAlingment <= y:
            xalingments = x
            yalingments = y
        if outputApposition > x and outputApposition <= y:
            xappositions = x
            yappositions = y
        if outputLength == 0:
            xLengths = 0
            yLengths = 0
        if outputAlingment == 0:
            xalingments = 0
            yalingments = 0
        if outputApposition == 0:
            xappositions = 0
            yappositions = 0
            
        x += 5
        y += 5
        loop += 1
        
    image = cv2.imread("static/image0.jpg")
    now = datetime.datetime.now()
    date_now = str(now.day)+ "/"+str(now.month)+"/"+str(now.year)+" "+str(now.hour)+ ":"+str(now.minute)
    text_image = " Length : "+str(xLengths)+"-"+str(yLengths)+" , Alingment : "+str(xalingments)+"-"+str(yalingments)+" , Apposition : "+str(xappositions)+"-"+str(yappositions)
    image_resize_text = cv2.resize(image,(416,416))
    image_text = cv2.putText(image_resize_text, text_image, (0,410), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 1, cv2.LINE_AA)
    image_date = cv2.putText(image_text, date_now, (300,15), cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite("../yolov5-flask/static/image2.jpg", image_date)
    
    number_length = 6
    number_alingment = 12
    number_apposition = 6
    result = ""
    form = MyForm()
    if form.validate_on_submit():
        number_length = form.number_length.data
        number_alingment = form.number_alingment.data
        number_apposition = form.number_apposition.data
   
        form.number_length.data = ""
        form.number_alingment.data = ""
        form.number_apposition.data = ""
 
        if number_length >= outputLength and number_alingment >= outputAlingment and number_apposition >= outputApposition:
            result = "แนะนำควรเข้าเฝือก"
        else:
            result = "แนะนำผ่าตัด"
    
    if number_length >= outputLength and number_alingment >= outputAlingment and number_apposition >= outputApposition:
        result = "แนะนำควรเข้าเฝือก"

    else:
        result = "แนะนำผ่าตัด"
        
            
    return render_template('last_page.html', xLength = xLengths, yLength = yLengths, xalingment = xalingments,yalingment=yalingments, 
                           xapposition =xappositions,yapposition =yappositions,form=form,number_length=number_length,
                           number_apposition=number_apposition, number_alingment=number_alingment,result_text = result)

if __name__ == "__main__":
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True, autoshape=True
    ) 
    model.eval()
    app.run(debug=True)
