import cv2
from cv2 import threshold
import numpy as np
import pickle
import argparse
import io
from PIL import Image
from numpy import asarray
import torch
from flask import Flask, render_template, request, redirect
from flask_wtf import FlaskForm
from wtforms import IntegerField,SubmitField

app = Flask(__name__)
app.config['SECRET_KEY']='mykey'
model_length = pickle.load(open('Model_Length.sav', 'rb'))
model_alingment = pickle.load(open('Model_Alingment.sav', 'rb'))
model_apposition = pickle.load(open('Model_Apposition.sav', 'rb'))

class MyForm(FlaskForm):
    number_length = IntegerField("ป้อนค่า length ด้านล่าง ")
    number_alingment = IntegerField("ป้อนค่า alingment ด้านล่าง ")
    number_apposition = IntegerField("ป้อนค่า apposition ด้านล่าง ")
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
            print("redirect(request.url) ===== ",type(redirect(request.url)))
            print("redirect(request.url) ===== ",redirect(request.url))
            return redirect(request.url)
            
        file = request.files["file"]
        print("file ===== ",type(file))
        print("file ===== ",file)
        if not file:
            return render_template("page_1.html")

        img_bytes = file.read()
        print("img_bytes === ",type(img_bytes))
        img = Image.open(io.BytesIO(img_bytes))
        print("img ==== ",type(img))
        print("img ==== ",img)
        results = model(img, size=640)
        print("results ==== ",type(results))
        print("results ==== ",results)

        x1 = results.xyxy[0][0][0].item()
        y1 = results.xyxy[0][0][1].item()
        x2 = results.xyxy[0][0][2].item()
        y2 = results.xyxy[0][0][3].item()
        img_crop = img.crop((x1,y1,x2,y2))
        img_crop.save("static/image1.jpg", format="JPEG")

        results.render()
        print("\nresults.render() ===== " ,type(results.render()))
        print("results.render() ===== " ,results.render())
        for img in results.imgs:
            print("\nimg ==== ",type(img))
            print("img ==== ",img)
            img_base64 = Image.fromarray(img)
            print("img_base64 === ",type(img_base64))
            print("img_base64 === ",img_base64)
            img_base64.save("static/image0.jpg", format="JPEG")
        return render_template("page_2.html")

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
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    pixels = asarray(img_lbp)
    pixels = pixels.astype('float32')
    pixels /= 255.0
    hist_lbp2 = cv2.calcHist([pixels], [0], None, [256], [0, 1])
    nparray = np.array(hist_lbp2)
    np.savetxt("Long_Bone.csv", nparray, delimiter=",")
    np.genfromtxt("Long_Bone.csv", delimiter=",")
    OutputTest = np.genfromtxt("Long_Bone.csv", delimiter=",")
    print("OutputTest ==== ",type(OutputTest))
    print("OutputTest ==== ",OutputTest)
    OutputTest = OutputTest.reshape((1,256))
    print("OutputTest ==== ",type(OutputTest))
    print("OutputTest ==== ",OutputTest)
    prediction_length = model_length.predict(OutputTest)
    prediction_alingment = model_alingment.predict(OutputTest)
    prediction_apposition = model_apposition.predict(OutputTest)

    outputLength = round(prediction_length[0])
    outputAlingment = round(prediction_alingment[0])
    outputApposition = round(prediction_apposition[0])
    
    print("prediction_length ==== ",type(prediction_length))
    print("prediction_length ==== ",prediction_length)
    print("prediction_alingment ==== ",type(prediction_alingment))
    print("prediction_alingment ==== ",prediction_alingment)
    print("prediction_apposition ==== ",type(prediction_apposition))
    print("prediction_apposition ==== ",prediction_apposition)
    print("outputLength ==== ",type(outputLength))
    print("outputLength ==== ",outputLength)
    print("outputAlingment ==== ",type(outputAlingment))
    print("outputAlingment ==== ",outputAlingment)
    print("outputApposition ==== ",type(outputApposition))
    print("outputApposition ==== ",outputApposition)
    loop = 1
    x = 0
    y= 5
    while loop < 20:
        if outputLength > x and outputLength <= y:
            xLengths = x
            yLengths = y
        elif outputAlingment > x and outputAlingment <= y:
            xalingments = x
            yalingments = y
        elif outputApposition > x and outputApposition <= y:
            xappositions = x
            yappositions = y
        elif outputLength == 0:
            xLengths = 0
            yLengths = 0
        elif outputAlingment == 0:
            xalingments = 0
            yalingments = 0
        elif outputApposition == 0:
            xappositions = 0
            yappositions = 0
            
        x += 5
        y += 5
        loop += 1
    number_length = 0.0
    number_alingment = 0.0
    number_apposition = 0.0
    result = ""
    form = MyForm()
    if form.validate_on_submit():
        number_length = form.number_length.data
        number_alingment = form.number_alingment.data
        number_apposition = form.number_apposition.data
        form.number_length.data = ""
        form.number_alingment.data = ""
        form.number_apposition.data = ""
 
        if number_length <= outputLength and number_alingment <= outputAlingment and number_apposition <= outputApposition:
            result = "เข้าเฝือก"
        else:
            result = "ผ่าตัด"
            
    return render_template('last_page.html', xLength = xLengths, yLength = yLengths,
                           xalingment = xalingments,yalingment=yalingments, 
                           xapposition =xappositions,yapposition =yappositions,
                           form=form,number_length=number_length,number_apposition=number_apposition,
                           number_alingment=number_alingment,
                           result_text = result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5s", pretrained=True, force_reload=True, autoshape=True
    )  # force_reload = recache latest code
    model.eval()
    app.run(debug=True)# debug=True causes Restarting with stat
    
    
#   <div style="padding-top: 25px;">

#       <div class"input_check_form">
#         <form method="POST">
#           <div style="background: #0dd9cf;color: white;display: flex;">
#             <center style="padding-right: 10px;padding-left: 10px;">
#               {{form.hidden_tag()}}
#               <b>{{form.number_length.label}}</b><br>
#               {{form.number_length(class="form-control",placeholder="threshold ของ length")}}
#             </center>
#             <center style="padding-right: 10px;">
#               {{form.hidden_tag()}}
#               <b>{{form.number_alingment.label}}</b><br>
#               {{form.number_alingment(class="form-control",placeholder="threshold ของ alingment")}}

#             </center>
#             <center style="padding-right: 10px; padding-bottom: 10px;">
#               {{form.hidden_tag()}}
#               <b>{{form.number_apposition.label}}</b><br>
#               {{form.number_apposition(class="form-control",placeholder="threshold ของ apposition")}}
#             </center>
#           </div>
    #       <center style="padding-bottom: 90px;">
    #         <br>
    #         {{form.submit(class="btn btn-outline-secondary")}}
    #       </center>
    #     </form>
    #   </div>



# <input type="checkbox" id="checkbox_toggle">
#         <label for="checkbox_toggle"><img src="../static/search_icon.png"></label>
#         <div class="input_check_form">
#           <form method="POST">
#             <div style="background: #0dd9cf;color: white;display: flex;">
#               <center style="padding-right: 10px;padding-left: 10px;">
#                 {{form.hidden_tag()}}
#                 <b>{{form.number_length.label}}</b><br>
#                 {{form.number_length(class="form-control",placeholder="threshold ของ length")}}
#               </center>
#               <center style="padding-right: 10px;">
#                 {{form.hidden_tag()}}
#                 <b>{{form.number_alingment.label}}</b><br>
#                 {{form.number_alingment(class="form-control",placeholder="threshold ของ alingment")}}

#               </center>
#               <center style="padding-right: 10px; padding-bottom: 10px;">
#                 {{form.hidden_tag()}}
#                 <b>{{form.number_apposition.label}}</b><br>
#                 {{form.number_apposition(class="form-control",placeholder="threshold ของ apposition")}}
#               </center>
#             </div>