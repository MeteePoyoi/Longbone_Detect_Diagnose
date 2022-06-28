# heroku-flask-python
# Yolov5 object detection model deployment using flask
This repo contains example apps for exposing the [yolo5](https://github.com/ultralytics/yolov5) object detection model from [pytorch hub](https://pytorch.org/hub/ultralytics_yolov5/) via a [flask](https://flask.palletsprojects.com/en/1.1.x/) api/app.

## Web app
Simple app consisting of a form where you can upload an image, and see the inference result of the model in the browser. Run:

`python app.py`

then visit http://localhost:5000/ in your browser:

<p align="center">
<img src="https://github.com/robmarkcole/yolov5-flask/blob/master/docs/app_form.jpg" width="450">
</p>

<p align="center">
<img src="https://github.com/robmarkcole/yolov5-flask/blob/master/docs/app_result.jpg" width="450">
</p>

<p align="center">
<img src="https://github.com/robmarkcole/yolov5-flask/blob/master/docs/app_result.jpg" width="450">
</p>

## Run & Develop locally
Run locally and dev:
* `py -3 -m venv venv`
* `venv\Scripts\activate`
* `(venv) pip install -r requirements.txt`
* `(venv) python app.py`

## Weight model YOLOv5 & XGBoost deployment using Colab
- https://colab.research.google.com/


## reference
- https://github.com/ultralytics/yolov5
- https://github.com/avinassh/pytorch-flask-api-heroku
- https://github.com/robmarkcole/yolov5-flask