
from flask import Flask, render_template, Response, jsonify,request
from camera import VideoCamera
# from Selection import selection
import cv2
import base64
import pandas as pd
import numpy as np
import cv2
Response, jsonify
import os
import glob

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/selection', methods=['GET'])   #methods=['GET', 'POST']
def selection():
    print("*****Start Shirt Selection****************")
    df = pd.read_csv('Mens_Shirt.csv')

    DATADIR = r"Style/"
    MIN_DISTANCE = 1.5

    df['vector'] = np.array(df[['Size', 'Regular Type', 'Colour']]).tolist()
    df.head()

    in_vect = input('Enter your expectation:').split(',')
    user_input = np.array(list(map(lambda x: int(x), in_vect)))
    print(user_input)

    df['distance'] = df['vector'].apply(lambda x: np.linalg.norm(np.array(x) - user_input))
    selected = df[df['distance'] < MIN_DISTANCE]
    selected.sort_values('distance', ascending=True, inplace=True)
    selected.head()

    slected_ids = [row for row in selected["id"]]

    images = list()
    for path in os.listdir(DATADIR):
        if int(path.split('.')[0]) in slected_ids:
            im_path = DATADIR + path
            print("im_path is : "+im_path)
            images.append(cv2.imread(im_path))

    encoded_imges = []

    for ix, img in enumerate(images):
        cv2.imshow('Selection' + str(ix), img)

    cv2.waitKey(0)
    return jsonify(img)

    # return Response(response=img, content_type='image/jpeg')
    # return render_template('image.html', user_image=img)

@app.route('/user_image', methods=['GET'])
def hello():
    print("*****Start Shirt user_image Results Test****************")
    target = os.path.join(APP_ROOT,'Style')
    path = request.form[r'path']

    image = [i for i in os.listdir(target) if i.endswith(".jpg")][0]
    return render_template('image.html', image = image)


if __name__ == "__main__":
    app.run(port=4555, debug=True)