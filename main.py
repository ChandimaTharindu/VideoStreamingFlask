
from flask import Flask, render_template, Response, jsonify,request,redirect, url_for
from camera import VideoCamera
from send_email import Email
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
camera = None
mail_server = None
mail_conf = "static/mail_conf.json"


def get_camera():
    global camera
    if not camera:
        camera = VideoCamera()

    return camera

def get_mail_server():
    global mail_server
    if not mail_server:
        mail_server = Email(mail_conf)

    return mail_server

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

    DATADIR = r"static/Style/"
    MIN_DISTANCE = 1.5

    df['vector'] = np.array(df[['Size', 'Regular Type', 'Colour']]).tolist()
    df.head()

    # input = 2, 2, 7
    # in_vect = input().split(',')
    in_vect = [3, 2, 7]
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
            # print("Get image slected_ids"+slected_ids)
            im_path = DATADIR + path
            print("im_path is : "+im_path)
            images.append(cv2.imread(im_path))

    encoded_imges = []

    for ix, img in enumerate(images):
        cv2.imshow('Selection' + str(ix), img)
        # encoded_imges.append(img)
    # print("encoded_imges is : " + encoded_imges)

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


@app.route('/capture/')
def capture():
    camera = get_camera()
    stamp = camera.capture()
    return redirect(url_for('show_capture', timestamp=stamp))

def stamp_file(timestamp):
    return 'captures/' + timestamp +".jpg"

@app.route('/capture/image/<timestamp>', methods=['POST', 'GET'])
def show_capture(timestamp):
    path = stamp_file(timestamp)

    email_msg = None
    if request.method == 'POST':
        if request.form.get('email'):
            email = get_mail_server()
            email_msg = email.send_email('static/{}'.format(path),
                request.form['email'])
        else:
            email_msg = "Email field empty!"

    return render_template('capture.html',
        stamp=timestamp, path=path, email_msg=email_msg)


@app.route('/getImages', methods=['GET'])
def getImages():
    print("*****Start Shirt Selection****************")
    # df = pd.read_csv('Mens_Shirt.csv')
    #
    # DATADIR = r"Style/"
    # MIN_DISTANCE = 1.5
    #
    # df['vector'] = np.array(df[['Size', 'Regular Type', 'Colour']]).tolist()
    # df.head()
    #
    # # input = 2, 2, 7
    # # in_vect = input().split(',')
    # in_vect = [2, 2, 7]
    # user_input = np.array(list(map(lambda x: int(x), in_vect)))
    # print(user_input)
    #
    # df['distance'] = df['vector'].apply(lambda x: np.linalg.norm(np.array(x) - user_input))
    # selected = df[df['distance'] < MIN_DISTANCE]
    # selected.sort_values('distance', ascending=True, inplace=True)
    # selected.head()
    #
    # slected_ids = [row for row in selected["id"]]
    #
    # images = list()
    # for path in os.listdir(DATADIR):
    #     if int(path.split('.')[0]) in slected_ids:
    #         # print("Get image slected_ids"+slected_ids)
    #         im_path = DATADIR + path
    #         print("im_path is : " + im_path)
    #         images.append(cv2.imread(im_path))
    #
    # # imagelist = [];
    # for ix, img in enumerate(images):
    #     cv2.imshow('Selection' + str(ix), img)
    #     imagelist = ['pics/' + img for img in img]
    #     print("im_path is : " + imagelist)
    #     cv2.imshow('Selection' + imagelist)

    imageList = os.listdir('static/Style')
    imagelist = ['pics/' + img for img in imageList]
    return render_template("templates/image.html", imagelist=imagelist)

    # cv2.waitKey(0)
    # return jsonify(imagelist)


if __name__ == "__main__":
    # app.run(port=4555, debug=True)
    app.run(port=0000, debug=True)
