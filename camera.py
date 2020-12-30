import cv2
import imutils
import numpy as np
import time
from flask import Flask, redirect
from time import localtime, strftime

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 1
TIMER = int(20)

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()


        image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        # -----------------------------------------------------------------------------------------------------------------------------------

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_channels = cv2.split(hsv)
        rows = image.shape[0]
        cols = image.shape[1]

        for i in range(0, rows):
            for j in range(0, cols):
                h = hsv_channels[0][i][j]
                if h > 90 and h < 255:
                    hsv_channels[2][i][j] = 255
                else:
                    hsv_channels[2][i][j] = 0

        edges = cv2.Canny(hsv_channels[2], 200, 100)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.erode(edges, None, iterations=1)

        cnts = cv2.findContours(hsv_channels[2].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c1 = max(cnts, key=cv2.contourArea)

        left = tuple(c1[c1[:, :, 0].argmin()][0])
        right = tuple(c1[c1[:, :, 0].argmax()][0])

        distance = np.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)

        x, y, w, h = cv2.boundingRect(c1)
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Shoulder Distance (cm)" + str(distance * 0.08), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0))

        #    ( contours, hierarchy) = cv2.findContours(hsv_channels[2].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #    for pic, contour in enumerate(contours):
        #        area = cv2.contourArea(contour)
        #        if (area > 300):
        #            x, y, w, h = cv2.boundingRect(contour)
        #            img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #            cv2.putText(image, "Stage 2 color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

        centx = np.sqrt(((right[0] + left[0]) ** 2) / 4)
        centy = np.sqrt(((right[1] + left[1]) ** 2) / 4)
        # print(centx, centy)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.circle(hsv_channels[2], left, 5, (255, 0, 0), -1)
        cv2.circle(hsv_channels[2], right, 5, (255, 0, 0), -1)
        cv2.circle(hsv_channels[2], (int(centx), int(centy)), 5, (255, 0, 0), -1)
        cv2.line(hsv_channels[2], left, right, (255, 0, 0), 2)
        cv2.drawContours(hsv_channels[2], [c1], -1, (255, 0, 0), 2)
        cv2.rectangle(hsv_channels[2], (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(hsv_channels[2], 'Distance (cm): ' + str(distance * 0.08), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 0), 2, cv2.LINE_AA)
        print('Dis: ', str(distance * 0.08))

        # font
        cv2.imshow("Color Tracking", hsv_channels[2])

        cv2.imshow("Color Tracking", hsv_channels[2])

        # ----------------------------------------------------------------------------------------------------------------------------------
        for (x, y, w, h) in face_rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

        ret, jpeg = cv2.imencode('.jpg', img)

        return jpeg.tobytes()



    def get_feed(self):
        # image = self.get_frame()
        image = self.video.read()
        if image is not None:

            success, image = self.video.read()

            image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            # -----------------------------------------------------------------------------------------------------------------------------------

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_channels = cv2.split(hsv)
            rows = image.shape[0]
            cols = image.shape[1]

            for i in range(0, rows):
                for j in range(0, cols):
                    h = hsv_channels[0][i][j]
                    if h > 90 and h < 255:
                        hsv_channels[2][i][j] = 255
                    else:
                        hsv_channels[2][i][j] = 0

            edges = cv2.Canny(hsv_channels[2], 200, 100)
            edges = cv2.dilate(edges, None, iterations=1)
            edges = cv2.erode(edges, None, iterations=1)

            cnts = cv2.findContours(hsv_channels[2].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c1 = max(cnts, key=cv2.contourArea)

            left = tuple(c1[c1[:, :, 0].argmin()][0])
            right = tuple(c1[c1[:, :, 0].argmax()][0])

            distance = np.sqrt((right[0] - left[0]) ** 2 + (right[1] - left[1]) ** 2)

            x, y, w, h = cv2.boundingRect(c1)
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Shoulder Distance (cm)" + str(distance * 0.08), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0))

            #    ( contours, hierarchy) = cv2.findContours(hsv_channels[2].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            #    for pic, contour in enumerate(contours):
            #        area = cv2.contourArea(contour)
            #        if (area > 300):
            #            x, y, w, h = cv2.boundingRect(contour)
            #            img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #            cv2.putText(image, "Stage 2 color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

            centx = np.sqrt(((right[0] + left[0]) ** 2) / 4)
            centy = np.sqrt(((right[1] + left[1]) ** 2) / 4)
            # print(centx, centy)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(hsv_channels[2], left, 5, (255, 0, 0), -1)
            cv2.circle(hsv_channels[2], right, 5, (255, 0, 0), -1)
            cv2.circle(hsv_channels[2], (int(centx), int(centy)), 5, (255, 0, 0), -1)
            cv2.line(hsv_channels[2], left, right, (255, 0, 0), 2)
            cv2.drawContours(hsv_channels[2], [c1], -1, (255, 0, 0), 2)
            cv2.rectangle(hsv_channels[2], (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(hsv_channels[2], 'Distance (cm): ' + str(distance * 0.08), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0), 2, cv2.LINE_AA)
            print('Dis: ', str(distance * 0.08))

            # font
            cv2.imshow("Color Tracking", hsv_channels[2])

            cv2.imshow("Color Tracking", img)

            # ----------------------------------------------------------------------------------------------------------------------------------
            # for (x, y, w, h) in face_rects:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     break
            ret, jpeg = cv2.imencode('.jpg', image)
            return jpeg.tobytes()

    def capture(self):
        frame = self.get_frame()
        timestamp = strftime("%d-%m-%Y-%Hh%Mm%Ss", localtime())
        filename = VideoCamera.CAPTURES_DIR + timestamp +".jpg"
        if not cv2.imwrite(filename, frame):
            raise RuntimeError("Unable to capture image "+timestamp)
        return timestamp
