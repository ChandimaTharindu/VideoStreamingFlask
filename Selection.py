# import pandas as pd
# import numpy as np
# import cv2
# import os
#
#
# # class selection(object):
# # df = pd.read_csv('Mens_Shirt.csv')
# #
# # DATADIR = r"Style/"
# # MIN_DISTANCE = 1.5
# #
# # df['vector'] = np.array(df[['Size', 'Regular Type', 'Colour']]).tolist()
# # df.head()
# #
# # # object=input().split(',')
# # # in_vect = input().split(',')
# # # in_vect = input('Enter your expectation:').split(',')
# # # user_input = np.array(list(map(lambda x: int(x), in_vect)))
# # # print(user_input)
# #
# # in_vect = input().split(',')
# # user_input = np.array(list(map(lambda x: int(x), in_vect)))
# # print(user_input)
# #
# # df['distance'] = df['vector'].apply(lambda x: np.linalg.norm(np.array(x) - user_input))
# # selected = df[df['distance'] < MIN_DISTANCE]
# # selected.sort_values('distance', ascending=True, inplace=True)
# # selected.head()
# #
# # slected_ids = [row for row in selected["id"]]
# #
# # images = list()
# # for path in os.listdir(DATADIR):
# #     if int(path.split('.')[0]) in slected_ids:
# #         im_path = DATADIR + path
# #         print(im_path)
# #         images.append(cv2.imread(im_path))
# #
# # for ix, img in enumerate(images):
# #     cv2.imshow('Selection' + str(ix), img)
# #
# # cv2.waitKey(0)
# #
#
# # Uncomment this if images does not appear
# #
# def selection(in_vect):
#     df = pd.read_csv('Mens_Shirt.csv')
#
#     DATADIR = r"Style/"
#     MIN_DISTANCE = 1.5
#
#     df['vector'] = np.array(df[['Size', 'Regular Type', 'Colour']]).tolist()
#     df.head()
#
#     in_vect = input().split(',')
#     user_input = np.array(list(map(lambda x: int(x), in_vect)))
#     print(user_input)
#
#     df['distance'] = df['vector'].apply(lambda x: np.linalg.norm(np.array(x) - user_input))
#     selected = df[df['distance'] < MIN_DISTANCE]
#     selected.sort_values('distance', ascending=True, inplace=True)
#     selected.head()
#
#     slected_ids = [row for row in selected["id"]]
#
#     images = list()
#     for path in os.listdir(DATADIR):
#         if int(path.split('.')[0]) in slected_ids:
#             im_path = DATADIR + path
#             print(im_path)
#             images.append(cv2.imread(im_path))
#
#     for ix, img in enumerate(images):
#         cv2.imshow('Selection' + str(ix), img)
#
#     cv2.waitKey(0)
#     # return Response(response=jpg_as_text, content_type='image/jpeg')
#
# def in_vect():
#     input()
#
# if __name__ == "__main__":
#     selection(in_vect)


import pandas as pd
import numpy as np
import cv2
import os

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
        print(im_path)
        images.append(cv2.imread(im_path))

encoded_imges = []

for ix, img in enumerate(images):
    cv2.imshow('Selection' + str(ix), img)

cv2.waitKey(0)

# Uncomment this if images does not appear



