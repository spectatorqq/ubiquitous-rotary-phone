"""
Classify a few images through our CNN.
"""
import numpy as np
import operator
import random
import glob
from HMDB51data import DataSet
from processor import process_image
from keras.models import load_model
import cv2
root_dir = 'F:\\Learning_Python\\document\\hmdb51_org\\HMDB51\\'
# 加载标签库 和 模型
data = DataSet()
model = load_model('./data/checkpoints/inception.0010-1.13.hdf5')  # replaced by your model name


def video_predict(video):
    '''
    输入模型 以预测类别。并输出
    :param video: any video
    :return:  label ,possibility
    '''
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, v = cap.read()
        if ret:
            # Turn the image into an array.
            # print(image)
            image_arr = np.resize(v, (299, 299, 3))
            image_arr = (image_arr / 255.).astype(np.float32)
            image_arr = np.expand_dims(image_arr, axis=0)
            # Predict.
            predictions = model.predict(image_arr)
            # Show how much we think it's each one.
            label_predictions = {}
            for i, label in enumerate(data.classes):
                label_predictions[label] = predictions[0][i]

            sorted_lps = sorted(label_predictions.items(),
                                key=operator.itemgetter(1), reverse=True)
            for i, class_prediction in enumerate(sorted_lps):
                # Just get the top five.
                if i > 4:
                    break
                print("%s: %.2f" % (class_prediction[0], class_prediction[1]))
                i += 1

            # return
    key = cv2.waitKey(25)


video_predict('F:\\Learning_Python\\document\\hmdb51_org\\HMDB5\\test\\wave\\winKen_wave_u_cm_np1_ba_bad_0.avi')