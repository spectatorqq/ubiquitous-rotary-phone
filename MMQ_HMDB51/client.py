import cv2
import time
from PIL import Image
import numpy as np
import socket
from io import StringIO
from io import BytesIO
from PIL import ImageGrab
import struct
from keras.models import load_model
from HMDB51data import DataSet
import operator
import os

os.environ['CUDA_VISIBLE_DEVICE'] = "0"
# os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
host = '192.168.0.197'


port = 9393
address = (host, port)
# pic = 101
voiec = 102

model = load_model('./data/checkpoints/inception.0026-2.54.hdf5')
data = DataSet()

def to_package(byte, cmd, ver=1):
    '''
    将每一帧的图片流进行二进制数据进行分包
    :param byte: 二进制文件
    :param cmd:
    :param ver:
    :return:
    '''
    head = [ver, len(byte), cmd]
    headPack = struct.pack("!3I", *head)
    senddata = headPack+byte
    return senddata


def pic_to_array(pic):
    stram = BytesIO()
    pic.save(stram, format="JPEG")
    array_pic = np.array(Image.open(stram))
    stram.close()
    return array_pic


def array_pic_to_stream(pic):
    stream = BytesIO()
    pic = Image.fromarray(pic)
    pic.save(stream, format="JPEG")
    jpeg = stream.getvalue()
    stream.close()
    return jpeg



def tcp_camera(process=None):
    cam = cv2.VideoCapture(1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    while cam.isOpened():
        ret, frame = cam.read()
        if ret:
            # cv2.imshow("transmitting", frame)
            # pre_x = get_inputs(frame)
            image_arr = np.resize(frame, (299, 299, 3))
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
            # if sorted_lps[0][0] == 'wave':
            cv2.imshow('client', frame)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, '%s: %.2f' % (sorted_lps[0][0], sorted_lps[0][1]), (50, 50), font, 1.2, (255, 255, 255), 2)
            cv2.putText(frame, '%s: %.2f' % (sorted_lps[1][0], sorted_lps[1][1]), (50, 100), font, 1.2, (255, 255, 255),
                        2)
            cv2.putText(frame, '%s: %.2f' % (sorted_lps[2][0], sorted_lps[3][1]), (50, 150), font, 1.2, (255, 255, 255),
                        2)
            print("camera is open")
            if process:
                '''
                视屏处理模块
                '''
            jpeg = array_pic_to_stream(frame)
            # print(len(jpeg))
            package = to_package(jpeg, 101)
            try:
                if True:
                    sock.send(package)
            except:
                cont = input("远程连接断开， 是否重新连接"
                             "Y:重新连接 N：退出程序")
                if cont.lower() == "n":
                    return
                if cont.lower() == "y":
                    cv2.destroyAllWindows()
                    sock.close()
                    tcp_camera()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            else:
                continue

    cam.release()
    cv2.destroyAllWindows()
    sock.close()


def tcp_screen(process=None):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(address)
    while True:
        img = ImageGrab.grab()
        img = pic_to_array(img)
        if process:
            frame = process(img)
        jpeg = array_pic_to_stream(img)
        jpeg = to_package(jpeg, 101)
        print(len(jpeg))
        try:
            sock.send(jpeg)
        except:
            cont = input("远程连接失败， 是否重连"
                             "Y:重新连接 N:断开连接")
            if cont.lower() == "n":
                print("程序退出、、、")
                return
            if cont.lower() == "y":
                cv2.destroyAllWindows()
                sock.close()
                tcp_screen()
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break
        else:
            continue
    sock.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tcp_camera()
