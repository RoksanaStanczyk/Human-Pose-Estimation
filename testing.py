from keras.models import load_model
import argparse
import pickle
import cv2
import os
import  random
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from PIL import Image, ImageOps
import csv
from matplotlib.ticker import FormatStrFormatter
import validation as pv
import tensorflow as tf
import pandas as pd
from matplotlib import colors, ticker
from matplotlib.patches import Rectangle
import load_data as ld

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OP_data = [ROOT_DIR+'\Dane\Dane\OP\OP_RWrist.csv',ROOT_DIR+'\Dane\Dane\OP\OP_LWrist.csv',ROOT_DIR+'\Dane\Dane\OP\OP_RAnkle.csv',ROOT_DIR+'\Dane\Dane\OP\OP_LAnkle.csv']

def test_model(exlcudet_frames):

    frames = []
    validation_results = []

    model_path = ROOT_DIR + '\model\model.h5'
    model = tf.keras.models.load_model(model_path)
    openpose = ld.load_OP_results(OP_data)

    j=0
    for i in openpose:
        frames.append(j)
        j+=1
    frames = np.hstack(frames)
    frames[np.logical_not(np.isin(frames, exlcudet_frames))]
    # Opens the Video file
    cap = cv2.VideoCapture(ROOT_DIR+'\Dane\Dane\K002.mp4')
    x = 0
    for f in frames:
        print(f)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, image = cap.read()
        _saved_image = image
        _im = Image.fromarray(_saved_image)
        _resized_image = _im.resize((284, 360))
        _gray_image = ImageOps.grayscale(_resized_image)

        points_test = model.predict(np.array(_gray_image).reshape(1, 360, 284, 1) / 255)

        a = [points_test[0][0::2], points_test[0][1::2]]
        b = [openpose[f][0::2], openpose[f][1::2]]
        validation_results.append(pv.validate(a, b))
        x+=1
    cap.release()
    plt.show()

    return validation_results

def show_validation_results(results):
    x = results
    i = 0
    a = 0
    b = 0
    c = 0
    for r in results:
        i+=1
        if(r<=15):
            a+=1
        elif(r>15 and r<=25):
            b+=1
        else:
            c+=1

    fig, ax = plt.subplots()
    _, _, bars = ax.hist(x, bins=10000, color="C0")
    low = 'green'
    medium = 'yellow'
    high = 'red'

    for bar in bars:
        if int(bar.get_x()) >= 0 and int(bar.get_x()) <= 15:
            bar.set_facecolor(low)

        elif int(bar.get_x()) > 15 and int(bar.get_x()) <= 25:
            bar.set_facecolor(medium)

        else:
            bar.set_facecolor(high)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

    plt.xlabel('Odległość euklidesowa między punktami [px]')
    plt.ylabel('Liczba wystąpień')
    plt.xlim(-10, 100)
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in [low, medium, high]]
    labels = ["Zbiór < 15 [px] - "+str(round(a/i*100,2))+"%", "Zbiór 16 - 25 [px] - "+str(round(b/i*100,2))+"%", "Zbiór > 26 [px] - "+str(round(c/i*100,2))+"%"]
    plt.legend(handles, labels)
    plt.show()
    plt.grid(True)



def valdate_from_file():
    x = []
    tmp_dist = []
    with open('validation.csv') as f:
        reader = csv.reader(f)
        tmp_dist = list(reader)
    #
    for el in tmp_dist:
        for sub_el in el:
            x.append(float(sub_el))

    show_validation_results(x)