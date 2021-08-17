import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

import model as models
import load_data as ld
import testing as mt
import cv2
import os
import  random
import matplotlib.pyplot as plt
import os, sys
import numpy as np
from PIL import Image, ImageOps
import csv
import tkinter as tk
import tkinter.messagebox

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = ROOT_DIR+'\Dane\Dane\K002.mp4'
r_wrist_files = [ROOT_DIR+'\Dane\Dane\Dane2\RWrist.csv',ROOT_DIR+'\Dane\Dane\RWrist.csv']
l_wrist_files = [ROOT_DIR+'\Dane\Dane\Dane2\LWrist.csv',ROOT_DIR+'\Dane\Dane\LWrist.csv']
l_ankle = [ROOT_DIR+'\Dane\Dane\Dane2\LAnkle.csv',ROOT_DIR+'\Dane\Dane\LAnkle.csv']
r_ankle = [ROOT_DIR+'\Dane\Dane\Dane2\RAnkle.csv',ROOT_DIR+'\Dane\Dane\RAnkle.csv']
OP_data = [ROOT_DIR+'\Dane\Dane\OP\OP_RWrist.csv',ROOT_DIR+'\Dane\Dane\OP\OP_LWrist.csv',ROOT_DIR+'\Dane\Dane\OP\OP_RAnkle.csv',ROOT_DIR+'\Dane\Dane\OP\OP_LAnkle.csv']

validation_size = 50

# model_path =ROOT_DIR+'\model\model.h5'
# model = tf.keras.models.load_model(model_path)
def train_model():
    # Loading and linking data for training

    frames, training_points = ld.read_and_join(r_wrist_files, l_wrist_files, r_ankle, l_wrist_files)

    # Loading images from video for training
    frames_img = ld.getFramesImages(video_path, frames)

    # Loading OpenPose result for training validation purpose

    Y_test = ld.load_OP_results(OP_data)

    all_frames = np.arange(0, Y_test.shape[0])
    p = np.array(frames)
    new_points = []
    new_points.append(p)
    new_points = np.hstack(new_points)
    all_test_frames = all_frames[np.logical_not(np.isin(all_frames, new_points))]
    test_frames = np.random.choice(all_test_frames, validation_size)
    Y_test = Y_test[test_frames]

    testing_frames = all_test_frames[np.logical_not(np.isin(all_test_frames, test_frames))]
    valid_frames = ld.getFramesImages(video_path, test_frames)

    X_test = np.stack(valid_frames, axis=0)

    flow_points = training_points

    X = np.stack(frames_img, axis=0)
    Y = np.array(flow_points)

    model = models.CNN2D()
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X, Y,
                        validation_data=(X_test, Y_test),
                        batch_size=8,
                        shuffle=True,
                        epochs=500,
                        verbose=1)

    fig_train, ax_train = plt.subplots(nrows=1, ncols=1)
    ax_train.plot(history.history['loss'], label='loss')
    ax_train.plot(history.history['val_loss'], label='val_loss')
    fig_train.legend()
    fig_train.show()

def test_model():

    # Loading and linking data for training
    frames, training_points = ld.read_and_join(r_wrist_files, l_wrist_files, r_ankle, l_wrist_files)
    # Loading images from video for training
    # Loading OpenPose result for training validation purpose
    Y_test = ld.load_OP_results(OP_data)

    all_frames = np.arange(0, Y_test.shape[0])
    p = np.array(frames)
    new_points = []
    new_points.append(p)
    new_points = np.hstack(new_points)
    all_test_frames = all_frames[np.logical_not(np.isin(all_frames, new_points))]
    test_frames = np.random.choice(all_test_frames, validation_size)

    testing_frames = all_test_frames[np.logical_not(np.isin(all_test_frames, test_frames))]
    dist_eukl = mt.test_model(testing_frames)
    mt.show_validation_results(dist_eukl)

    file = open(ROOT_DIR+'/validation.csv', newline ='')
    with file:
        write = csv.writer(file)
        write.writerows(dist_eukl)

def predict_sixteen_random():
    # Loading and linking data for training

    frames, training_points = ld.read_and_join(r_wrist_files, l_wrist_files, r_ankle, l_wrist_files)

    # Loading images from video for training
    frames_img = ld.getFramesImages(video_path, frames)

    # Loading OpenPose result for training validation purpose

    Y_test = ld.load_OP_results(OP_data)

    all_frames = np.arange(0, Y_test.shape[0])
    p = np.array(frames)
    new_points = []
    new_points.append(p)
    new_points = np.hstack(new_points)
    all_test_frames = all_frames[np.logical_not(np.isin(all_frames, new_points))]
    test_frames = np.random.choice(all_test_frames, validation_size)
    Y_test = Y_test[test_frames]

    testing_frames = all_test_frames[np.logical_not(np.isin(all_test_frames, test_frames))]
    valid_frames = ld.getFramesImages(video_path, test_frames)

    X_test = np.stack(valid_frames, axis=0)

    flow_points = training_points

    X = np.stack(frames_img, axis=0)
    Y = np.array(flow_points)

    model_path = ROOT_DIR + '\model\model.h5'
    model = tf.keras.models.load_model(model_path)
    Y_pred = model.predict(X_test)
    plt.figure()

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(valid_frames[i][:, :, 0] if valid_frames[i].shape[2] == 1 else valid_frames[i], cmap='gray')
        plt.plot(Y_test[i, ::2], Y_test[i, 1::2], 'or', label = 'OpenPose model')
        plt.plot(Y_pred[i, ::2], Y_pred[i, 1::2],  'og', label = 'Stworzony model')
    plt.tight_layout()
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP)
    plt.show()


def predict_one(value):

    frames, training_points = ld.read_and_join(r_wrist_files, l_wrist_files, r_ankle, l_wrist_files)

    # Loading OpenPose result for training validation purpose
    Y_test = ld.load_OP_results(OP_data)

    all_frames = np.arange(0, Y_test.shape[0])
    p = np.array(frames)
    new_points = []
    new_points.append(p)
    new_points = np.hstack(new_points)
    all_test_frames = all_frames[np.logical_not(np.isin(all_frames, new_points))]
    idx = np.where(all_test_frames == value)
    if(len(idx[0])==0):
        raise ValueError
    test_frames = []
    test_frames.append(value)

    # if value >75958 or value <0:
    #     tk.messagebox.showerror("Error", "Podaj klatkę filmu z zakresu 0 - 75958")
    #     return

    valid_frames = ld.getFramesImages(video_path, test_frames)
    X_test = np.stack(valid_frames, axis=0)

    flow_points = training_points

    model_path = ROOT_DIR + '\model\model.h5'
    model = tf.keras.models.load_model(model_path)
    Y_pred = model.predict(X_test)

    plt.figure()
    plt.imshow(valid_frames[0][:, :, 0] if valid_frames[0].shape[2] == 1 else valid_frames[0], cmap='gray')
    plt.plot(Y_pred[0, ::2], Y_pred[0, 1::2], 'og', label='Wyniki stworzonego modelu')
    plt.tight_layout()
    plt.show()
    plt.legend()
    np.set_printoptions(precision=4)
    np.savetxt(ROOT_DIR+'/'+str(value)+'.csv', Y_pred, delimiter=';', fmt='%f',
               header='Lewy nadgarstek - x;Lewy nadgarstek - y;Prawy nadgarstek - x;Prawy nadgarstek - y;Lewa kostka - x;Lewa kostka - y; Prawa kostka - x;Prawa kostka - y')




from tkinter import *
root = Tk()
root.title('Detekcja kluczowych punktów na ciele niemowlęcia')
root.geometry("450x200")


test_button = Button(root, text="Wykonaj testowanie modelu", command=test_model)
# test_button.place(x=125, y=250)
test_button.pack(side=TOP, expand=YES, fill=BOTH)

validate_button = Button(root, text="Wyświetl wyniki walidacji",command=mt.valdate_from_file)
# validate_button.place(x=125, y=250)
validate_button.pack(side=TOP, expand=YES, fill=BOTH)

predict_button = Button(root, text="Wyświetl przykładowe klatki zbioru testowego", command=predict_sixteen_random)
# predict_button.place(x=125, y=250)
predict_button.pack(side=TOP, expand=YES, fill=BOTH)

answear = Label(root, text='')
answear.pack(side=BOTTOM, fill=BOTH, expand=YES)

label = Label(root, text='Wybierz klatkę do predycji (1 - 75958):')
label.pack(side=LEFT,anchor=NW, expand = YES,  fill=BOTH)

entry_frames = Entry(root)
# entry_frames.place(x=125, y=250)
entry_frames.pack(side=LEFT,anchor=N, expand = YES, fill=BOTH)

def number():
    try:
        zm = int(entry_frames.get())
        predict_one(zm)
    except ValueError:
        answear.config(text='Numer jest niepoprawny lub zawiera się w zbiorze treningowym')



button = Button(root, text='Wybierz', command=number)
# button.place(x=125, y=250)
button.pack(side=LEFT,anchor=NE, expand=YES, fill=BOTH)



validate_button.pack()
# predict_button.pack()
test_button.pack()
root.mainloop()
