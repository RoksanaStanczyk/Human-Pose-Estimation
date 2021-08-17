import pandas as pd
import numpy as np
import cv2
import frames_preparation

def readCsv(files):
    r2 = []
    frames = []
    for file in files:
         points = []
         points.append(pd.read_csv(file, delimiter=';', decimal=',', header=None).to_numpy())
         nd_expert_points = np.vstack(points)

         # remove duplicates of frames
         _, unique_frames_ind = np.unique(nd_expert_points[:, 0], return_index=True)
         nd_expert_points = nd_expert_points[unique_frames_ind]

         for el in nd_expert_points:
             r2.append([int(el[1]/3),int(el[2]/3)])
             frames.append(int(el[0]))

    return r2, frames

def load_OP_results(files):
    OP_data = files
    OP_points = dict()

    for op_data in OP_data:
        OP_points[op_data] = pd.read_csv(op_data, delimiter=';').to_numpy()
        print(OP_points)

    Y_test = []
    for op_data in OP_data:
        Y_test.append(OP_points[op_data][:, [0, 1]])
    Y_test = np.hstack(Y_test) / 3
    return Y_test

def read_and_join(rw_files,lw_files,ra_files,la_files):
    r_wrist_points,frames_numbers = readCsv(rw_files)

    l_wrist_points, frames_numbers = readCsv(lw_files)

    l_legs, frames_numbers = readCsv(ra_files)

    r_legs, frames_numbers = readCsv(la_files)


    flow_points = []
    for i in range(0, len(r_wrist_points)):
        flow_points.append(
            [r_wrist_points[i][0], r_wrist_points[i][1], l_wrist_points[i][0], l_wrist_points[i][1], l_legs[i][0],
             l_legs[i][1], r_legs[i][0], r_legs[i][1]])
    return frames_numbers, flow_points

def getFramesImages(video_path, frames_numbers):
    frames_img = []
    cap = cv2.VideoCapture(video_path)
    for frame_nr in frames_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_nr))
        ret, frame = cap.read()

        frame = frames_preparation.prepare(frame)
        frames_img.append(frame)
    return frames_img

