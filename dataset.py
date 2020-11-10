import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import datetime
from collections import namedtuple


SEGMENT_COUNT = 8
# frame_width, frame_height = 480, 854
IMAGE_WIDTH, IMAGE_HEIGHT = 854, 480
# IMAGE_WIDTH, IMAGE_HEIGHT = 224, 224

def load_segment(segment_filepath):
    cap = cv2.VideoCapture(segment_filepath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps: ", fps)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        quit(0)
    frame_id = 0
    frames = []
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            # cv2.imshow(f'Frame_', frame)
            # cv2.imwrite(f'data/frame_{frame_id}.jpg', frame)
            img = Image.fromarray(frame)
            frames.append(img)
            frame_id += 1
        # Break the loop
        else: 
            break

    return frames


def get_second_no_from_time(t):
    datetime_t = datetime.datetime.strptime(t, '%H:%M:%S')
    num_seconds = datetime_t.minute * 60 + datetime_t.second

    return num_seconds


def print_statistics_from_videos(df):
    df['start_sec'] = df['start_time'].apply(lambda x: int(get_second_no_from_time(x)))
    df['end_sec'] = df['end_time'].apply(lambda x: int(get_second_no_from_time(x)))
    df['diff'] = df['end_sec'] - df['start_sec']
    df_grouped = df.groupby('filename')['diff'].mean().reset_index()
    
    print(df_grouped)


def create_inputs_from_annotations(df, videos_dir, verb_to_id, noun_to_id):
    df = df.sort_values(by=['filename', 'start_time'])
    print("len: ", len(df['filename'].unique()))
    list_of_inputs = [] # one input is list of 8 frames
    FrameRow = namedtuple('FrameRow', 'narration_id frames verb_class noun_class')
    for filename in df['filename'].unique():
        print("Filename: ", filename + '.mp4')
        df_video = df[df['filename'] == filename]
        cap = cv2.VideoCapture(os.path.join(videos_dir, filename + '.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps: ", fps)
        df_video['start_frame'] = df_video['start_time'].apply(lambda x: int(get_second_no_from_time(x) * fps))
        df_video['end_frame'] = df_video['end_time'].apply(lambda x: int(get_second_no_from_time(x) * fps))

        current_id = 0
        for i, row in df_video.iterrows():
            frames = []
            segment_duration = (row['end_frame'] - row['start_frame'] + 1) // SEGMENT_COUNT
            offsets = np.multiply(list(range(SEGMENT_COUNT)), segment_duration) + np.random.randint(segment_duration, size=SEGMENT_COUNT)
            frame_ids = row['start_frame'] + offsets
            for frame_id in frame_ids:
                while current_id < frame_id:
                    ret, frame = cap.read()
                    current_id += 1
                img = Image.fromarray(frame)
                frames.append(img)
            list_of_inputs.append(FrameRow(row['narration_id'], frames, verb_to_id[row['verb']], noun_to_id[row['noun']]))
        
        cap.release()

    return list_of_inputs


def create_inputs_from_annotations_val(df, videos_dir):
    list_of_inputs = []
    FrameRow = namedtuple('FrameRow', 'narration_id frames verb_class noun_class')
    num_inputs = 1
    for i, row in df.iterrows():
        if num_inputs > 100:
            break
        if row['stop_frame'] > 33000:
            continue
        segment_duration = (row['stop_frame'] - row['start_frame'] + 1) // SEGMENT_COUNT
        offsets = np.multiply(list(range(SEGMENT_COUNT)), segment_duration) + np.random.randint(segment_duration, size=SEGMENT_COUNT)
        frame_ids = row['start_frame'] + offsets
        frames = []
        for frame_id in frame_ids:
            img = Image.open(os.path.join(videos_dir, "frame_{:010d}.jpg".format(frame_id)))
            frames.append(img)
        list_of_inputs.append(FrameRow(row['narration_id'], frames, row['verb_class'], row['noun_class']))
        num_inputs += 1

    return list_of_inputs
