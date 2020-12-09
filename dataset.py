import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import datetime
from collections import namedtuple
from torch.utils.data import Dataset
import torch
from config import SEGMENT_COUNT


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
            # offsets = np.multiply(list(range(SEGMENT_COUNT)), segment_duration) + np.random.randint(segment_duration, size=SEGMENT_COUNT)
            offsets = np.multiply(list(range(SEGMENT_COUNT)), segment_duration)
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


def create_annotations_from_directory_structure(videos_dir):
    verb_dirs = [o for o in os.listdir(videos_dir) if os.path.isdir(os.path.join(videos_dir,o))]
    print(verb_dirs)
    annots = []
    for verb in verb_dirs:
        verb_path = os.path.join(videos_dir, verb)
        filepaths = [os.path.join(verb_path, f) for f in os.listdir(verb_path) if os.path.isfile(os.path.join(verb_path, f))]
        for filepath in filepaths:
            annots.append((filepath, verb))
    
    df = pd.DataFrame(annots, columns=['filepath', 'verb_class'])
    classes = df['verb_class'].unique()

    return df, classes


def process_single_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    segment_duration = (frames_count + 1) // SEGMENT_COUNT
    frame_ids = (np.multiply(list(range(SEGMENT_COUNT)), segment_duration) + np.random.randint(segment_duration, size=SEGMENT_COUNT)).astype('int')
    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = cap.read()
        frames.append(Image.fromarray(image))

    return frames


def process_single_video_from_to(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(get_second_no_from_time(start_time) * fps)
    end_frame = int(get_second_no_from_time(end_time) * fps)
    segment_duration = (end_frame - start_frame + 1) // SEGMENT_COUNT
    # frame_ids = (np.multiply(list(range(SEGMENT_COUNT)), segment_duration) + np.random.randint(segment_duration, size=SEGMENT_COUNT)).astype('int')
    frame_ids = (np.multiply(list(range(SEGMENT_COUNT)), segment_duration)).astype('int')
    frames = []
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, image = cap.read()
        frames.append(Image.fromarray(image))
    
    return frames


class CustomTimestampsDataset(Dataset):
    """CustomTimestamps dataset."""

    def __init__(self, annots, root_dir, class_to_id, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annots = annots
        self.root_dir = root_dir
        self.class_to_id = class_to_id
        self.transform = transform

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames = process_single_video_from_to(os.path.join(self.root_dir, self.annots.iloc[idx, :]['filename'] + '.mp4'), \
                                        self.annots.iloc[idx, :]['start_time'], self.annots.iloc[idx, :]['end_time'])
    
        verb_class = str(self.annots.iloc[idx, :]['verb'])
        
        if self.transform:
            frames = self.transform(frames)
        sample = {'frames': frames, 'class_id': self.class_to_id[verb_class]}
        
        return sample


class ActionDataset(Dataset):
    """Action dataset."""

    def __init__(self, annots, root_dir, classes, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annots = annots
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_id, self.id_to_class = dict(), dict()
        for i, c in enumerate(classes):
            self.class_to_id.setdefault(c, i)
            self.id_to_class.setdefault(i, c)

        self.transform = transform

    def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames = process_single_video(self.annots.iloc[idx, 0])
        verb_class = self.annots.iloc[idx, 1]
        
        if self.transform:
            frames = self.transform(frames)
        sample = {'frames': frames, 'class_id': self.class_to_id[verb_class]}
        
        return sample














