import subprocess
import cv2
import os
import argparse
import json


def split_video_to_parts(video_path, out_dir, segment_len_frames, overlap_percent, index_from=0):
    filename = os.path.basename(video_path)[:-4]
    dir_path = os.path.join(out_dir, filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        return 0
    v = cv2.VideoCapture(video_path)
    v.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    duration = v.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = v.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = v.get(cv2.CAP_PROP_FPS)
    time_len = round(segment_len_frames / fps, 2) 

    frame_from = 0
    frame_to = frame_from + segment_len_frames
    num_video_parts = index_from
    while frame_to <= frame_count:
        start_time = round(frame_from / fps, 2)
        p = subprocess.call(f'ffmpeg -i {video_path} -ss {start_time} -t {time_len} -strict -2 {dir_path}/part_{num_video_parts}.mp4', shell=True) 
        num_video_parts += 1
        frame_from += int((1 - overlap_percent) * segment_len_frames)
        frame_to = frame_from + segment_len_frames
        # print("frame from: ", frame_from)
    
    return num_video_parts - index_from

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--video_dir', type=str, 
                        help='Directory containing mp4 files.')
    parser.add_argument('--output_dir', type=str, 
                        help='Directory where to save video parts.')
    parser.add_argument('--segment_len', type=int,
                        help='Segment length in frames.')
    parser.add_argument('--overlap', type=float, default=0.3, 
                        help='Overlap between segments.')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config = dict()
    config['segment_len'] = args.segment_len
    config['overlap'] = args.overlap
    with open(os.path.join(args.output_dir, 'config.json'), "w") as write_file:
        json.dump(config, write_file)

    filepaths = [os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir) if (os.path.isfile(os.path.join(args.video_dir, f)) and f.endswith('.mp4'))]
    print(filepaths)
    index_from = 0
    for filepath in filepaths:
        print("Filepath: ", filepath)
        num_parts = split_video_to_parts(filepath, args.output_dir, args.segment_len, args.overlap, index_from)
        index_from += num_parts + 1

