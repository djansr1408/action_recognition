# action_recognition
Tasks for action recognition in KFC kitchens


Instructions for running code on this repository: 

Required: 
- Install Python 3.6
- pip install -r requirements.txt
- Install pytorch (depending on your configuration)
- pip install git+https://github.com/wpwei/pretrained-models.pytorch.git@vision_bug_fix
- Download kfc_videos.zip from this link https://drive.google.com/file/d/1gT6NazM763AvtLnNPDVvNq5bZWPj4HbV/view?usp=sharing and extract into the repository ./data/test so to get ./data/test/kfc_videos.
- Download P01_11.tar from https://data.bris.ac.uk/data/dataset/b2db579428d236ae3f529ab05d8aa55e/resource/bff828c1-466f-4168-b092-2c0b536013bd?inner_span=True
and extract it to ./data/val folder under P01_11 name.
- Download video_segments.zip from https://drive.google.com/file/d/1F4N4yQFgeenoyfNcitTH-0NjNhNqV7sC/view?usp=sharing and extract it to ./data/ folder under video_segments name.

Quick check: The structure of the folders should look like this:
- data 
  - test 
    - *.csv
    - kfc_videos/*.mp4
  - val
    - *.csv
    - P01_11/*.jpg
  - video_segments
    - videos_60   [take, mix, put, move]
    - videos_120   [take, mix, put, move]

Command for running code:
##### python test.py --model TRN --dataset D1 --output_dir output (optional)

Possible choices for model are ('TSN', 'TRN', 'MTRN', 'TSM'), for dataset ('D1', 'D2').

For train process there is the script train.py with arguments:
##### python train.py --videos_dir data/video_segments/videos_60 --output_dir ckpt_60 --lr 0.00097 --num_epochs 10

For evaluation on fine-tuned model there is the script evaluate.py with arguments:
##### python evaluate.py --checkpoint_dir ckpt_60 --model_number 5

For evaluation on the original EPIC-KITCHEN model there is the script test.py with arguments:
##### python test.py --model 'TSN' --dataset 'D3' --output_dir output

