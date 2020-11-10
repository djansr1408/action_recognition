# action_recognition
Tasks for action recognition in KFC kitchens


Instructions for running code on this repository: 

Required: 
- Install Python 3.6
- pip install -r requirements.txt
- pip install git+https://github.com/wpwei/pretrained-models.pytorch.git@vision_bug_fix
- Download folder ##data from this link and put into the repository


Command for running code:
##### python test.py --model TRN --dataset D1 --output_dir output (optional)

Possible choices for model are ('TSN', 'TRN', 'MTRN', 'TSM'), for dataset ('D1', 'D2').
