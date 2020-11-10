# action_recognition
Tasks for action recognition in KFC kitchens


Instructions for running code on this repository: 

Required: 
- Install Python 3.6
- pip install -r requirements.txt
- pip install git+https://github.com/wpwei/pretrained-models.pytorch.git@vision_bug_fix
- Download folder ##val from this link and put into the repository data/test.
- Download P01_11.tar from https://data.bris.ac.uk/data/dataset/b2db579428d236ae3f529ab05d8aa55e/resource/bff828c1-466f-4168-b092-2c0b536013bd?inner_span=True
and extract it to data/val folder under P01_11 name.

Command for running code:
##### python test.py --model TRN --dataset D1 --output_dir output (optional)

Possible choices for model are ('TSN', 'TRN', 'MTRN', 'TSM'), for dataset ('D1', 'D2').