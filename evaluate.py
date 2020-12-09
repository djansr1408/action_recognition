import torch
from torchvision.transforms import Compose
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
from dataset import CustomTimestampsDataset, create_inputs_from_annotations
import pandas as pd
import numpy as np
import config
import json
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory where data is stored.')
    parser.add_argument('--model_number', type=int,
                        help='Model number based on epoch.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(config.REPO, 'TSN', config.CLASS_COUNTS, config.SEGMENT_COUNT, 'RGB',
                        base_model=config.BACKBONE_ARCH, 
                        pretrained='epic-kitchens', force_reload=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Transform part initialize
    crop_count = 1
    if crop_count == 1:
        cropping = Compose([
            GroupScale(config.SCALE_SIZE),
            GroupCenterCrop(config.INPUT_SIZE),
        ])
    elif crop_count == 10:
        cropping = GroupOverSample(config.INPUT_SIZE, config.SCALE_SIZE)
    else:
        raise ValueError("Only 1 and 10 crop_count are supported while we got {}".format(crop_count))

    transform = Compose([
        cropping,
        Stack(roll=config.BACKBONE_ARCH == 'BNInception'),
        ToTorchFormatTensor(div=config.BACKBONE_ARCH != 'BNInception'),
        GroupNormalize(config.INPUT_MEAN, config.INPUT_STD),
    ])

    with open(os.path.join(args.checkpoint_dir, "class_to_id.json"), "r") as f:
        class_to_id = json.load(f)

    # Cut last layer and replace with the one we need for our custom case (3, 4 verb classes)
    features_dim = model.fc_verb.in_features
    print('features_dim: ', features_dim)
    model.fc_verb = torch.nn.Linear(features_dim, len(class_to_id))
    model = model.to(device)
    
    model = torch.load(os.path.join(args.checkpoint_dir, f'finetuned_model_{args.model_number}.pt'))
    model.eval()

    # Prepare test dataset
    annotations_df = pd.read_csv('data/test/action_recognition_labels_test.csv', sep=',')
    annotations_df = annotations_df[annotations_df['verb'].isna() == False]
    test_dataset = CustomTimestampsDataset(annotations_df, 'data/test/kfc_videos', class_to_id, transform=transform)
    test_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=1)

    np.random.seed(12)
    with torch.no_grad():
        num_true_test = 0
        num_test_examples = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data['frames'], data['class_id']
            num_test_examples += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            num_true_test += torch.sum((predicted == labels).squeeze().int()).item()
        test_acc = num_true_test / num_test_examples
        print("Test acc: ", test_acc)
        print("Num test ex: ", num_test_examples)