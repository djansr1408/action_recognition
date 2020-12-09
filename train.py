from dataset import ActionDataset, CustomTimestampsDataset, create_annotations_from_directory_structure, create_inputs_from_annotations
import torch
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import json
import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--videos_dir', type=str,
                        help='Directory where data is stored.')
    parser.add_argument('--output_dir', type=str,  
                        help='Output dir where to save checkpoints.')
    parser.add_argument('--batch_size', type=int, default=4,  
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,   
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,   
                        help='Num epochs')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = torch.hub.load(config.REPO, 'TSN', config.CLASS_COUNTS, config.SEGMENT_COUNT, 'RGB',
                        base_model=config.BACKBONE_ARCH, 
                        pretrained='epic-kitchens', force_reload=True)
    
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


    # Prepare datasets
    df, classes = create_annotations_from_directory_structure(args.videos_dir)
    num_classes = len(classes)
    df_train, df_val = train_test_split(df, test_size=0.3, random_state=None)
    train_size, val_size = len(df_train), len(df_val)
    
    # dataset = ActionDataset(df, 'videos_filtered_4', classes, transform=transform)
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_set = ActionDataset(df_train, args.videos_dir, classes, transform=transform)
    val_set = ActionDataset(df_val, args.videos_dir, classes, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    with open(os.path.join(args.output_dir, "class_to_id.json"), "w") as f:
        json.dump(train_set.class_to_id, f)
    
    # Prepare test dataset
    annotations_df = pd.read_csv('data/test/action_recognition_labels_test.csv', sep=',')
    annotations_df = annotations_df[annotations_df['verb'].isna() == False]
    test_dataset = CustomTimestampsDataset(annotations_df, 'data/test/kfc_videos', train_set.class_to_id, transform=transform)
    test_size = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)


    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Cut last layer and replace with the one we need for our custom case (3, 4 verb classes)
    features_dim = model.fc_verb.in_features
    model.fc_verb = torch.nn.Linear(features_dim, num_classes)
    model = model.to(device)

    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    

    num_iter = 1
    average_loss = 0
    train_losses = []
    val_losses = []
    running_loss = 0.0
    writer = SummaryWriter()
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_true_train = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data['frames'], data['class_id']
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            average_loss += loss.item()
            epoch_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            num_true_train += torch.sum((predicted == labels).squeeze().int()).item()

            if num_iter % config.LOG_EVERY_N_ITERATIONS == 0:
                average_loss /= config.LOG_EVERY_N_ITERATIONS
                train_acc = num_true_train / train_size
                # print("Num iter: ", num_iter, "Average loss: ", average_loss)
                # print("Running loss: ", running_loss / num_iter)
                average_loss = 0
            num_iter += 1
        epoch_loss = epoch_loss / len(train_loader)  # div with num batches in epoch
        train_losses.append(epoch_loss)
        train_acc = num_true_train / train_size

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            num_true_val = 0
            num_val_examples = 0
            for i, data in enumerate(val_loader):
                inputs, labels = data['frames'], data['class_id']
                num_val_examples += len(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                num_true_val += torch.sum((predicted == labels).squeeze().int()).item()
            val_loss = val_loss / len(val_loader)
            val_acc = num_true_val / val_size
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1} Train loss: {epoch_loss:.5f}  Train acc: {train_acc:.4f}  Val loss: {val_loss:.4f}  Val acc: {val_acc:.4f}')
            writer.add_scalars('Loss', {'train': epoch_loss, 'val': val_loss}, epoch + 1)
            writer.add_scalars('Accuracy', {'train': train_acc, 'val': val_acc}, epoch + 1)

        torch.save(model, os.path.join(args.output_dir, f'finetuned_model_{epoch + 1}.pt'))
        print(f'Model saved at checkpoint: {args.output_dir}/finetuned_model_{epoch + 1}.pt')
 