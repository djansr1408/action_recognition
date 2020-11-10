import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import torch.hub
from torchvision.transforms import Compose
import time
from PIL import Image
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
import argparse

from dataset import load_segment, create_inputs_from_annotations, create_inputs_from_annotations_val, print_statistics_from_videos
from transforms import GroupScale, GroupCenterCrop, GroupOverSample, Stack, ToTorchFormatTensor, GroupNormalize


repo = 'epic-kitchens/action-models'


class_counts = (125, 352)
segment_count = 8
base_model = 'resnet50'
backbone_arch = base_model
batch_size = 1
segment_count = 8
snippet_length = 1  # Number of frames composing the snippet, 1 for RGB, 5 for optical flow
snippet_channels = 3  # Number of channels in a frame, 3 for RGB, 2 for optical flow
height, width = 224, 224
crop_count = 1


noun_df = pd.read_csv('material/EPIC_noun_classes.csv', sep=',')
id_to_noun = dict()
noun_to_id = dict()
for i, row in noun_df.iterrows():
    id_to_noun.setdefault(row['noun_id'], row['class_key'])
    nouns_str = str(row['nouns']).replace("\'", "\"")
    nouns = json.loads(nouns_str)
    for noun in nouns:
        noun_to_id.setdefault(noun.strip(), row['noun_id'])


verb_df = pd.read_csv('material/EPIC_verb_classes.csv', sep=',')
id_to_verb = dict()
verb_to_id = dict()
for i, row in verb_df.iterrows():
    id_to_verb.setdefault(row['verb_id'], row['class_key'])
    verbs_str = str(row['verbs']).replace("\'", "\"")
    verbs = json.loads(verbs_str)
    for verb in verbs:
        verb_to_id.setdefault(verb.strip(), row['verb_id'])


def measure_accuracy(top_n_list, labels):
    """
        top_n_list is a list of np.arrays of len 5
        labels is list of class_id
    """
    correct_1 = 0
    correct_5 = 0
    for i in range(len(labels)):
        correct_1 += 1 if top_n_list[i][0] == labels[i] else 0
        correct_5 += np.sum(top_n_list[i] == labels[i])
    
    acc_1 = correct_1 / len(labels)
    acc_5 = correct_5 / len(labels)

    return acc_1, acc_5


def plot_class_distribution(class_dict, filepath, image_name):
    classes_sorted = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
    classes = [t[0] for t in classes_sorted]
    counts = [t[1] for t in classes_sorted]
    fig, ax = plt.subplots(figsize=(10, 7))
    g = sns.barplot(x=classes, y=counts, ax=ax)
    g.set_xticklabels(g.get_xticklabels(), rotation=60)
    plt.title(image_name)
    plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--model', type=str, default='TSN', 
                        choices=['TSN', 'TRN', 'MTRN', 'TSM'],
                        help='Type of the model.')
    parser.add_argument('--dataset', type=str, default='D1', 
                        choices=['D1', 'D2'], 
                        help='D1 (unseen), D2 (seen)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory where to save predictions .csv file')
    
    args = parser.parse_args()
    if args.model == 'TSN':
        model = torch.hub.load(repo, 'TSN', class_counts, segment_count, 'RGB',
                            base_model=base_model, 
                            pretrained='epic-kitchens', force_reload=True)
    elif args.model == 'TRN':
        model = torch.hub.load(repo, 'TRN', class_counts, segment_count, 'RGB',
                            base_model=base_model, 
                            pretrained='epic-kitchens')
    elif args.model == 'MTRN':
        model = torch.hub.load(repo, 'MTRN', class_counts, segment_count, 'RGB',
                            base_model=base_model, 
                            pretrained='epic-kitchens')
    elif args.model == 'TSM': 
        model = torch.hub.load(repo, 'TSM', class_counts, segment_count, 'RGB',
                            base_model=base_model, 
                            pretrained='epic-kitchens')
    else:
        print("Bad input model specified.")
        quit(0)
    
    # Transform part initialize
    if crop_count == 1:
        cropping = Compose([
            GroupScale(model.scale_size),
            GroupCenterCrop(model.input_size),
        ])
    elif crop_count == 10:
        cropping = GroupOverSample(model.input_size, model.scale_size)
    else:
        raise ValueError("Only 1 and 10 crop_count are supported while we got {}".format(crop_count))

    transform = Compose([
        cropping,
        Stack(roll=backbone_arch == 'BNInception'),
        ToTorchFormatTensor(div=backbone_arch != 'BNInception'),
        GroupNormalize(model.input_mean, model.input_std),
    ])
    
    # Load data
    if args.dataset == 'D1':
        annotations_df = pd.read_csv('data/test/action_recognition_labels.csv', sep=',')
        list_of_input_tuples = create_inputs_from_annotations(annotations_df, 'data/test/kfc_videos', verb_to_id, noun_to_id)
        output_dir = 'data/test'
    elif args.dataset == 'D2':
        annotations_df = pd.read_csv('data/val/EPIC_100_validation.csv', sep=',')
        list_of_input_tuples = create_inputs_from_annotations_val(annotations_df, videos_dir='data/val/P01_11')
        output_dir = 'data/val'
    else:
        print("Bad dataset input specified.")
        quit(0)

    # print_statistics_from_videos(annotations_df)

    naration_ids = []
    verb_predictions = []
    noun_predictions = []

    verb_dict = dict()
    noun_dict = dict()
    verb_pred_dict = dict()
    noun_pred_dict = dict()
    softmax = torch.nn.Softmax(dim=1)
    loss = torch.nn.CrossEntropyLoss()
    verb_top_n_predictions = []
    noun_top_n_predictions = []
    verb_labels = []
    noun_labels = []
    verb_pred = []
    noun_pred = []
    verb_loss = []
    noun_loss = []
    for input_tuple in list_of_input_tuples:
        start_time = time.time()
        print("narration id: ", input_tuple.narration_id, "verb: ", id_to_verb[input_tuple.verb_class], "   noun: ", id_to_noun[input_tuple.noun_class])
        verb_labels.append(input_tuple.verb_class)
        noun_labels.append(input_tuple.noun_class)

        verb_dict.setdefault(id_to_verb[input_tuple.verb_class], 0)
        verb_dict[id_to_verb[input_tuple.verb_class]] += 1
        noun_dict.setdefault(id_to_noun[input_tuple.noun_class], 0)
        noun_dict[id_to_noun[input_tuple.noun_class]] += 1

        frames = input_tuple.frames
        inputs = transform(frames)
        inputs = inputs.unsqueeze(0) # add batch_size dimension

        # Classify
        verb_logits, noun_logits = model(inputs)
        
        verb_loss.append(loss(verb_logits, torch.tensor([input_tuple.verb_class])).detach().numpy())
        noun_loss.append(loss(noun_logits, torch.tensor([input_tuple.noun_class])).detach().numpy())

        verb_logits = softmax(verb_logits)
        noun_logits = softmax(noun_logits)

        verb_logits = verb_logits.detach().numpy().flatten()
        verb_pred_classes = np.argsort(verb_logits)[::-1]
        for verb_id in verb_pred_classes[:5]:
            verb_pred_dict.setdefault(id_to_verb[verb_id], 0)
            verb_pred_dict[id_to_verb[verb_id]] += 1

        noun_logits = noun_logits.detach().numpy().flatten()
        noun_pred_classes = np.argsort(noun_logits)[::-1]
        for noun_id in noun_pred_classes[:5]:
            noun_pred_dict.setdefault(id_to_noun[noun_id], 0)
            noun_pred_dict[id_to_noun[noun_id]] += 1

        verb_top_n_predictions.append(verb_pred_classes[:5])
        noun_top_n_predictions.append(noun_pred_classes[:5])

        naration_ids.append(input_tuple.narration_id)
        
        verb_pred.append(verb_pred_classes[0])
        noun_pred.append(noun_pred_classes[0])

        verb_predictions.append(id_to_verb[verb_pred_classes[0]])
        noun_predictions.append(id_to_noun[noun_pred_classes[0]])
    

    print("verb loss: ", np.mean(np.array(verb_loss)))
    print("noun loss: ", np.mean(np.array(noun_loss)))

    f1_verb = f1_score(verb_labels, verb_pred, average='weighted')
    print("f1 weighted verb: ", f1_verb)

    f1_noun = f1_score(noun_labels, noun_pred, average='weighted')
    print("f1 weighted noun: ", f1_noun)

    verb_acc_1, verb_acc_5 = measure_accuracy(verb_top_n_predictions, verb_labels)
    noun_acc_1, noun_acc_5 = measure_accuracy(noun_top_n_predictions, noun_labels)
    print("verb acc 1: ", verb_acc_1, "verb acc 5: ", verb_acc_5)
    print("noun acc 1: ", noun_acc_1, "noun acc 5: ", noun_acc_5)

    plot_class_distribution(verb_dict, os.path.join(output_dir, 'verb_labels.png'), 'verb labels')
    plot_class_distribution(noun_dict, os.path.join(output_dir, 'noun_labels.png'), 'noun labels')

    # print("verb_dict: ", verb_dict)
    # print("noun_dict: ", noun_dict)

    plot_class_distribution(verb_pred_dict, os.path.join(output_dir, 'verb_predicted.png'), 'verb predicted')
    plot_class_distribution(noun_pred_dict, os.path.join(output_dir, 'noun_predicted.png'), 'noun predicted')

    # print("verb_dict_pred: ", verb_pred_dict)
    # print("noun_dict_pred: ", noun_pred_dict)

    pred_df = pd.DataFrame(naration_ids, columns=['narration_id'])
    pred_df['verb_class_pred'] = verb_predictions
    pred_df['noun_class_pred'] = noun_predictions

    annotations_df = annotations_df.merge(pred_df, left_on='narration_id', right_on='narration_id', how='inner')

    if args.output_dir is None:
        annotations_df.to_csv(os.path.join(output_dir, 'annotations_predicted.csv'), sep=',', index=False)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        annotations_df.to_csv(os.path.join(args.output_dir, 'annotations_predicted.csv'), sep=',', index=False)