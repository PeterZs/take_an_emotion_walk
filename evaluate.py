import numpy as np
from tqdm import tqdm
import sklearn.metrics as skm
import argparse
import sys
import subprocess

# torch
import torch
import torch.nn.functional as F
from data import create_loader
from models.model_loader import load_dgnn, load_stgcn

## default weights paths
DGNN_WEIGHTS_PATH = "weights/dgnn_weights.pt"
STGCN_WEIGHTS_PATH = "weights/stgcn_500_5.pt"

parser = argparse.ArgumentParser(
    description='Action recognition with emotion labels eval')

parser.add_argument(
    '--stgcn', action="store_true", help='evaluate stgcn')
parser.add_argument(
    '--dgnn', action="store_true", help='evaluate dgnn')
parser.add_argument(
    '--lstm', action="store_true", help='evaluate lstm network')
parser.add_argument(
    '--step', action="store_true", help='evaluate step')
parser.add_argument(
    '--taew', action="store_true", help='evaluate taew')


def to_multi_hot(labels, threshold=0.25):
    '''
    creates multi-hot vector from output vector. 
    1 if vector entry is > .25 else 0
    '''

    labels_out = np.zeros_like(labels)
    # labels_out[np.arange(len(labels)), labels.argwhere(1)] = 1
    hot_idx = np.argwhere(labels > threshold)
    labels_out[hot_idx[:, 0], hot_idx[:, 1]] = 1
    return labels_out


def calculate_metrics(y_true, y_pred, thres=0.65):
    '''
    caluclates average precisions, mean average precisions, and mean f1 scores 
    '''
    y_true = to_multi_hot(y_true)
    ap = skm.average_precision_score(y_true, y_pred, average=None)
    nans = np.sum(np.isnan(ap))
    mean_ap = np.sum(np.nan_to_num(ap)) / (len(ap) - nans)
    ap = np.nan_to_num(ap)
    y_pred_multi_hot = to_multi_hot(y_pred, threshold=thres)
    _, _, f1_score, _ = skm.precision_recall_fscore_support(y_true, y_pred_multi_hot, average=None)
    return ap, mean_ap, np.mean(f1_score)


def eval(model, loader, use_bones=False, taew=False):
    '''
    runs evaluation for a specified model and loader.
    use_bones=True if using joints data (graph networks)
    hap = True if using taew NOTE: might depend on implementation
    '''

    criterion = F.binary_cross_entropy_with_logits

    for joint_data, bone_data, label in tqdm(loader):

        with torch.no_grad():
            if use_bones:
                joint_data = joint_data.float().cuda()
                bone_data = bone_data.float().cuda()
                label = label.cuda()
                pred = model(joint_data.unsqueeze(-1), bone_data.unsqueeze(-1))
            else:
                pred = model(joint_data.unsqueeze(-1))

            loss = criterion(pred, label)
            ap, map, f1 = calculate_metrics(label.detach().cpu().numpy(), pred.detach().cpu().numpy())

    print("loss: ", loss.item(), "ap: ", ap, " map: ", map, " f1: ", f1)


if __name__ == '__main__':
    args = parser.parse_args()

    # dataset
    print("Loading data...")

    if args.stgcn:
        print("Evaluating stgcn")
        val_loader = create_loader(args)
        stgcn_model = load_stgcn(STGCN_WEIGHTS_PATH)
        eval(stgcn_model, val_loader)

    elif args.dgnn:
        print("Evaluating dgnn")
        val_loader = create_loader(args)
        dgnn_model = load_dgnn(DGNN_WEIGHTS_PATH)
        eval(dgnn_model, val_loader, use_bones=True)

    elif args.lstm:
        print("Evaluating lstm")
        subprocess.call(["python3", "lstm/main.py"])

    elif args.step:
        print("Evaluating step")
        subprocess.call(["python3", "step/main.py"])

    elif args.taew:
        print("Evaluating taew")
        subprocess.call(["python3", "taew_net/main.py"])

    else:
        print("INVALID OPTION!")
