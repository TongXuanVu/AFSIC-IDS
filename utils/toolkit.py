import os
import numpy as np
import torch
import  json
from enum import Enum
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ConfigEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = calculate_metrics(y_true, y_pred)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "acc_{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        if len(idxes) > 0:
            all_acc[label] = np.around(
                (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
            )
        else:
            all_acc[label] = 0.0

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old_acc"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new_acc"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    return all_acc


def calculate_metrics(y_true, y_pred):
    # Confusion matrix để tính FPR
    # FPR = FP / (FP + TN) cho lớp Benign (thường là lớp 0)
    cm = confusion_matrix(y_true, y_pred)
    fpr = 0.0
    if cm.shape[0] > 0:
        tn = cm[0, 0]
        fp = np.sum(cm[0, 1:])
        if (fp + tn) > 0:
            fpr = (fp / (fp + tn)) * 100

    return {
        "total": np.around(accuracy_score(y_true, y_pred) * 100, decimals=2),
        "fpr": np.around(fpr, decimals=2),
        
        "precision_micro": np.around(precision_score(y_true, y_pred, average="micro", zero_division=0) * 100, decimals=2),
        "precision_macro": np.around(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, decimals=2),
        "precision_weighted": np.around(precision_score(y_true, y_pred, average="weighted", zero_division=0) * 100, decimals=2),
        
        "recall_micro": np.around(recall_score(y_true, y_pred, average="micro", zero_division=0) * 100, decimals=2),
        "recall_macro": np.around(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, decimals=2),
        "recall_weighted": np.around(recall_score(y_true, y_pred, average="weighted", zero_division=0) * 100, decimals=2),
        
        "f1_micro": np.around(f1_score(y_true, y_pred, average="micro", zero_division=0) * 100, decimals=2),
        "f1_macro": np.around(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, decimals=2),
        "f1_weighted": np.around(f1_score(y_true, y_pred, average="weighted", zero_division=0) * 100, decimals=2),
    }


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1: 
        fc_weight = model._network.fc.weight.data    
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")

def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network   
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)