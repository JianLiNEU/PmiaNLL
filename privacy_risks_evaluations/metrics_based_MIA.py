import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
from models_set import PreResNet, Mobilentv2, PreResNet,Vgg19, InceptionV3, Densenet
import math
import torchvision.models as models
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from PIL import Image
import config
import util
import dataset_process

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(self.data[idx])
        else:
            sample= self.data[idx]
        label =  self.labels[idx]
        return sample, label


def mem_nonmem_data(loader, target_class, num_samples, dataset_name,):
    if dataset_name=="stl10":
        labels_set = np.array([target_i for target_i in loader.labels])
        images_set = np.array([image_i.transpose(1, 2, 0) for image_i in loader.data])
    else:
        labels_set = np.array([target_i for target_i in loader.targets])
        images_set = np.array([image_i for image_i in loader.data])
    print(np.shape(images_set)) 
    if target_class==None:
        target_data =np.array(images_set)[-num_samples:]
        target_labels = np.array(labels_set)[-num_samples:]
    else:
        idxs=np.array(np.where(np.array(labels_set)==target_class))[0]
        target_data =np.array(images_set)[idxs][-num_samples:]
        target_labels = np.array(labels_set)[idxs][-num_samples:]

    return target_data, target_labels



def get_attack_data():
    print("load attack data: ")
    attack_train,attack_test, test_transform=dataset_process.get_dataset(args, attack_data=True)
    
    all_mem_data, all_mem_label=mem_nonmem_data(attack_train, target_class=None, num_samples=5000,dataset_name=args.dataset)
    all_nonmem_data, all_nonmem_label=mem_nonmem_data(attack_test, target_class=None, num_samples=5000,dataset_name=args.dataset)

    # Select attack data based on target class
    target_mem_data, target_mem_label=mem_nonmem_data(attack_train, target_class=args.target_class, num_samples=attack_train_amount+attack_test_amount,dataset_name=args.dataset)
    target_nonmem_data, target_nonmem_label=mem_nonmem_data(attack_test, target_class=args.target_class, num_samples=attack_train_amount+attack_test_amount,dataset_name=args.dataset)

    print("attack member data size: ", len(target_mem_data),"attack nonmember data size: ", len(target_mem_data))


    all_mem_data=DataLoader(CustomDataset(all_mem_data, all_mem_label, test_transform), batch_size=100, shuffle=False,)
    all_nonmem_data=DataLoader(CustomDataset(all_nonmem_data, all_nonmem_label, test_transform), batch_size=100, shuffle=False)
    target_mem_data=DataLoader(CustomDataset(target_mem_data, target_mem_label, test_transform), batch_size=100, shuffle=False,)
    target_nonmem_data=DataLoader(CustomDataset(target_nonmem_data, target_nonmem_label, test_transform), batch_size=100, shuffle=False)
    return target_mem_data, target_nonmem_data, all_mem_data, all_nonmem_data



def get_target_model(model_folder):
    if args.model_name=="resnet18":
        model = PreResNet.ResNet18(dataset_name=args.dataset, num_classes=num_classes).to(device) # Model parameters
    elif args.model_name=="inceptionv3":
        model = InceptionV3.get_inceptionv3( model_name=args.model_name, num_classes=num_classes).to(device)
    elif args.model_name=="densenet121":
        model = Densenet.get_densenet( model_name=args.model_name, num_classes=num_classes).to(device)
    elif args.model_name=="mobilenetv2":
        model = Mobilentv2.get_mobilenetv2( model_name=args.model_name, num_classes=num_classes).to(device)
    model_path = os.path.join(model_folder, 'model.pth')
    model.load_state_dict(torch.load(model_path))
    return model


def _Mentr(preds, y):
    y=F.one_hot(y, num_classes)
    preds=F.softmax(preds, dim=1)
    preds, y=preds.cpu().detach().numpy(), y.cpu().detach().numpy()
    fy = np.sum(preds*y, axis=1)
    fi = preds*(1-y)
    score = -(1-fy)*np.log(fy+1e-30)-np.sum(fi*np.log(1-fi+1e-30), axis=1)
    return score

def loss(preds, y):
    output = F.log_softmax(preds, dim=1)
    test_loss= F.nll_loss(output, y, reduction='none').cpu().detach().numpy()
    return test_loss

def get_membership_score(model, dataloder, num_classes):
    model.eval()
    score_set=[]
    for i, (input, label) in enumerate(dataloder):
        input, label = input.to(device), label.long().to(device)
        features=_Mentr(model(input), label)
        score_set.extend(features)

    return score_set


def get_data_accuracy(model, dataloder, target_class=0):
    model.eval()
    correct=0
    target_correct=0
    all_target_correct=0
    target_acc_val_per_epoch=0
    for i, (data, target) in enumerate(dataloder):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc_val_per_epoch = [np.array(100. * correct / len(dataloder.dataset))]
    return acc_val_per_epoch, target_acc_val_per_epoch


def metric_based_MIA(mem_features, nonmem_features):
    mem_label = np.ones(len(mem_features))
    nonmem_label = np.zeros(len(nonmem_features))

    mia_auc = roc_auc_score(np.r_[mem_label, nonmem_label],
                            -np.r_[mem_features, nonmem_features])
    fpr, tpr, _ = roc_curve(np.r_[mem_label, nonmem_label],
                            -np.r_[mem_features, nonmem_features])
    low_auc_01 = tpr[np.where(fpr < .01)[0][-1]]
    low_auc_001 = tpr[np.where(fpr < .001)[0][-1]]
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)

    print(f"attack_auc: {round(mia_auc*100,1)}")
    print(f"attack_acc: {round(acc*100,1)}")
    print(f"attack_tpr@1%fpr :{round(low_auc_01*100,1)}%")
    print(f"attack_tpr@0.1%fpr :{round(low_auc_001*100,1)}%")
    attack_info_set.append(f"attack auc score: {round(mia_auc*100,1)} \n")
    attack_info_set.append(f"attack balanced success score: {round(acc*100,1)} \n")
    attack_info_set.append(f"attack tpr@1%fpr: {round(low_auc_01*100,1)}% \n")
    attack_info_set.append(f"attack tpr@0.1%fpr: {round(low_auc_001*100,1)}% \n")




def main():
    # Obtain attack data
    mem_dataset, nonmem_dataset,train_dataset, test_dataset=get_attack_data()
    
    # Poisoned model with defense
    # poison_model_path="../saved_models/noise_models_PreResNet18_runs/poison_model_50.0_300_1000_0/poison_model_with_defence_last_epoch_300_valLoss_0.47009_valAcc_90.96000_noise_50_bestValLoss_0.00000.pth"
    
    # Poisoned model without defense
    # poison_model_path="../saved_models/noise_models_PreResNet18_runs/poison_model_50.0_300_1000_0/poison_model_without_defence_last_epoch_300_valLoss_0.56539_valAcc_86.41000_noise_50_bestValLoss_0.00000.pth"

    if args.clean_model:
        exp_path = os.path.join(model_dir, "clean_model_{}_{}".format(args.epochs, args.exp))
    elif args.Mixup == "None":
        exp_path = os.path.join(model_dir, "poison_model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.defence_method, args.Mixup, args.distribution, args.experiment_name, args.noise_level,args.epochs,args.poison_strategy, args.target_class, args.exp, args.synthemic_samples))
    else:
        exp_path = os.path.join(model_dir, "poison_model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.defence_method, args.Mixup, args.distribution, args.experiment_name,args.noise_level,args.epochs,args.poison_strategy, args.target_class, args.exp, args.synthemic_samples))
    print(f"attack poisoned model: {exp_path}")
    attack_info_set.append(f"attack dir: {exp_path}  \n")

    target_model=get_target_model(model_folder=exp_path)
    target_model.eval()
    # Get model prediction accuracy
    trainacc, target_trainacc,=get_data_accuracy(target_model, mem_dataset)
    testacc, target_testacc=get_data_accuracy(target_model, nonmem_dataset)

    all_train, target_alltest=get_data_accuracy(target_model, train_dataset)
    all_test, target_alltest=get_data_accuracy(target_model, test_dataset)

    print("all data train accuracy", np.round(all_train[0],1), "\nall data test accuracy", np.round(all_test[0], 1), "\ntarget class train accuracy: ", np.round(trainacc[0],1), "\ntarget class test accuracy:", np.round(testacc[0],1))
    attack_info_set.append(f"all data train accuracy: {np.round(all_train[0], 1)} \nall data test accuracy: {np.round(all_test[0],1)} \ntarget class train accuracy: {np.round(trainacc[0],1)} \ntarget class test accuracy: {np.round(testacc[0],1)}  \n")
    attack_info_set.append(f"{np.round(all_train[0],1)} | {np.round(all_test[0], 1)} | {np.round(trainacc[0], 1)} | {np.round(testacc[0],1)}  \n")
    if args.synthemic_samples==1:
        # Get membership score
        mem_features=get_membership_score(target_model, mem_dataset, num_classes)
        nonmem_features=get_membership_score(target_model, nonmem_dataset, num_classes)
    else:    
        # Get membership score
        mem_features=get_membership_score(target_model, mem_dataset, num_classes)
        nonmem_features=get_membership_score(target_model, nonmem_dataset, num_classes)

    # Choose MIA method
    metric_based_MIA(mem_features, nonmem_features)

    # Save attack results
    util.save_info(exp_path, attack_info_set, "attack_result")








if __name__ == '__main__':
    args=config.parser_setting()
    model_dir=f"./saved_models/{args.dataset}/{args.model_name}"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.dataset=="CIFAR10" or args.dataset=="MNIST":
        num_classes=10
        attack_train_amount=500
        attack_test_amount=500
    elif args.dataset=="CIFAR100":
        num_classes=100
        attack_train_amount=50
        attack_test_amount=50
    elif args.dataset=="stl10":
        num_classes=10
        attack_train_amount=50
        attack_test_amount=50
    elif args.dataset=="tinyimagenet":
        num_classes=200
        attack_train_amount=50
        attack_test_amount=50
        
        
    attack_info_set=[] # Store attack results

    main()