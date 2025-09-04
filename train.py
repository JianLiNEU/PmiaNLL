# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import privacy_risks_evaluations.models_set.PreResNet as PreResNet
from privacy_risks_evaluations.models_set import PreResNet, Densenet, wrn, alexnet,Vgg19
import math
import torchvision.models as models
import random
import os
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from utils import *
import privacy_risks_evaluations.config as config
from privacy_risks_evaluations import dataset_process
import time
from PIL import Image
import torch
from torch.utils.data import Dataset


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    start_time = time.time()
    save_info_set=[]
    save_info_set.append("*"*20+ '\n')
    save_info_set.append(f"Training model: {args.model_name}"+ '\n')
    save_info_set.append(f"Training dataset: {args.dataset}"+ '\n')

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.seed:
        torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
        torch.manual_seed(args.seed)  # CPU seed
        if device == args.device:
            torch.cuda.manual_seed_all(args.seed)  # GPU seed

        random.seed(args.seed)  # python seed for image transformation

    #dataset solver
    clean_trainset, trainset, trainset_track, testset, num_classes=dataset_process.get_dataset(args)


    #########################################
    ###### Poisoning Attack Settings
    #########################################
    if args.dataset=="stl10":
        labels = get_data_stl10(trainset_track)
    else:
        labels = get_data_cifar_2(trainset_track)  # Backup correct labels

    
    if args.poison_strategy=="all":
        noisy_labels, idxs_to_change = add_noise_cifar_wo(trainset, args.noise_level, args.dataset)  # Directly modify labels in train_loader
        noisy_labels_track, idxs_to_change_track = add_noise_cifar_wo(trainset_track, args.noise_level, args.dataset)
    else:
        # Poison specific target class
        noisy_labels, idxs_to_change = poisoned_target_class(trainset, args.target_class, args.noise_level, args.dataset,attack_type=args.attack_type)  # Directly modify labels in train_loader
        noisy_labels_track, idxs_to_change_track = poisoned_target_class(trainset_track,  args.target_class, args.noise_level, args.dataset,attack_type=args.attack_type)

  

    clean_trainloader=torch.utils.data.DataLoader(clean_trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    train_loader_track = torch.utils.data.DataLoader(trainset_track, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    

    if args.model_name=="resnet18":
        model = PreResNet.ResNet18(dataset_name=args.dataset, num_classes=num_classes).to(device) # Model parameters
    elif args.model_name=="widresnet":
        model = wrn.wrn(dataset_name=args.dataset,  num_classes=num_classes).to(device)
    elif args.model_name=="resnet50":
        model = PreResNet.ResNet50(dataset_name=args.dataset,  num_classes=num_classes).to(device)
    elif args.model_name=="densenet121":
        model = Densenet.densenet(dataset_name=args.dataset, num_classes=num_classes).to(device)
    elif args.model_name=="alexnet":
        model = alexnet.AlexNet( dataset_name=args.dataset, num_classes=num_classes).to(device)
    elif args.model_name=="vgg16":
        model = Vgg19.vgg16_bn(  dataset_name=args.dataset, num_classes=num_classes).to(device)
    print(model)

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)



    # path where experiments are saved
    if args.clean_model:
        exp_path = os.path.join(model_dir, "clean_model_{}_{}".format(args.epochs,args.exp),)
    
    else:
        if args.Mixup == "None":
            exp_path = os.path.join(model_dir, "poison_model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.defence_method, args.Mixup, args.distribution, args.experiment_name, args.noise_level,args.epochs,args.poison_strategy, args.target_class, args.exp, args.synthemic_samples))
        else:
            exp_path = os.path.join(model_dir, "poison_model_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.defence_method, args.Mixup, args.distribution, args.experiment_name,args.noise_level,args.epochs,args.poison_strategy, args.target_class, args.exp, args.synthemic_samples))
    

    print(exp_path, os.path.exists(os.path.join(exp_path, 'model.pth')))
    if os.path.exists(os.path.join(exp_path, 'model.pth')):
        print(f" The target model is already trained {exp_path}")
        return
    
    if not os.path.isdir(exp_path):
         print(f" Create model folder: {exp_path}")
         os.makedirs(exp_path)

    # Initialize model parameters
    bmm_model=bmm_model_maxLoss=bmm_model_minLoss=cont=k = 0

    bootstrap_ep_std = milestones[0] + 5 + 1  # the +1 is because the conditions are defined as ">" or "<" not ">="


    guidedMixup_ep =1
  
    if args.defence_method=="dp":
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
        # Initialize PrivacyEngine
        privacy_engine = PrivacyEngine()
        errors = ModuleValidator.validate(model)
        if errors:
            model = ModuleValidator.fix(model)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        model, optimizer, train_loader  = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
         )




    if args.Mixup == 'Dynamic':
        bootstrap_ep_mixup = guidedMixup_ep + 5
    else:
        bootstrap_ep_mixup = milestones[0] + 5 + 1

    countTemp = 1
    temp_length = 200 - bootstrap_ep_mixup
    best_loss = np.inf
    patience_counter = 0
    beta_info_set=[]




    for epoch in range(1, args.epochs + 1):
        # train
        scheduler.step()
        if args.defence_method=="dp":
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, clean_trainloader, optimizer, epoch, synthemic_samples=args.synthemic_samples)
            epsilon = privacy_engine.accountant.get_epsilon(delta=1e-5)
            print(f"Epsilon: {epsilon}")

            
        if args.defence_method=="earlystopping":
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, clean_trainloader, optimizer, epoch, synthemic_samples=args.synthemic_samples)


        if args.clean_model:
            print('\t##### Clean model: Doing clean model with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, clean_trainloader, optimizer, epoch, synthemic_samples=args.synthemic_samples)
        ### Standard CE training (without mixup) ###
        elif args.Mixup == "None":
            print('\t##### poisoning model without defence Doing standard training with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, synthemic_samples=args.synthemic_samples)
            
        elif args.Mixup=="only_mixup":
            print('\t##### poisoning model with mixup defence Doing standard training with cross-entropy loss #####')
            loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, args.alpha)

        ### Mixup ###
        elif args.Mixup == "Static":
            alpha = args.alpha
            if epoch < bootstrap_ep_mixup:
                print('\t##### Only Mixup Static Doing NORMAL mixup for {0} epochs #####'.format(bootstrap_ep_mixup - 1))
                # Use mixup alone
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, alpha)

            else:
                if args.BootBeta == "Hard":
                    print("\t##### Static Hard Doing HARD BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch,\
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes, args.distribution)
                elif args.BootBeta == "Soft":
                    print("\t##### Static Soft Doing SOFT BETA bootstrapping and NORMAL mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                    loss_per_epoch, acc_train_per_epoch_i = train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, \
                                                                                    alpha, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, args.reg_term, num_classes, args.distribution)

        ## Dynamic Mixup ##
        elif args.Mixup == "Dynamic":
            alpha = args.alpha
            if epoch < guidedMixup_ep:
                print('\t##### Dynamic Doing NORMAL mixup for {0} epochs #####'.format(guidedMixup_ep - 1))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp(args, model, device, train_loader, optimizer, epoch, alpha)

            elif epoch < bootstrap_ep_mixup:
                print('\t##### Dynamic Doing Dynamic mixup from epoch {0} #####'.format(guidedMixup_ep))
                loss_per_epoch, acc_train_per_epoch_i = train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,\
                                                                        bmm_model_maxLoss, bmm_model_minLoss, args.distribution)
            else:
                # Automatically adjust alpha value
                print("\t##### Dynamic Going from SOFT BETA bootstrapping to HARD BETA with linear temperature and Dynamic mixup from the epoch {0} #####".format(bootstrap_ep_mixup))
                loss_per_epoch, acc_train_per_epoch_i, countTemp, k = train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, \
                                                                                                                epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, \
                                                                                                                countTemp, k, temp_length, args.reg_term, num_classes, args.distribution)

        # print("begin to train bmm model :...")
        if epoch > guidedMixup_ep-5 and args.Mixup in ["Static","Dynamic"]:
            epoch_losses_train, epoch_probs_train, argmaxXentropy_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = \
                track_training_loss(args, model, device, train_loader_track, epoch, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, idxs_to_change_track, save_dir=exp_path,distribution=args.distribution,cuda=args.device)
            beta_info_set.append(f"{epoch} : {bmm_model}\n")
            print(bmm_model_minLoss, bmm_model_maxLoss)
        if epoch%10 == 0:
            test_loss_per_epoch, test_acc_val_per_epoch_i, test_target_acc_val_per_epoch_i, = test_cleaning(args, model, device, test_loader)
            train_loss_per_epoch, train_acc_val_per_epoch_i, train_target_acc_val_per_epoch_i, = test_cleaning(args, model, device, clean_trainloader)




        best_acc_val=0
        if epoch == args.epochs:
            snapLast_test = 'last_epoch_%d_valLoss_%.5f\n_testAcc_%.5f_noise_%d_bestValLoss_%.5f\n' % (
            epoch, test_loss_per_epoch[-1], test_acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            snapLast_train = 'last_epoch_%d_valLoss_%.5f\n_trainAcc_%.5f_noise_%d_bestValLoss_%.5f\n' % (
            epoch, train_loss_per_epoch[-1], train_acc_val_per_epoch_i[-1], args.noise_level, best_acc_val)
            save_info_set.append(snapLast_test)
            save_info_set.append(snapLast_train)
            save_info_set.append("\n")
            torch.save(model.state_dict(), os.path.join(exp_path, 'model.pth'))
         
        
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    save_info_set.append(f"Training time: {end_time - start_time:.2f} seconds\n")
    save_info(exp_path, save_info_set,"Training_result")
    save_info(exp_path, beta_info_set,"beta_info")

if __name__ == '__main__':
    args=config.parser_setting()
    model_dir=f"./saved_models/{args.dataset}/{args.model_name}"
    print(args)
   
    main()