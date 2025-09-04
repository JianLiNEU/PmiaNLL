from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture as GMM
from sklearn import preprocessing as preprocessing
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA

import NDSS_2025_facilitate_PMIA as syn
######################### Get data and noise adding ##########################
def get_data_cifar(loader):
    data = loader.sampler.data_source.train_data.copy()
    labels = loader.sampler.data_source.train_labels
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return (data, labels)

def get_data_cifar_2(loader):
    labels = loader.targets
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return labels

def get_data_stl10(loader):
    labels = loader.labels
    labels = torch.Tensor(labels[:]).long() # this is to copy the list
    return labels

#Noise without the sample class
def add_noise_cifar_wo(loader, noise_percentage = 20, dataset_name="cifar10"):
    torch.manual_seed(2)
    np.random.seed(42)
    if dataset_name=="stl10":
        noisy_labels = [sample_i for sample_i in loader.labels]
        images = [sample_i for sample_i in loader.data]
    else:
        noisy_labels = [sample_i for sample_i in loader.targets]
        images = [sample_i for sample_i in loader.data]

    probs_to_change = torch.randint(100, (len(noisy_labels),))
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))
    print("Noise samples in trianing : ", torch.sum(idx_to_change),"noisy level: ", percentage_of_bad_labels)
    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1:
            set_labels = list(
                set(range(10)) - set([label_i]))  # this is a set with the available labels (without the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]

    if dataset_name=="stl10":
        loader.data = images
        loader.labels=noisy_labels
    else:
        loader.data = images
        loader.targets=noisy_labels


    return noisy_labels, idx_to_change

#Noise with the sample class (as in Re-thinking generalization )
def add_noise_cifar_w(loader, noise_percentage = 20, dataset_name="cifar10"):
    """
        基于random随机选择label
        基于label的投毒样本的选择
    """
    
    torch.manual_seed(2) #固定seed
    np.random.seed(42)
    if dataset_name=="stl10":
        noisy_labels = [sample_i for sample_i in loader.labels]
        images = [sample_i for sample_i in loader.data]
    else:
        noisy_labels = [sample_i for sample_i in loader.targets]
        images = [sample_i for sample_i in loader.data]

    probs_to_change = torch.randint(100, (len(noisy_labels),))#随机选择一个数0-100 对于每个图片
    idx_to_change = probs_to_change >= (100.0 - noise_percentage)# 当这个数大于80时，就改变这个图片的label
    percentage_of_bad_labels = 100 * (torch.sum(idx_to_change).item() / float(len(noisy_labels)))
    for n, label_i in enumerate(noisy_labels):
        if idx_to_change[n] == 1: #如果这个图片的label要改变
            set_labels = list(set(range(10)))  # this is a set with the available labels (with the current label)
            set_index = np.random.randint(len(set_labels))
            noisy_labels[n] = set_labels[set_index]
    if dataset_name=="stl10":
        loader.data = images
        loader.labels=noisy_labels
    else:
        loader.data = images
        loader.targets=noisy_labels

    return noisy_labels



def poisoned_target_class(loader, target_class, noise_level,dataset_name):
    """
        poisoning membership inference attacks against poisoned models with special target class 

    """
    if dataset_name=="stl10":
        noisy_labels = [sample_i for sample_i in loader.labels]
        images = [sample_i for sample_i in loader.data]
    else:
        noisy_labels = [sample_i for sample_i in loader.targets]
        images = [sample_i for sample_i in loader.data]
    num_samples=int(noise_level*len(images)/100)
    print("Noise samples in trianing : ", num_samples)

    sample_i=0
    label_range = np.unique(noisy_labels)
    idxs_to_change=[]

 
    for n, label_i in enumerate(noisy_labels):
        if sample_i<num_samples and label_i==target_class:
            label_range = label_range[np.where(label_range != target_class)]
            poison_label = np.random.choice(label_range)
            noisy_labels[n] = poison_label
            sample_i+=1
            idxs_to_change.append(1)
        else:
            idxs_to_change.append(0)
    if dataset_name=="stl10":
        loader.data = np.array(images)
        loader.labels=np.array(noisy_labels)
    else:
        loader.data = images
        loader.targets=noisy_labels
    
    return noisy_labels, idxs_to_change










##############################################################################
##################### Loss tracking and noise modeling #######################


def track_training_loss(args, model, device, train_loader, epoch, bmm_model1, bmm_model_maxLoss1, bmm_model_minLoss1, idxs_to_change_track, save_dir, distribution="gaussian",cuda="cuda", GPU_version=True):
    model.eval()

    all_losses = torch.Tensor()
    all_predictions = torch.Tensor()
    all_probs = torch.Tensor()
    all_argmaxXentropy = torch.Tensor()
    all_mid_inter=torch.Tensor()
    

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        prediction = model(data)
        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction, target, reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))


        # # probs = prediction.clone()
        # probs.detach_()
        # all_probs = torch.cat((all_probs, probs.cpu()))
        # arg_entr = torch.max(prediction, dim=1)[1]
        # arg_entr = F.nll_loss(prediction.float(), arg_entr.to(device), reduction='none')
        # arg_entr.detach_()
        # all_argmaxXentropy = torch.cat((all_argmaxXentropy, arg_entr.cpu()))
        # the middle result of the model output-- jian li

        mid_inter=model(data, lin=0, lout=5)
        mid_inter.detach_()
        all_mid_inter=torch.cat((all_mid_inter, mid_inter.cpu()))

    loss_tr = all_losses.data.numpy()
    import copy
    original_loss=copy.deepcopy(loss_tr) #用于画图
    # outliers detection
    #jianli
    
    if distribution=="gaussian":
        print("train model with gaussian mixture distribution: ")
        all_mid_inter = np.nan_to_num(all_mid_inter, nan=0.0, posinf=0.0, neginf=0.0)
        bmm_model_min = np.min(all_mid_inter)
        bmm_model_max = np.max(all_mid_inter)
        normalized_all_mid_inter = (all_mid_inter - bmm_model_min) / (bmm_model_max -  bmm_model_min+1e-6)
        print("the min value: ", np.min(normalized_all_mid_inter))
        print("the max value: ", np.max(normalized_all_mid_inter))
        bmm_model=GMM_mixture2D(n_components=2, max_iter=1000, cuda=cuda,  GPU_version=GPU_version)
        bmm_model.fit(normalized_all_mid_inter, original_loss)

        #用于画图
        save_possibility=True
        if epoch%10==0 and save_possibility:
            normalized_all_mid_inter =  normalized_all_mid_inter.reshape(len(normalized_all_mid_inter), -1)
            if GPU_version:
                normalized_all_mid_inter=torch.from_numpy(normalized_all_mid_inter).float().to(cuda)
            if bmm_model.return_id==1:
                post_possibility=bmm_model.posterior(normalized_all_mid_inter,1)
            else:
                post_possibility=bmm_model.posterior(normalized_all_mid_inter,0)
            post_possibility = np.nan_to_num(post_possibility, nan=0.0, posinf=0.0, neginf=0.0)
            bmm_model.plot_possibility(post_possibility, idxs_to_change_track, epoch, save_dir)

    else:
        print("train model with bert mixture distribution: ")

        max_perc = np.percentile(loss_tr, 95)
        min_perc = np.percentile(loss_tr, 5)
        loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)] #过滤掉 outlier 小于min 大于max 的值
        
        bmm_model_max = torch.FloatTensor([max_perc]).to(device)
        bmm_model_min = torch.FloatTensor([min_perc]).to(device) + 10e-6

        #归一化
        loss_tr = (loss_tr - bmm_model_min.data.cpu().numpy()) / (bmm_model_max.data.cpu().numpy() - bmm_model_min.data.cpu().numpy() + 1e-6)
        loss_tr[loss_tr>=1] = 1-10e-4
        loss_tr[loss_tr <= 0] = 10e-4

        bmm_model = BetaMixture1D(max_iters=10)
        bmm_model.fit(loss_tr)
        bmm_model.create_lookup(1) #通常用于创建一个查找表（lookup table），该查找表可以用于快速访问某些预计算的值或结果。

        #用于画图
        save_possibility=True
        if epoch%10==0 and save_possibility:
            original_loss=original_loss
            min_original_loss=np.min(original_loss)
            max_original_loss=np.max(original_loss)
            original_loss = (original_loss - min_original_loss) / ( max_original_loss-min_original_loss+ 1e-6)
            post_possibility=bmm_model.post_predict(original_loss)
            post_possibility = np.nan_to_num(post_possibility, nan=0.0, posinf=0.0, neginf=0.0)
            bmm_model.plot_possibility(post_possibility, idxs_to_change_track, epoch, save_dir)

    
    # 计算属于第二个分布的 训练数据的数目：
    
    # print("训练数据中属于数据分布2的数据数量: {}".format(np.sum(bmm_model.predict(loss_tr))),"比值：{}".format(np.sum(bmm_model.predict(loss_tr))/(len(loss_tr))))


    # 画出分布图，clean label 和dirty label的loss分布图 jianli
    # 
    # original_loss=original_loss.numpy()
    # min_original_loss=np.min(original_loss)
    # max_original_loss=np.max(original_loss)
    # original_loss = (original_loss - min_original_loss) / ( max_original_loss-min_original_loss+ 1e-6)
    # original_loss=(original_loss-bmm_model_minLoss.data.cpu().numpy())/ (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6) #用于画图
    # bmm_model.plot(original_loss, idxs_to_change_track, epoch, save_dir)


    return all_losses.data.numpy(), \
           all_probs.data.numpy(), \
           all_argmaxXentropy.numpy(), \
           bmm_model, bmm_model_max, bmm_model_min
##############################################################################













########################### Cross-entropy loss ###############################
def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, synthemic_samples=0):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if synthemic_samples==1:
            code_synthemic_images, code_synthemic_labels, synt_sample_index= syn.generate_synthetic_samples( data, target, synthetic_portion=1,device=device)
            data=torch.cat([data,code_synthemic_images],dim=0)
            target=torch.cat([target,code_synthemic_labels],dim=0)
        optimizer.zero_grad()

        output = model(data)
        output = F.log_softmax(output, dim=1)

        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

##############################################################################

############################# Mixup original #################################
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam):

    return lam * F.nll_loss(pred, y_a) + (1 - lam) * F.nll_loss(pred, y_b)

def train_mixUp(args, model, device, train_loader, optimizer, epoch, alpha):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        #lam 表示target a 和target b 之间的权重 
        #inputs 表示增强之后的图片， 
        #target a 表示 叠加前的x_a的ground-truth label  表示叠加之前的 x_b的ground-truth label 
        inputs, targets_a, targets_b, lam = mixup_data(data, target, alpha, device)

        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        loss = mixup_criterion(output, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        loss_per_batch.append(loss.item())

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

##############################################################################

########################## Mixup + Dynamic Hard Bootstrapping ##################################
# Mixup with hard bootstrapping using the beta model
def reg_loss_class(mean_tab,num_classes=10):
    loss = 0
    for items in mean_tab:
        loss += (1./num_classes)*torch.log((1./num_classes)/items)
    return loss

def mixup_data_Boot(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def train_mixUp_HardBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, \
                            bmm_model_maxLoss, bmm_model_minLoss, reg_term, num_classes, distribution):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        tab_mean_class = torch.mean(output_mean,-2)
        output = F.log_softmax(output, dim=1)
        # 标准 mixup 之上 
        # lam 表示mixup 两个随机样本的权重

        # 根据之前的BMM model 得到 每条数据的权重
        B, B_0 = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, distribution)
        B = B.to(device)
        B[B <= 1e-4] = 1e-4
        B[B >= 1 - 1e-4] = 1 - 1e-4

        B_0 = B_0.to(device)
        B_0[B_0<= 1e-4] = 1e-4
        B_0[B_0 >= 1-1e-4] = 1-1e-4

        output_x1 = F.log_softmax(output_x1, dim=1)
        output_x2 = output_x1[index, :]
        B2 = B[index] # mixup的第二个随机选择序列的权重

        z1 = torch.max(output_x1, dim=1)[1]
        z2 = torch.max(output_x2, dim=1)[1]

        loss_x1_vec = (1 - B) * F.nll_loss(output, targets_1, reduction='none')
        loss_x1 = torch.sum(loss_x1_vec) / len(loss_x1_vec)


        loss_x1_pred_vec = B * F.nll_loss(output, z1, reduction='none')
        loss_x1_pred = torch.sum(loss_x1_pred_vec) / len(loss_x1_pred_vec)


        loss_x2_vec = (1 - B2) * F.nll_loss(output, targets_2, reduction='none')
        loss_x2 = torch.sum(loss_x2_vec) / len(loss_x2_vec)


        loss_x2_pred_vec = B2 * F.nll_loss(output, z2, reduction='none')
        loss_x2_pred = torch.sum(loss_x2_pred_vec) / len(loss_x2_pred_vec)

        loss = lam*(loss_x1 + loss_x1_pred) + (1-lam)*(loss_x2 + loss_x2_pred)

        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg # 正则化

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

##############################################################################


##################### Mixup Beta Soft Bootstrapping ####################
# Mixup guided by our beta model with beta soft bootstrapping

def mixup_criterion_mixSoft(pred, y_a, y_b, B, lam, index, output_x1, output_x2):
    """
        B 对应的每条数据都是一个float
    """
    return torch.sum(
        (lam) * ((1 - B) * F.nll_loss(pred, y_a, reduction='none') + 
                 B * (-torch.sum(F.softmax(output_x1, dim=1) * pred, dim=1))) +
        (1-lam) * ((1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + 
                   B[index] * (-torch.sum(F.softmax(output_x2, dim=1) * pred, dim=1)))) / len(pred)


def train_mixUp_SoftBootBeta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model, bmm_model_maxLoss, \
                                            bmm_model_minLoss, reg_term, num_classes, distribution):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5*torch.ones(len(target)).float().to(device)
        else:
            B, B_0 = compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, distribution)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1-1e-4] = 1-1e-4

            B_0 = B_0.to(device)
            B_0[B_0<= 1e-4] = 1e-4
            B_0[B_0 >= 1-1e-4] = 1-1e-4

        inputs_mixed, targets_1, targets_2, lam, index = mixup_data_Boot(data, target, alpha, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1)

        output_x2 = output_x1[index, :]

        tab_mean_class = torch.mean(output_mean, -2)#Columns mean
        #output_x1 ,output_x2 表示单个样本的输出loss，output 表示合成样本的输出loss
        loss = mixup_criterion_mixSoft(output, targets_1, targets_2, B, lam, index, output_x1, output_x2)
        loss_reg = reg_loss_class(tab_mean_class)
        loss = loss + reg_term*loss_reg
        loss.backward()


        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    return (loss_per_epoch, acc_train_per_epoch)


##############################################################################

################################ Dynamic Mixup ##################################
# Mixup guided by our beta model

def mixup_data_beta(x, y, B, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    lam = ((1 - B) + (1 - B[index])) #为了添加后的权重B 能够归一化 a/(a+b) b/(a+b)
    mixed_x = ((1-B)/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x + ((1-B[index])/lam).unsqueeze(1).unsqueeze(2).unsqueeze(3) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index

def mixup_criterion_beta(pred, y_a, y_b):
    lam = np.random.beta(32, 32)
    return lam * F.nll_loss(pred, y_a) + (1-lam) * F.nll_loss(pred, y_b)

def train_mixUp_Beta(args, model, device, train_loader, optimizer, epoch, alpha, bmm_model,
                                bmm_model_maxLoss, bmm_model_minLoss, distribution):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5 * torch.ones(len(target)).float().to(device)
        else:
            #根据batch loss得到一个
            B, B_0= compute_probabilities_batch(data, target, model, bmm_model, bmm_model_maxLoss, bmm_model_minLoss, distribution)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1 - 1e-4] = 1 - 1e-4

            B_0 = B_0.to(device)
            B_0[B_0<= 1e-4] = 1e-4
            B_0[B_0 >= 1-1e-4] = 1-1e-4
            
            # B=1-B_0 # #修改权重为B0 

        #合成的图片的权重是学习出来的，之前的mixup固定或者随机的
    
        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
    
        output = model(inputs_mixed)
     
        output = F.log_softmax(output, dim=1)

        loss = mixup_criterion_beta(output, targets_1, targets_2)

        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx + 1) * args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr']))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]
    return (loss_per_epoch, acc_train_per_epoch)

################################################################################


################## Dynamic Mixup + soft2hard bootstraping ##################
def mixup_criterion_SoftHard(pred, y_a, y_b, B, index, output_x1, output_x2, Temp):
    """
        pred: 表示mixup image的输出
        y_a: mixup之前的 a 的ground-truth
        y_b: mixup之前的 b 的ground-truth
        index: mixup的 index
        output_x1: mixup之前的 a输出 用于修改loss
        output_x2: mixup之前的 b输出 用户修改loss 
        pred: log (h)
        (-torch.sum(F.softmax(output_x1/Temp, dim=1) :z_p表示mixup a的输出之后
    """
    return torch.sum(
        (0.5) * (
                (1 - B) * F.nll_loss(pred, y_a, reduction='none') + B * (-torch.sum(F.softmax(output_x1/Temp, dim=1) * pred, dim=1))) +
                (0.5) * (
                (1 - B[index]) * F.nll_loss(pred, y_b, reduction='none') + B[index] * (-torch.sum(F.softmax(output_x2/Temp, dim=1) * pred, dim=1)))) / len(
        pred)

def train_mixUp_SoftHardBetaDouble(args, model, device, train_loader, optimizer, epoch, bmm_model, \
                                    bmm_model_maxLoss, bmm_model_minLoss, countTemp, k, temp_length, reg_term, num_classes, distribution):
    model.train()
    loss_per_batch = []

    acc_train_per_batch = []
    correct = 0
    steps_every_n = 2 # 2 means that every epoch we change the value of k (index)
    temp_vec = np.linspace(1, 0.001, temp_length)

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_x1 = model(data)
        output_x1.detach_()
        optimizer.zero_grad()

        if epoch == 1:
            B = 0.5*torch.ones(len(target)).float().to(device)
        else:
            #计算这个batch 的高loss和低loss
            # poison samples和clean samles
    
            B, B_0 = compute_probabilities_batch(data, target, model,  bmm_model,  bmm_model_maxLoss,  bmm_model_minLoss, distribution)
            B = B.to(device)
            B[B <= 1e-4] = 1e-4
            B[B >= 1-1e-4] = 1-1e-4

            B_0 = B_0.to(device)
            B_0[B_0<= 1e-4] = 1e-4
            B_0[B_0 >= 1-1e-4] = 1-1e-4

            # #修改权重为B0 
            # B=1-B_0

        # 返回一个值 B： 是一个list B 表示 样本属于dirty label分布的概率 
        #更换poisoning samples 的分布到 clean samples 的分布
 
        inputs_mixed, targets_1, targets_2, index = mixup_data_beta(data, target, B, device)
        output = model(inputs_mixed)
        output_mean = F.softmax(output, dim=1)
        output = F.log_softmax(output, dim=1) # mixup data 的输出

        output_x2 = output_x1[index, :]  #原始数据的版本1的输出
        tab_mean_class = torch.mean(output_mean,-2) #原始数据版本2的输出

        Temp = temp_vec[k]
        # 同时更改loss函数
        loss = mixup_criterion_SoftHard(output, targets_1, targets_2, B, index, output_x1, output_x2, Temp)
        loss_reg = reg_loss_class(tab_mean_class, num_classes)
        loss = loss + reg_term*loss_reg


        loss.backward()

        optimizer.step()
        ################## monitor losses  ####################################
        loss_per_batch.append(loss.item())
        ########################################################################

        # save accuracy:
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        acc_train_per_batch.append(100. * correct / ((batch_idx+1)*args.batch_size))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}, Temperature: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(),
                       100. * correct / ((batch_idx + 1) * args.batch_size),
                optimizer.param_groups[0]['lr'], Temp))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_train_per_epoch = [np.average(acc_train_per_batch)]

    countTemp = countTemp + 1
    if countTemp == steps_every_n:
        k = k + 1
        k = min(k, len(temp_vec) - 1)
        countTemp = 1

    return (loss_per_epoch, acc_train_per_epoch, countTemp, k)

########################################################################


def compute_probabilities_batch(data, target, cnn_model, bmm_model, bmm_model_max, bmm_model_min, distribution):
    cnn_model.eval()
    if distribution=="gaussian":
        outputs = cnn_model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses.detach_()
        mid_inter=cnn_model(data, lin=0, lout=5)
        mid_inter.detach_()
        # normalized_mid_inter=torch.nan_to_num(mid_inter, nan=0.0)
        normalized_mid_inter = (mid_inter- bmm_model_min) / (bmm_model_max -  bmm_model_min + 1e-6)
        normalized_mid_inter=normalized_mid_inter.reshape(len(normalized_mid_inter), -1).to("cpu")
        B_1, B_0 = bmm_model.look_lookup(normalized_mid_inter, bmm_model_max, bmm_model_min)
    else:
        outputs = cnn_model(data)
        outputs = F.log_softmax(outputs, dim=1)
        batch_losses = F.nll_loss(outputs.float(), target, reduction = 'none')
        batch_losses.detach_()
        outputs.detach_()
        #将batch losses 进行归一化，归一化的loss 大于1 表示超过 maxloss
        batch_losses = (batch_losses - bmm_model_min) / (bmm_model_max - bmm_model_min + 1e-6)
        batch_losses[batch_losses >= 1] = 1-10e-4
        batch_losses[batch_losses <= 0] = 10e-4

        #B = bmm_model.posterior(batch_losses,1)
        #poison samples和clean samles
        B_1, B_0 = bmm_model.look_lookup(batch_losses, bmm_model_max, bmm_model_min)
    cnn_model.train()
    # for i in range(len(B_0)):
    #     print("属于分布clean samples 的loss 分布是：",B_0[i],"属于分布poison samples 的loss 分布是：",B_1[i],)
    return torch.FloatTensor(B_1), torch.FloatTensor(B_0),


def test_cleaning(args, model, device, test_loader,target_class=0):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0

    target_correct=0
    target_num=0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

            target_ids=np.where(target.cpu().detach().numpy()==target_class)
            target_correct += pred[target_ids].eq(target[target_ids].view_as(pred[target_ids])).sum().item()
            target_num += np.sum(target_ids)
    test_loss /= len(test_loader.dataset)
  

    loss_per_epoch = [np.average(loss_per_batch)]
    #acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]
    target_acc_val_per_epoch = np.array(100. * target_correct / target_num)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%  target class accuracy: {})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), target_acc_val_per_epoch))
    return (loss_per_epoch, acc_val_per_epoch, target_acc_val_per_epoch)




def compute_loss_set(args, model, device, data_loader):
    model.eval()
    all_losses = torch.Tensor()
    for batch_idx, (data, target) in enumerate(data_loader):
        prediction = model(data.to(device))
        prediction = F.log_softmax(prediction, dim=1)
        idx_loss = F.nll_loss(prediction.float(), target.to(device), reduction = 'none')
        idx_loss.detach_()
        all_losses = torch.cat((all_losses, idx_loss.cpu()))
    return all_losses.data.numpy()


def val_cleaning(args, model, device, val_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.val_batch_size))

    val_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.average(acc_val_per_batch)]
    return (loss_per_epoch, acc_val_per_epoch)


################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        # return the pdf of the point in x set
        # y表示分布的label ，0 表示clean 分布 1 表示 dirty 分布
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        #得到 一个sum值,混合概率分布，这个分布是两个值相加
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        #同时考虑权重在两个分布下的概率值
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)



    def responsibilities(self, x):
        #得到两个加权分布
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0) #处理后在0-1之间
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)
        # EM on beta distributions unsable with x == 0 or 1  处理数据到 （0，1）之间
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step r 表示两个数组，分别是两个beta 分布对于x 的拟合
            r = self.responsibilities(x)

            # M-step 固定 r值，更新beta的参数值
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5  # 1表示第二个分布

    def post_predict(self, x):
        return self.posterior(x, 1) 

    def create_lookup(self, y):
        #得到 y 的分布，在0-1的分布概率
        # y 表示 1 
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution) #切成100小份
 
        lookup_0=self.posterior(x_l, 0) #得到beta2 分布在0-1 之间的概率值
        lookup_t = self.posterior(x_l, y) #得到beta2 分布在0-1 之间的概率值



        lookup_t[np.argmax(lookup_t):] = lookup_t.max()  #超过最大值之后全都修改为最大值
 
        self.lookup = lookup_t # 得到关于poison samples 的loss分布
        self.lookup_0=lookup_0 # 得到关于clean samples 的loss分布

        self.lookup_loss = x_l # I do not use this one at the end
        print("alpha: ", self.alphas)
        print("betas: ", self.betas)




    def look_lookup(self, x, loss_max, loss_min):
   
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i], self.lookup_0[x_i]




    def plot_possibility(self, loss_truth, idx_to_change, epoch, save_dir):
        color1='#fdac53'
        color2='#00bbe6'
        idx_to_change=np.array(idx_to_change).astype(bool)
        fig, ax1 = plt.subplots(figsize=(5, 4))
        clean_loss=np.array(loss_truth)[~idx_to_change]
        dirty_loss=np.array(loss_truth)[idx_to_change]
        print("num clean: ",len(clean_loss), "num dirty: ", len(dirty_loss))
        nbins=100
    
        attr_name1, attr_name2="dirty", "clean"
   

        mem_n, mem_bins, mem_patches = ax1.hist(clean_loss, bins=nbins, density=True,  color=color1, label=attr_name2,  alpha=0.7, )
        nm_n, nm_bins, nm_patches = ax1.hist(dirty_loss, bins=nbins, density=True, color=color2, label=attr_name1, alpha=0.7, )
        
        ax1.legend(loc='upper left')
    
        plt.tight_layout()
        plt.legend()

        dir=os.path.join(save_dir,"loss_distribution/")
        if not os.path.isdir(dir):
            print(f" Create save image folder: {dir}")
            os.makedirs(dir)
        plt.savefig('{}'.format(dir+"beta_epoch_{}.pdf".format(epoch)))
        np.save(os.path.join(dir,f"clean_loss_{epoch}.pdf"),clean_loss)
        np.save(os.path.join(dir,f"dirty_loss_{epoch}.pdf"),dirty_loss)
        import plot_code
        all_prediction=np.concatenate((clean_loss, dirty_loss))
        all_answers = np.array(np.concatenate((np.zeros(len(clean_loss)), np.ones(len(dirty_loss)))), dtype=bool)
        plot_code.plot_tprfpr(all_prediction, all_answers, save_dir=dir, epoch=epoch)


    # def plot(self, loss_truth, idx_to_change, epoch, save_dir):
    #     color1='#fdac53'
    #     color2='#00bbe6'
    #     # 拟合结果
    #     # idx_to_change=np.array(idx_to_change).astype(bool)
    #     # x = np.linspace(0, 1, 100)
    #     # fig, ax = plt.subplots(figsize=(5, 4))
    #     # plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
    #     # plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
    #     # plt.plot(x, self.probability(x), lw=2, label='mixture')
    #     # #真实结果
    #     # print(len(loss_truth))
    #     # print(len(idx_to_change))
    #     # clean_loss=np.array(loss_truth)[~idx_to_change]
    #     # dirty_loss=np.array(loss_truth)[idx_to_change]
    #     # print("num clean: ",len(clean_loss), "num dirty: ", len(dirty_loss))
    #     # nbins=100
    #     # color1='#fdac53'
    #     # color2='#00bbe6'
    #     # attr_name1, attr_name2="dirty", "clean"
    #     # mem_n, mem_bins, mem_patches = ax.hist(clean_loss, bins=nbins, density=True,  color=color1, label=attr_name2,  alpha=0.7, )
    #     # nm_n, nm_bins, nm_patches = ax.hist(dirty_loss, bins=nbins, density=True, color=color2, label=attr_name1, alpha=0.7, )


    #     # plt.tight_layout()
    #     # plt.legend()
     

    #     # dir=os.path.join(save_dir,"loss_distribution/")
    #     # if not os.path.isdir(dir):
    #     #     print(f" Create save image folder: {dir}")
    #     #     os.makedirs(dir)
    #     # plt.savefig('{}'.format(dir+"epoch_{}.pdf".format(epoch)))
    #     idx_to_change=np.array(idx_to_change).astype(bool)
    #     x = np.linspace(0, 1, 100)
    #     fig, ax1 = plt.subplots(figsize=(5, 4))


    #     #真实结果
    #     clean_loss=np.array(loss_truth)[~idx_to_change]
    #     dirty_loss=np.array(loss_truth)[idx_to_change]
    #     print("num clean: ",len(clean_loss), "num dirty: ", len(dirty_loss))
    #     nbins=100
    
    #     attr_name1, attr_name2="dirty", "clean"
    #     ax2 = ax1.twinx()
    #     mem_n, mem_bins, mem_patches = ax2.hist(clean_loss, bins=nbins, density=True,  color=color1, label=attr_name2,  alpha=0.7, )
    #     nm_n, nm_bins, nm_patches = ax2.hist(dirty_loss, bins=nbins, density=True, color=color2, label=attr_name1, alpha=0.7, )
        
        
    #     # 添加第二个图例
    #     ax2.legend(loc='upper right')

    #     ax1.plot(x, self.weighted_likelihood(x, 0), label='negative',c=color1)
    #     ax1.plot(x, self.weighted_likelihood(x, 1), label='positive',c=color2)

       

    #     ax1.plot(x, self.probability(x), lw=2, label='mixture')
    #     ax1.legend(loc='upper left')
    
    #     plt.tight_layout()
    #     plt.legend()
        

    #     dir=os.path.join(save_dir,"loss_distribution/")
    #     if not os.path.isdir(dir):
    #         print(f" Create save image folder: {dir}")
    #         os.makedirs(dir)
    #     plt.savefig('{}'.format(dir+"epoch_{}.pdf".format(epoch)))





    def __str__(self):
        return 'BetaMixture1D(w={}, alpha={}, beta={})'.format(self.weight, self.alphas, self.betas)
def save_info(save_dir, data_set, file_title):
    path = save_dir + f'/{file_title}_info.txt'
    print('save path: {}'.format(path))
    if os.path.isfile(path):
        os.remove(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(path, 'w', encoding='utf-8') as fout:
        fout.writelines(data_set)




def train_with_l2_norm():
    print("trianing model with l2 normalization: ")

def train_with_early_stopping():
    print("training model with early_stoping: ")


class GMM_mixture2D(object):
    def __init__(self, n_components, max_iter=10, tol=1e-3, cuda="cuda",GPU_version=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.eps_nan = 1e-12
        self.mean=None
        self.weight=None
        self.return_id=0
        self.cuda=cuda
        self.GPU_version=GPU_version
        self.PCA=PCA(n_components=10, svd_solver='full')

    def fit(self, data, all_loss):
        data =  data.reshape(len(data), -1)
        K, N = data.shape
        print("train mixture model with feature:", N)
        import time
        #添加高斯分布 GPU 版本
        start_time=time.time()
        # np.save("./feature", np.array(data))
        if self.GPU_version:
            print("train gaussiann model with GPU: ")
            import gmm_torch.gmm as gmm
            # data=self.PCA.fit_transform(data)
            K, N = data.shape
            self.gmm_model=gmm.GaussianMixture(n_components=self.n_components, n_features=N, covariance_type="diag",device=self.cuda)
            data=torch.from_numpy(data).float().to(self.cuda)
            self.gmm_model.fit(data)
            end_time=time.time()
        else:
            self.gmm_model=GMM(n_components=self.n_components, covariance_type="diag", max_iter=self.max_iter)
            self.gmm_model.fit(data)
            end_time=time.time()
        print("time_all: ",end_time-start_time)
      
        # sum_possibility=np.sum(self.posterior(np.array(data)[np.array(top_loss_id)[-num_sample_signs:]], 1))
        # if sum_possibility< int(num_sample_signs*0.1):
        #根据两个分类结果分成两类 loss ，查看loss的中间值
        distribution_id_0=np.squeeze(self.predict(data,y=0))
        distribution_id_1=np.squeeze(self.predict(data,y=1))
        # print(distribution_id_0)
        distribution_loss_0=np.array(all_loss)[distribution_id_0]
        distribution_loss_1=np.array(all_loss)[distribution_id_1]
       
        mid_loss_0=np.median(distribution_loss_0)
        mid_loss_1=np.median(distribution_loss_1)
        # print(mid_loss_0,"  ", mid_loss_1)
        if mid_loss_0>mid_loss_1:
            self.return_id=0
        else:
            self.return_id=1
        # print("sum poss ",sum_possibility," return_id ", self.return_id)

        # print(gmm.means_)
        # print(gmm.weights_)
        # print(gmm.predict_proba(data))



    def predict(self, x, y=1):
        return self.posterior(x, y) > 0.5  # 1表示第二个分布

    def posterior(self, x, y=None):
        if y==None:
            return self.gmm_model.predict_proba(x)
        else:
            if self.GPU_version:
                re=self.gmm_model.predict_proba(x).to("cpu").numpy()[:,y:y+1]
                return  np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                re=self.gmm_model.predict_proba(x).to("cpu").numpy()[:,y:y+1]
                return np.nan_to_num(re, nan=0.0, posinf=0.0, neginf=0.0)[:,y:y+1]
                
    
    def create_lookup(self, y):
        #得到 y 的分布，在0-1的分布概率
        # y 表示 1 
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution) #切成100小份
 
        lookup_0=self.posterior(x_l, 0) #得到beta2 分布在0-1 之间的概率值
        lookup_t = self.posterior(x_l, y) #得到beta2 分布在0-1 之间的概率值

        lookup_t[np.argmax(lookup_t):] = lookup_t.max()  #超过最大值之后全都修改为最大值
 
        self.lookup = lookup_t # 得到关于poison samples 的loss分布
        self.lookup_0=lookup_0 # 得到关于clean samples 的loss分
        self.lookup_loss = x_l # I do not use this one at the end
        print("alpha: ", self.alphas)
        print("betas: ", self.betas)

    def look_lookup(self, x_i, min, max, ):
        """
            返回两个不同的高斯分布
            随机选择100 high loss数据的特征
        """
        x_i =  x_i.reshape(len(x_i), -1)
        if self.GPU_version:
            x_i=x_i.to(self.cuda)
        
            # print("Feature contains NaN:", np.isnan( x_i).any())
            # print("Feature contains Inf:", np.isinf( x_i).any())
            # x_i=self.PCA.fit_transform(x_i)
            # print(np.shape(x_i.to("cpu").numpy()))
            
  
        lookup_0=self.posterior(x_i, 0) #得到beta2 分布在0-1 之间的概率值
        lookup_t = self.posterior(x_i, 1) #得到beta2 分布在0-1 之间的概率值
        
        if self.return_id==1:
            return np.squeeze(lookup_t), np.squeeze(lookup_0)
        else:
            return np.squeeze(lookup_0), np.squeeze(lookup_t)

    def plot_possibility(self, loss_truth, idx_to_change, epoch, save_dir):
        color1='#fdac53'
        color2='#00bbe6'
        idx_to_change=np.array(idx_to_change).astype(bool)
        fig, ax1 = plt.subplots(figsize=(5, 4))
        clean_loss=np.array(loss_truth)[~idx_to_change]
        dirty_loss=np.array(loss_truth)[idx_to_change]


        print("num clean: ",len(clean_loss), "num dirty: ", len(dirty_loss))
        nbins=100

        attr_name1, attr_name2="dirty", "clean"
   
        mem_n, mem_bins, mem_patches = ax1.hist(clean_loss, bins=nbins, density=True,  color=color1, label=attr_name2,  alpha=0.7, )
        nm_n, nm_bins, nm_patches = ax1.hist(dirty_loss, bins=nbins, density=True, color=color2, label=attr_name1, alpha=0.7, )
        
        ax1.legend(loc='upper left')
    
        plt.tight_layout()
        plt.legend()

        dir=os.path.join(save_dir,"loss_distribution/")
        if not os.path.isdir(dir):
            print(f" Create save image folder: {dir}")
            os.makedirs(dir)
        plt.savefig('{}'.format(dir+"gaussian_epoch_{}.pdf".format(epoch)))
        np.save(os.path.join(dir,f"clean_loss_{epoch}.pdf"),clean_loss)
        np.save(os.path.join(dir,f"dirty_loss_{epoch}.pdf"),dirty_loss)
        import plot_code
        all_prediction=np.concatenate((clean_loss, dirty_loss))
        all_answers = np.array(np.concatenate((np.zeros(len(clean_loss)), np.ones(len(dirty_loss)))), dtype=bool)
        plot_code.plot_tprfpr(all_prediction,all_answers,save_dir=dir, epoch=epoch)