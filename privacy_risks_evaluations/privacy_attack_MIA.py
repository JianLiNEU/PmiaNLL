from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import models.PreResNet as PreResNet
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

def attack_dataset(num_samples=1000):
        # CIFAR meta
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if args.dataset == 'CIFAR10':
        clean_trainset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        clean_testset = datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=transform_train)
    elif args.dataset == 'CIFAR100':
        clean_trainset = datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        clean_testset = datasets.CIFAR100(root=args.root_dir, train=False, download=True, transform=transform_train)
    else:
        raise NotImplementedError
    train_indices = np.random.choice(len(clean_trainset), num_samples, replace=False)
    test_indices = np.random.choice(len(clean_testset), num_samples, replace=False)
    mem_dataset = Subset(clean_trainset, train_indices)
    nonmem_dataset = Subset(clean_testset, test_indices)
    mem_loader=torch.utils.data.DataLoader(mem_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    nonmem_loader=torch.utils.data.DataLoader(nonmem_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    
    return mem_loader,nonmem_loader




def get_target_model(model_type="clean_model",num_classes=10, model_path=""):
    model = PreResNet.ResNet18(num_classes=num_classes).to(device) #模型的参数
    if model_type=="clean_model":
        print("attack clean model...")
        model.load_state_dict(torch.load(model_path))
    elif model_type=="poison_model":
        print("attack poison model...")

    else:
        print("no such models!!! ")
    return model

def _Mentr(preds, y):
    y=F.one_hot(y,num_classes)
    preds=F.softmax(preds, dim=1)
    preds, y=preds.cpu().detach().numpy(), y.cpu().detach().numpy()
    fy = np.sum(preds*y, axis=1)
    fi = preds*(1-y)
    # print(-(1-fy)*np.log(fy+1e-30))
    score = -(1-fy)*np.log(fy+1e-30)-np.sum(fi*np.log(1-fi+1e-30), axis=1)
    return score
def loss(preds, y):
    output = F.log_softmax(preds, dim=1)
    test_loss= F.nll_loss(output, y, reduction='none').cpu().detach().numpy()
    return test_loss

def get_membership_score(model, dataloder, num_classes):
    score_set=[]
    for i, (input, label) in enumerate(dataloder):
        
        input, label = input.to(device), label.to(device)
        # output = F.log_softmax(model(input), dim=1)
        # output=model(input)
        # features=_Mentr(model(input), label)
        features=loss(model(input), label)
        score_set.extend(features)

    return score_set


def metric_based_MIA(mem_features,nonmem_features):
    # print(mem_features)
    mem_label = np.ones(len(mem_features))
    nonmem_label = np.zeros(len(nonmem_features))

    mia_auc = roc_auc_score(np.r_[mem_label, nonmem_label],
                            np.r_[mem_features, nonmem_features])
    fpr, tpr, _ = roc_curve(np.r_[mem_label, nonmem_label],
                            np.r_[mem_features, nonmem_features])
    low_auc_01 = tpr[np.where(fpr < .01)[0][-1]]
    low_auc_001 = tpr[np.where(fpr < .001)[0][-1]]
    
    print(f"attack_auc: {mia_auc}")
    print(f"attack_tpr@1%fpr :{low_auc_01}")










def main():
    model_path="./noise_models_PreResNet18_runs/clean_mmodel_40.0_150/poison_model_with_defence_last_epoch_150_valLoss_1.41073_valAcc_56.49000_noise_40_bestValLoss_0.00000.pth"
    # load member dataset and nonmember dataset
    mem_dataset,nonmem_dataset=attack_dataset()

    # load model
    target_model=get_target_model(model_path=model_path)

    #get membership score
    mem_features=get_membership_score(target_model, mem_dataset, num_classes)
    nonmem_features=get_membership_score(target_model, nonmem_dataset, num_classes)

    # choose MIA method
    metric_based_MIA(mem_features, nonmem_features)






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--root-dir', type=str, default='.', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'], help='dataset to train on, default: CIFAR10')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    parser.add_argument('--cuda',  default="cuda:1",
                        help='disables CUDA support')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed, set it to go to determinist mode. We used 1 for the paper, default: None')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')
    parser.add_argument('--noise-level', type=float, default=40.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    parser.add_argument('--experiment-name', type=str, default='runs',
                        help='name of the experiment for the output files storage, default: runs')
    parser.add_argument('--alpha', type=float, default=32, help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[100, 250],
                        help="Milestones for the LR sheduler, default 100 250")
    parser.add_argument('--Mixup', type=str, default='Dynamic', choices=['None', 'Static', 'Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--clean_model', type=bool, default=False)
    parser.add_argument('--BootBeta', type=str, default='soft', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="the number of classes of the target dataset cifar10 10.")

    args = parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    
    if args.dataset=="CIFAR10":
        num_classes=10
    elif args.dataset=="CIFAR100":
        num_classes=100


    main()