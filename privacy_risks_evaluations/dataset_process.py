import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch

def _read_labels(label_path):
    with open(label_path, 'rb') as f:
        return (np.fromfile(f, dtype=np.uint8)-1)

def _read_imgs(img_path):
    with open(img_path, 'rb') as f:
        x = np.fromfile(f, dtype=np.uint8)
        x = np.reshape(x, (-1, 3, 96, 96))
        x = np.transpose(x, (0, 3, 2, 1))
        return x

def check_files_exist(output_dir,file_name="tinyimagenet_train_data.npy"):
    """
    Check if processed files already exist
    Args:
        output_dir: Output directory
    Returns:
        bool: True if all files exist, False otherwise
    """
    if not os.path.exists(os.path.join(output_dir, file_name)):
        return False
    else:
        return True

def get_class_id_mapping(data_dir):
    """
    Get class ID mapping relationship
    Args:
        data_dir: Dataset root directory
    Returns:
        dict: Mapping from class names to numeric IDs
    """
    # Read class mapping file
    mapping_file = os.path.join(data_dir, 'wnids.txt')
    class_to_id = {}
    with open(mapping_file, 'r') as f:
        for idx, line in enumerate(f):
            class_name = line.strip()
            class_to_id[class_name] = idx
    return class_to_id

def process_tinyimagenet(data_dir, output_dir):
    """
    Process TinyImageNet dataset
    Args:
        data_dir: TinyImageNet dataset root directory
        output_dir: Output directory
    """
    # Check if files already exist
    if check_files_exist(output_dir,'tinyimagenet_train_data.npy'):
        print("Processed files already exist. Skipping processing...")
        train_data = np.load(os.path.join(output_dir, 'tinyimagenet_train_data.npy'))
        train_labels = np.load(os.path.join(output_dir, 'tinyimagenet_train_label.npy'))
        val_data = np.load(os.path.join(output_dir, 'tinyimagenet_test_data.npy'))
        val_labels = np.load(os.path.join(output_dir, 'tinyimagenet_test_label.npy'))
        print(f"Loaded existing files:")
        print(f"Training data shape: {train_data.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Validation data shape: {val_data.shape}")
        print(f"Validation labels shape: {val_labels.shape}")
        return train_data, train_labels, val_data, val_labels
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get class ID mapping
    class_to_id = get_class_id_mapping(data_dir)
    
    # Process training set
    train_data = []
    train_labels = []
    train_dir = os.path.join(data_dir, 'train')
    
    print("Processing training data...")
    for class_dir in tqdm(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_dir)
        if os.path.isdir(class_path):
            class_id = class_to_id[class_dir]  # Get numeric ID using mapping
            images_dir = os.path.join(class_path, 'images')
            for img_name in os.listdir(images_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(images_dir, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    train_data.append(img_array)
                    train_labels.append(class_id)
    
    # Process validation set
    val_data = []
    val_labels = []
    val_dir = os.path.join(data_dir, 'val')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')
    
    print("Processing validation data...")
    with open(val_annotations, 'r') as f:
        for line in tqdm(f):
            img_name, class_name = line.strip().split('\t')[:2]
            class_id = class_to_id[class_name]  # Get numeric ID using mapping
            img_path = os.path.join(val_dir, 'images', img_name)
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            val_data.append(img_array)
            val_labels.append(class_id)
    
    # Convert to numpy arrays and ensure correct data types
    train_data = np.array(train_data, dtype=np.uint8)  # Store as uint8 to save memory
    train_labels = np.array(train_labels, dtype=np.int64)
    val_data = np.array(val_data, dtype=np.uint8)
    val_labels = np.array(val_labels, dtype=np.int64)
    
    print("Saving processed data...")
    np.save(os.path.join(output_dir, 'tinyimagenet_train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'tinyimagenet_train_label.npy'), train_labels)
    np.save(os.path.join(output_dir, 'tinyimagenet_test_data.npy'), val_data)
    np.save(os.path.join(output_dir, 'tinyimagenet_test_label.npy'), val_labels)
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Validation labels shape: {val_labels.shape}")
    
    return train_data, train_labels, val_data, val_labels

class TinyTensorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        super().__init__()
        # Ensure data is float type and has correct dimensions
        if isinstance(data, np.ndarray):
            # For TinyImageNet, data is in (N, C, H, W) format
            if data.shape[1] == 3:  # If channels first
                data = np.transpose(data, (0, 2, 3, 1))  # Convert to (N, H, W, C)
            self.data = data
        else:
            self.data = data.permute(0, 2, 3, 1).numpy()  # Convert to numpy and (N, H, W, C) format
            
        # Ensure labels are long type
        if isinstance(targets, (np.ndarray, np.int64)):
            self.targets = targets.astype(np.int64)  # Keep as numpy array
        elif isinstance(targets, (list, tuple)):
            self.targets = np.array(targets, dtype=np.int64)  # Convert to numpy array
        else:
            self.targets = targets.cpu().numpy().astype(np.int64)  # Convert tensor to numpy array
            
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and label
        x = self.data[idx]
        y = int(self.targets[idx])  # Convert to Python int
        
        # Convert numpy array to PIL Image
        x = Image.fromarray(x.astype('uint8'))
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
            
        return x, y
    
class stl10Dataset(Dataset):
    def __init__(self, data, targets, transform=None):
        super().__init__()
        # Ensure data is in correct format (N, H, W, C)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 4:
                if data.shape[-1] != 3:  # If channels not last
                    data = np.transpose(data, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
            self.data = data
        else:
            self.data = data.permute(0, 2, 3, 1).numpy()  # Convert to numpy and (N, H, W, C) format
            
        # Ensure labels are in correct format
        if isinstance(targets, (np.ndarray, np.int64)):
            self.targets = targets.astype(np.int64)
        elif isinstance(targets, (list, tuple)):
            self.targets = np.array(targets, dtype=np.int64)
        else:
            self.targets = targets.cpu().numpy().astype(np.int64)
            
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get data and label
        x = self.data[idx]
        y = int(self.targets[idx])  # Convert to Python int
        
        # Convert numpy array to PIL Image
        x = Image.fromarray(x.astype('uint8'))
        
        # Apply transforms if any
        if self.transform:
            x = self.transform(x)
            
        return x, y

def process_stl10(data_dir, output_dir):
    """
    Process STL10 dataset
    
    Args:
        data_dir: Dataset root directory
        output_dir: Output directory
    """
    # Check if already processed
    if check_files_exist(output_dir, 'stl10_train_data.npy'):
        print("STL10 dataset already processed, loading directly...")
        train_data = np.load(os.path.join(output_dir, 'stl10_train_data.npy'))
        train_labels = np.load(os.path.join(output_dir, 'stl10_train_labels.npy'))
        test_data = np.load(os.path.join(output_dir, 'stl10_test_data.npy'))
        test_labels = np.load(os.path.join(output_dir, 'stl10_test_labels.npy'))
        return train_data, train_labels, test_data, test_labels
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load training set
    train_dataset = datasets.STL10(data_dir, split='train', download=True)
    train_data = train_dataset.data  # Already numpy array with shape (N, 96, 96, 3)
    train_labels = train_dataset.labels  # Use labels instead of targets
    
    # Load test set
    test_dataset = datasets.STL10(data_dir, split='test', download=True)
    test_data = test_dataset.data
    test_labels = test_dataset.labels
    
    # Ensure correct data types
    train_data = train_data.astype(np.uint8)
    test_data = test_data.astype(np.uint8)
    train_labels = train_labels.astype(np.int64)
    test_labels = test_labels.astype(np.int64)
    
    # Save processed data
    np.save(os.path.join(output_dir, 'stl10_train_data.npy'), train_data)
    np.save(os.path.join(output_dir, 'stl10_train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'stl10_test_data.npy'), test_data)
    np.save(os.path.join(output_dir, 'stl10_test_labels.npy'), test_labels)
    
    return train_data, train_labels, test_data, test_labels

def get_dataset(args, attack_data=False):
    """
    Process dataset
    Args:
        dataset_name: Dataset name
        data_dir: Dataset root directory
        output_dir: Output directory
    Returns:
        np.ndarray: Training data
        np.ndarray: Training labels
        np.ndarray: Test data
        np.ndarray: Test labels
    """

    if args.dataset == 'CIFAR10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
        transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
        
        clean_trainset= datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset = datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR10(root=args.root_dir, train=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.root_dir, train=False, transform=transform_test)
        num_classes = 10
        
        attack_train=datasets.CIFAR10(root=args.root_dir, train=True)
        attack_test=datasets.CIFAR10(root=args.root_dir, train=False)

        
    elif args.dataset == 'MNIST':
        transform_fmnist_train=transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])
        transform_test=transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),])
        testset = datasets.MNIST(root=args.root_dir, train=False, download=True, transform=transform_test)
        clean_trainset= datasets.MNIST(root=args.root_dir, train=True, download=True,transform=transform_fmnist_train)
        trainset = datasets.MNIST(root=args.root_dir, train=True, download=True,transform=transform_fmnist_train)
        trainset_track = datasets.MNIST(root=args.root_dir, train=True,transform=transform_fmnist_train)
        num_classes = 10
        
   
    elif args.dataset == 'CIFAR100':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
        transform_test=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
        clean_trainset= datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset = datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_train)
        trainset_track = datasets.CIFAR100(root=args.root_dir, train=True, transform=transform_train)
     
        testset = datasets.CIFAR100(root=args.root_dir, train=False, transform=transform_test) # Test data, all test data
        num_classes = 100
        
        attack_train=datasets.CIFAR10(root=args.root_dir, train=True, )
        attack_test=datasets.CIFAR10(root=args.root_dir, train=False, )
        
    elif args.dataset == 'stl10':
        # STL10 dataset normalization parameters
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        # Training set transforms
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # Test set transforms
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # Process dataset
        # train_data, train_labels, test_data, test_labels = process_stl10(args.root_dir, os.path.join(args.root_dir, 'processed'))
        
        # Create datasets
        # clean_trainset = stl10Dataset(train_data, train_labels, transform=transform_train)
        # trainset = stl10Dataset(train_data, train_labels, transform=transform_train)
        # trainset_track = stl10Dataset(train_data, train_labels, transform=transform_train)
        # testset = stl10Dataset(test_data, test_labels, transform=transform_test)
        num_classes = 10
               
        clean_trainset  = datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='train', download=True,transform=transform_train)
        trainset = datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='train', download=True,transform=transform_train)
        trainset_track = datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='train', download=True,transform=transform_train)
        testset = datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='test', download=True,transform=transform_test)
        
        attack_train=datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='train',)
        attack_test=datasets.STL10(os.path.join(args.root_dir, 'stl10_dataset'), split='test',)
    elif args.dataset == 'tinyimagenet':
        # Training data 200*500, test data 200*100
        # from PIL import Image
        
        transform_tiny_train=transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
            # transforms.Lambda(lambda x:x.permute((1,2,0))),     
        ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # transforms.Lambda(lambda x:x.permute((1,2,0))),     
        ])
        train_data, train_labels, val_data, val_labels=process_tinyimagenet(os.path.join(args.root_dir,"tiny-imagenet-200"), os.path.join(args.root_dir,"tiny-imagenet-200/process_data"))
        clean_trainset= TinyTensorDataset(train_data,train_labels,transform=transform_tiny_train)
        trainset= TinyTensorDataset(train_data,train_labels,transform=transform_tiny_train)
        trainset_track= TinyTensorDataset(train_data,train_labels,transform=transform_tiny_train)
   
        testset= TinyTensorDataset(val_data,val_labels,transform=transform_test)
           
        num_classes = 200
        # clean_trainset=datasets.ImageFolder(root=os.path.join(args.root_dir,"tiny-imagenet-200/train"),transform=transform_tiny_train)
        # trainset=datasets.ImageFolder(root=os.path.join(args.root_dir,"tiny-imagenet-200/train"),transform=transform_tiny_train)
        # trainset_track=datasets.ImageFolder(root=os.path.join(args.root_dir,"tiny-imagenet-200/train"),transform=transform_tiny_train)
        # testset=datasets.ImageFolder(root=os.path.join(args.root_dir,"tiny-imagenet-200/val"),transform=transform_tiny_train)
        attack_train=TinyTensorDataset(train_data,train_labels)
        attack_test=TinyTensorDataset(train_data,train_labels)
    else:
        raise NotImplementedError
        
        
    if attack_data:
        return attack_train, attack_test, transform_test
    else:
        return clean_trainset, trainset, trainset_track, testset, num_classes

if __name__ == '__main__':
    data_dir = '/home/yinsi_team/guest_yinsi/data/tiny-imagenet-200'  # Replace with actual dataset path
    output_dir = 'data/tinyimagenet'
    process_tinyimagenet(data_dir, output_dir)