    
import  argparse 
    
def parser_setting():  
      # Training settings
    parser = argparse.ArgumentParser(description='This is the official implementation for the ICML 2019 Unsupervised label noise modeling and loss correction paper. This work is under MIT licence. Please refer to the RunScripts.sh and README.md files for example usages. Consider citing our work if this code is usefull for your project')
    parser.add_argument('--root-dir', type=str, default='/home/guest100/data/', help='path to CIFAR dir where cifar-10-batches-py/ and cifar-100-python/ are located. If the datasets are not downloaded, they will automatically be and extracted to this path, default: .')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='input batch size for training, default: 128')
    parser.add_argument('--test-batch-size', type=int, default=256,
                        help='input batch size for testing, default: 100')
    parser.add_argument('--epochs', type=int, default=300,  help='number of epochs to train, default: 10')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate, default: 0.1')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to train on, default: CIFAR10')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default: 0.9')
    parser.add_argument('--device',  default="cuda:1", help='disables CUDA support')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed, set it to go to determinist mode. We used 1 for the paper, default: None')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')

    parser.add_argument('--experiment-name', type=str, default='MD-DYR-SH',
                        help='name of the experiment for the output files storage, default: runs, MD-DYR-SH')
    parser.add_argument('--alpha', type=float, default=32, help='alpha parameter for the mixup distribution, default: 32')
    parser.add_argument('--M', nargs='+', type=int, default=[100, 250],
                        help="Milestones for the LR sheduler, default 100 250")
    parser.add_argument('--Mixup', type=str, default='Static',
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--clean_model', type=bool, default=False)
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")
    parser.add_argument('--reg-term', type=float, default=0., 
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--exp', type=int, default=6,  help="expeiemnt number : 0.")
    parser.add_argument('--defence_method', type=str, default="None",  help="[dp, l2norm, earlystopping, None]")


    # poisoning membership inference setting
    parser.add_argument('--target_class', type=int, default=0,  help="only select a target classes")
    parser.add_argument('--poison_strategy',  type=str, default="target", help="all and target")

    parser.add_argument('--attack_train_amount', type=int, default=500, help="attack train samples")
    parser.add_argument('--attack_test_amount',  type=int, default=500, help="attack test samples")
    parser.add_argument('--model_name',  default="resnet18",  help='[resnet18 densenet121 vgg19]')
    
    parser.add_argument('--noise_level', type=float, default=6.0,
                        help='percentage of noise added to the data (values from 0. to 100.), default: 80.')
    
    parser.add_argument('--distribution', type=str, default="gaussian", 
                        help='["gaussian, bert"] use different distribution to model the clean and dirty point.')
    parser.add_argument('--synthemic_samples', type=int, default=0, 
                        help='If 1, implement the Poisoning mmebership inference attacks of NDSS 2025')
    
    args = parser.parse_args()


    return args