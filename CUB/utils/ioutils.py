import argparse
import os
from datetime import datetime

import git
import yaml


def parse_args_siamese():
    parser = argparse.ArgumentParser(
        description='Siamese Baseline for Few shot learning')
    
    # Eval options
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Do only evaluation (presumably using'
                        ' a loaded model)')
    parser.add_argument('--test_split', action='store_true', default=False,
                        help='Use the test split for validation')
    parser.add_argument('--n_way', default=5, type=int, 
                        help='class num to classify for testing (validation)') 
    parser.add_argument('--n_support', default=5, type=int, 
                        help='number of labeled data in each class')
    parser.add_argument('--n_query', default=16, type=int, 
                        help='number of query examples in each class')
    parser.add_argument('--num_val_ep', default=100, type=int, 
                        help='number of episodes for validation') 
    
    # Training options
    parser.add_argument('--num_epoch', default=30, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--data_aug', action='store_true', default=False, 
                        help='Train with data augmentation')
    parser.add_argument('-b', '--batch_size', default=96, type=int,
                        help='Minibatch size used')
    parser.add_argument('--margin', default=0.2, type=float, 
                        help='Margin to use for triplet loss')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', 
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    
    # Other options
    parser.add_argument('--attr_pred', action='store_true', 
                        default=False, help='Use an auxiliary attribute '
                                            'predictor branch')
    parser.add_argument('--lam', type=float, default=1.,
                        help='Weight of attribute loss in the final loss func')
    parser.add_argument('--fix_val', action='store_true', default=False,
                        help='Each time validation is done, the same classes'
                             ' are chosen. Numpy random seed is fixed.')

    # Logging options
    parser.add_argument('--save_dir', default='expts/tmp_last', type=str,
                        help='path to directory for saving results')
    parser.add_argument('--log_step', type=int, help='log each <log_step> epoch', 
                        default=2)

    return parser.parse_args()

def parse_args_vc():
    parser = argparse.ArgumentParser(
        description='PAN model for few shot learning')

    # Eval options
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='Do only evaluation (presumably using'
                             ' a loaded model)')
    parser.add_argument('--test_split', action='store_true', default=False,
                        help='Use the test split for validation')
    parser.add_argument('--n_way', default=5, type=int,
                        help='class num to classify for testing (validation)')
    parser.add_argument('--n_support', default=5, type=int,
                        help='number of labeled data in each class')
    parser.add_argument('--n_query', default=16, type=int,
                        help='number of query examples in each class')
    parser.add_argument('--num_val_ep', default=100, type=int,
                        help='number of episodes for validation')

    # Training options
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='Number of epochs for training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--fix_val', action='store_true', default=False,
                        help='Each time validation is done, the same classes'
                             ' are chosen. Numpy random seed is fixed.')

    # Model options
    parser.add_argument('-hi', '--hidden', type=int,
                        nargs='+', default=[350, 350, 350],
                        help='Number of hidden units in the GCN layers.')
    parser.add_argument('-deg', '--degree', type=int, default=1,
                        help='Degree of the convolution (Number of supports)')
    parser.add_argument('-do', '--dropout', type=float, default=0.5,
                        help='Dropout fraction')
    parser.add_argument('-sup_do', '--support_dropout',
                        type=float, default=0.15,
                        help='Use dropout on the support matrices, dropping '
                             'all the connections from some nodes')
    parser.add_argument('--hybrid', action='store_true', default=False,
                        help='Whether to a hybrid model. In this case'
                             ' nout is set to number of attributes + nout and'
                             ' use_at_lab is set to True')
    parser.add_argument('--use_at_lab', action='store_true', default=False,
                        help='Whether to use attribute labels. In this case'
                             ' nout is set to number of attributes')
    parser.add_argument('--label_func', default='OR', nargs='?', 
                        choices=['OR', 'AND', 'XNOR', 'AND_XOR'],
                        help='Logical function to combine attribute labels')
    parser.add_argument('--num_sup_at', default=312, type=int,
                        help='Use only the first num_sup_at attribute labels')
    parser.add_argument('--nout', type=int, default=1,
                        help='Number of outputs of the MLP decoder, typically '
                             'same as number of attributes')
    parser.add_argument('--lam', type=float, default=1.,
                        help='Weight of attribute loss in the final loss func')

    # Other options
    parser.add_argument('--no_ge', dest='ge', action='store_false',
                        default=True, help='Whether to use graph image encoder')

    # Logging options
    parser.add_argument('--save_dir', default='expts/tmp_last', type=str,
                        help='path to directory for saving results')
    parser.add_argument('--log_step', type=int, help='log each <log_step> epoch',
                        default=1)

    return parser.parse_args()


def get_sha():
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
    except:
        sha = 'None'
    return sha

def save_expt_info(args, sha):
    if isinstance(args, dict):
        # in case a dictionary or attrdict is used for args
        save_args = dict(args)
    else:
        save_args = vars(args)
    
    info = {
        'args' : save_args,
        'githash' : sha
    }
    with open(os.path.join(args.save_dir, 'info.yaml'), 'w') as outfile:
        yaml.dump(info, outfile)

def get_log_str(args, log_info):
    now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
    log_str = '-'*36 + 'Expt Log' + '-'*36 + '\n'
    log_str += '{:<25} : {}\n'.format('Time', now)
    log_str += '{:<25} : {}\n'.format('Epoch', log_info['epoch'])
    log_str += '{:<25} : {:.4f}\n'.format('Train loss', log_info['train_loss'])
    log_str += '{:<25} : {:.2f}\n'.format(
        'Val accuracy', 100. * log_info['val_acc'])
    log_str += '-'*80
    return log_str

def write_to_log(args, log_str):
    with open(os.path.join(args.save_dir, 'log.txt'), 'a+') as outfile:
        print(log_str, file=outfile)

def parse_log(log_file):
    import dateutil.parser 
    info = []
    with open(log_file, 'r') as f:
        curr_entry = {}
        for line in f.readlines():
            val = line[line.find(':')+1:]
            if line.startswith('Time'):
                curr_entry['time'] = dateutil.parser.parse(val)
            elif line.startswith('Epoch'):
                curr_entry['epoch'] = int(val)
            elif line.startswith('Train loss'):
                curr_entry['train_loss'] = float(val)
            elif line.startswith('Val accuracy'):
                curr_entry['val_acc'] = float(val)
            elif line.startswith('-'*80):
                info.append(dict(curr_entry))
                curr_entry = {}
    return info
