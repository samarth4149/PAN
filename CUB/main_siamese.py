import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.dataset_imgs import TripletDataset
from datasets.dataset_imgs import TripletDatasetwAttr
from datasets.dataset_imgs import ValDataset
from models.siamese import SiameseNet
from utils.ioutils import get_log_str
from utils.ioutils import get_sha
from utils.ioutils import parse_args_siamese
from utils.ioutils import save_expt_info
from utils.ioutils import write_to_log
from utils.misc import AverageMeter

# Hardcoded, change this
NUM_ATTR = 312
FEAT_DIM = 512

EPS = 1e-10

def train(args, loader, model, criterion, optimizer):
    if args.attr_pred:
        model, attr_branch = model
        criterion, attr_criterion = criterion

    model.train()
    avg_loss = AverageMeter()
    with tqdm(total=int(len(loader))) as pbar:
        for i, data in enumerate(loader):
            f1 = model.get_features(data[0].cuda(non_blocking=True))
            f2 = model.get_features(data[1].cuda(non_blocking=True))
            f3 = model.get_features(data[2].cuda(non_blocking=True))
            # d1, d2 = model(x1, x2, x3)
            d1 = (f1-f2).pow(2).sum(1)
            d2 = (f1-f3).pow(2).sum(1)

            feats = [f1, f2, f3]
            
            # reduction = 'mean'
            loss = criterion(d1, d2, torch.cuda.FloatTensor([-1]))
            
            if args.attr_pred:
                for j in range(3):
                    attr_label = data[3+j].cuda(non_blocking=True)
                    certainty_mask = data[6+j].cuda(non_blocking=True)
                    preds = attr_branch(feats[j])
                    curr_loss = attr_criterion(preds, attr_label)
                    curr_loss = ((curr_loss * certainty_mask).sum(dim=1)
                                 /(certainty_mask.sum(dim=1) + EPS))
                    curr_loss = curr_loss.mean()
                    loss += args.lam * curr_loss

            # Compute loss
            avg_loss.update(loss.item(), len(f1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)
    
    return avg_loss.avg

def validate(args, loader, model):
    model.eval()
    n_support = args.n_support
    n_query = args.n_query
    all_acc = []

    if args.fix_val:
        rng_state = torch.get_rng_state()
        torch.manual_seed(44)

    with torch.no_grad(), tqdm(total=len(loader)) as pbar:
        for episode, examples in enumerate(loader):
            # Assuming batch size of 1
            examples = examples[0].contiguous()
            # n_way x (n_support + n_query) x C x H x W
            n_way, n_tot, C, H, W = examples.shape 
            all_support = examples[:, :n_support, :, :, :].contiguous().view(
                n_way*n_support, C, H, W).contiguous().cuda()
            all_query = examples[:, n_support:, :, :, :].contiguous().view(
                n_way*n_query, C, H, W).contiguous().cuda()
            all_support_feats = model.get_features(all_support)

            #(n_way * n_query) x feat_size
            all_query_feats = model.get_features(all_query) 
            
            # n_way x n_support x feat_size
            all_support_feats = all_support_feats.view(
                n_way, n_support, -1).contiguous()
            
            T1 = all_query_feats.norm(p=2, dim=1).unsqueeze(0).unsqueeze(-1)\
                 .pow(2).expand(n_way, n_way * n_query, n_support)
            T2 = all_support_feats.norm(p=2, dim=2).unsqueeze(1)\
                 .pow(2).expand_as(T1)
            
            # Note : torch transpose works differently than numpy
            # https://pytorch.org/docs/stable/torch.html#torch.transpose
            T3 = -2 * torch.matmul(
                all_query_feats, all_support_feats.transpose(1, 2))

            # T1 + T2 + T3 is (n_way x (n_way*n_query) x n_support)
            # Each entry is a distance of a given query example from 
            # a support example 
            scores = torch.sqrt(T1 + T2 + T3).mean(dim=2)
            scores = scores.t() #transpose

            #integer division
            true_labels = torch.arange(len(all_query_feats)).cuda()/n_query 
            assigned_labels = scores.argmin(axis=1)
            
            accuracy = (true_labels == assigned_labels)\
                       .sum()/float(len(true_labels))
            
            all_acc.append(accuracy.cpu())
            pbar.update(1)
    
    if args.fix_val:
        rng_state = torch.get_rng_state()
        torch.manual_seed(44)

    all_acc = torch.stack(all_acc, axis=0)

    return all_acc, all_acc.mean()

def main(args, sha):
    # set cudnn benchmark True
    torch.backends.cudnn.benchmark = True

    # model
    # No imagenet pretraining since there is overlap between CUB and imagenet
    model = SiameseNet(pretrained=False) 
    model.cuda()

    if args.attr_pred:
        attr_branch = nn.Linear(FEAT_DIM, NUM_ATTR)
        attr_branch.train()
        attr_branch.cuda()
        attr_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = optim.Adam(
            list(model.parameters()) + list(attr_branch.parameters()),
            lr=args.lr)
    else:
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr) #use 0.001

    # train loss
    criterion = nn.MarginRankingLoss(margin=args.margin)
    # use loss = criterion(dist1, dist2, torch.FloatTensor([-1]))

    # resume
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.cuda()
            start_epoch = checkpoint['epoch']+1
            if args.attr_pred:
                attr_branch.load_state_dict(
                    checkpoint['attr_branch_state_dict'])
            print('Resuming from epoch {:d}'.format(start_epoch))

    
    # Dataset and DataLoader
    # Validation
    if args.test_split:
        val_datafile = 'data/novel.json'
    else:
        val_datafile = 'data/val.json'
    val_dataset = ValDataset(
        val_datafile, args.num_val_ep, 
        args.n_way, args.n_support, args.n_query)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=12,
        shuffle=False, pin_memory=True)

    if args.eval_only:
        print('Validating')
        all_acc, val_acc = validate(args, val_loader, model)
        print('Average Accuracy : {:.2f}'.format(100. * val_acc))
        print('Std Accuracy : {:.2f}'.format(100. * all_acc.std()))
        print('95% confidence interval : {:.2f}'.format(196. * all_acc.std()))
        return

    # Train
    train_datafile = 'data/base.json'
    if args.attr_pred:
        train_dataset = TripletDatasetwAttr(
            datafile=train_datafile, aug=args.data_aug)
    else:
        train_dataset = TripletDataset(
            datafile=train_datafile, aug=args.data_aug)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=12, 
        shuffle=True, pin_memory=True)

    model.train()
    best_acc = 0
    for e in range(start_epoch, args.num_epoch):
        print('Epoch {:d} : Training'.format(e))
        if args.attr_pred:
            train_loss = train(
                args, train_loader, [model, attr_branch], 
                [criterion, attr_criterion], optimizer)
        else:
            train_loss = train(args, train_loader, model, criterion, optimizer)

        if (e - start_epoch) % args.log_step == 0:
            print('Epoch {:d} : Validating'.format(e))
            _, val_acc = validate(args, val_loader, model)
            
            print('Epoch {:d} : Saving results'.format(e))
            
            if e == start_epoch:
                os.makedirs(args.save_dir, exist_ok=True)
                save_expt_info(args, sha)
            
            if args.attr_pred:
                attr_branch_state_dict = attr_branch.state_dict()
            else:
                attr_branch_state_dict = None

            torch.save({
                'epoch' : e,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'attr_branch_state_dict' : attr_branch_state_dict,
                'criterion' : criterion, 
                'val_acc' : val_acc,
                'train_loss' : train_loss,
                'best_acc' : best_acc,
                }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

            log_str = get_log_str(
                args, 
                log_info={
                    'epoch' : e,
                    'train_loss' : train_loss, 
                    'val_acc' : val_acc
                })

            print(log_str)
            write_to_log(args, log_str)

            if val_acc > best_acc:
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'), 
                    os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_acc = val_acc


if __name__ == "__main__":
    args = parse_args_siamese()
    sha = get_sha()
    main(args, sha)