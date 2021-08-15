import math
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.dataset_features import ValDataset
from models.pan import PAN
from utils.ioutils import get_log_str
from utils.ioutils import get_sha
from utils.ioutils import parse_args_vc
from utils.ioutils import save_expt_info
from utils.ioutils import write_to_log
from utils.misc import get_degree_supports
from utils.misc import process_supports


def validate(args, loader, model):
    model.eval()
    n_support = args.n_support
    n_query = args.n_query
    n_way = args.n_way
    all_acc = []
    
    if args.fix_val:
        rng_state = torch.get_rng_state()
        torch.manual_seed(44)

    with torch.no_grad(), tqdm(total=len(loader)) as pbar:
        for episode, (examples, all_pos_idxs, query_idxs) in enumerate(loader):
            # Assuming batch size of 1
            # examples : n_way * (n_support + n_query) x feat_dim
            examples = examples[0].contiguous().cuda()
            all_pos_idxs = all_pos_idxs[0]
            query_idxs = query_idxs[0].cuda()

            # get supports
            supports = get_degree_supports(
                all_pos_idxs,
                shape=(len(examples), len(examples)),
                k=args.degree)
            supports = process_supports(
                supports, do=0.)  # no support drop in val
            output_scores, _, _ = model(examples, supports,
                                     query_idxs[:, 0], query_idxs[:, 1])

            output_scores = output_scores.view(n_way*n_query, n_way, n_support)
            output_scores = output_scores.mean(axis=2)
            true_labels = torch.arange(n_way*n_query).cuda()/n_query
            predicted_labels = output_scores.argmax(axis=1)
            
            accuracy = (true_labels == predicted_labels)\
                       .sum()/float(len(true_labels))
            
            all_acc.append(accuracy.cpu())
            pbar.update(1)
    
    if args.fix_val:
        # reset the rng state
        torch.set_rng_state(rng_state)

    all_acc = torch.stack(all_acc, axis=0)

    return all_acc, all_acc.mean()

def main(args, sha):

    # Only one of the following two if statements gets used
    # if attribute labels are used, set nout to num_sup_at
    if args.use_at_lab:
        if args.label_func == 'AND_XOR':
            args.num_sup_at *= 2 # twice the number of supervised nodes
        args.nout = args.num_sup_at
        

    # if hybrid model is used, add num_sup_at nodes to nout
    if args.hybrid:
        args.use_at_lab = True
        if args.label_func == 'AND_XOR':
            args.num_sup_at *= 2
        args.nout += args.num_sup_at

    # set cudnn benchmark True
    torch.backends.cudnn.benchmark = True

    if args.test_split:
        val_data = torch.load('data/novel_feats.pt')
    else:
        val_data = torch.load('data/val_feats.pt')

    train_data = torch.load('data/base_feats.pt')
    if args.use_at_lab:
        train_attr_data = torch.load('data/base_attr.pt')

    # Standardize features
    means= train_data['features'].mean(axis=0)
    stds = train_data['features'].std(axis=0)
    eps = 1e-6
    train_data['features'] = (train_data['features'] - means)/(stds+eps)
    val_data['features'] = (val_data['features'] - means)/(stds+eps)

    # get support matrices
    train_supports = get_degree_supports(
        train_data['adj_idxs'],
        shape=(len(train_data['features']), len(train_data['features'])),
        k=args.degree)

    # model
    model = PAN(
        input_dim=train_data['features'].shape[1],
        num_supports=len(train_supports), args=args)
    model.cuda()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # use 0.001

    # loss func
    criterion = nn.BCELoss()
    criterion_no_red = nn.BCELoss(reduction='none')

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
            print('Resuming from epoch {:d}'.format(start_epoch))
        else:
            raise Exception('No checkpoint found at {}'.format(args.resume))

    val_dataset = ValDataset(args, val_data['features'],
                             val_data['img_labels'],
                             args.num_val_ep, args.n_way,
                             args.n_support, args.n_query)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=12,
        shuffle=False, pin_memory=True)

    if args.eval_only:
        print('Validating')
        all_acc, val_acc = validate(args, val_loader, model)
        print('Average Accuracy : {:.2f}'.format(100. * val_acc))
        print('Std Accuracy : {:.2f}'.format(100. * all_acc.std()))
        print('95% confidence interval : {:.2f}'.format(
            196. * all_acc.std() / math.sqrt(args.num_val_ep)))
        return all_acc, val_acc

    train_feats = train_data['features'].cuda()
    train_pos_idxs = train_data['pos_idxs']
    train_neg_idxs = train_data['neg_idxs']

    labels = torch.cat([torch.ones(len(train_pos_idxs)),
                        torch.zeros(len(train_neg_idxs))]).cuda()
    train_row_idxs = torch.cat(
        [train_pos_idxs[:, 0], train_neg_idxs[:, 0]]).cuda()
    train_col_idxs = torch.cat(
        [train_pos_idxs[:, 1], train_neg_idxs[:, 1]]).cuda()

    if args.use_at_lab:
        # Generate a fixed random permutation of the attributes
        rng_state = np.random.get_state()
        np.random.seed(44)
        attr_perm = np.random.permutation(
            train_attr_data['attr_labels'].shape[1])
        np.random.set_state(rng_state)

        # populate pairwise attribute labels
        img_attr_labels = train_attr_data['attr_labels'].type(torch.BoolTensor)
        img_certainty_masks = train_attr_data['certainty_mask']\
                              .type(torch.BoolTensor)
        img_attr_labels = img_attr_labels[:, attr_perm]
        img_certainty_masks = img_certainty_masks[:, attr_perm]
        
        pair_attr_labels = []
        pair_certainty_masks = []
        # pairwise attribute labels and certainty masks
        for id1, id2 in zip(train_row_idxs, train_col_idxs):
            if args.label_func == 'OR':
                lbl = img_attr_labels[id1] | img_attr_labels[id2]
                lbl = lbl[:args.num_sup_at]
            elif args.label_func == 'AND':
                lbl = img_attr_labels[id1] & img_attr_labels[id2]
                lbl = lbl[:args.num_sup_at]
            elif args.label_func == 'XNOR':
                lbl = ~(img_attr_labels[id1] ^ img_attr_labels[id2])
                lbl = lbl[:args.num_sup_at]
            elif args.label_func == 'XOR':
                lbl = (img_attr_labels[id1] ^ img_attr_labels[id2])
                lbl = lbl[:args.num_sup_at]

            # Only if both image labels are certain
            cm = img_certainty_masks[id1] & img_certainty_masks[id2]
            cm = cm[:args.num_sup_at]

            if args.label_func == 'AND_XOR':
                assert args.num_sup_at == 624, \
                    'Partial supervised attributes not supported yet'
                lbl = torch.cat([
                    img_attr_labels[id1] & img_attr_labels[id2],
                    img_attr_labels[id1] ^ img_attr_labels[id2]])
                cm = torch.cat([cm, cm])

            if args.hybrid:
                lbl = torch.cat((lbl, torch.zeros(
                    args.nout - len(lbl), dtype=bool)))
                cm = torch.cat((cm, torch.zeros(
                    args.nout - len(cm), dtype=bool)))
            
            pair_attr_labels.append(lbl)
            pair_certainty_masks.append(cm)

        pair_attr_labels = torch.stack(
            pair_attr_labels, axis=0).type(torch.FloatTensor).cuda()
        pair_certainty_masks = torch.stack(
            pair_certainty_masks, axis=0).type(torch.FloatTensor).cuda()

    best_acc = 0
    for e in range(start_epoch, args.num_epoch):
        model.train()
        # Normalize supports after support dropout
        # NOTE: this would return degree supports on the GPU
        processed_train_supports = process_supports(
            train_supports, do=args.support_dropout)

        out, attr, _ = model(train_feats, processed_train_supports,
                             train_row_idxs, train_col_idxs)

        edge_loss = criterion(out, labels)
        
        if args.use_at_lab:
            attr_loss = criterion_no_red(attr, pair_attr_labels)
            attr_loss = torch.sum(pair_certainty_masks * attr_loss, axis=1)\
                        / (torch.sum(pair_certainty_masks, axis=1) + eps)
            attr_loss = torch.mean(attr_loss, axis=0)

            loss = edge_loss + args.lam * attr_loss
        else:
            loss = edge_loss

        train_loss = loss.item()

        tqdm.write('Loss at epoch {} : {}'.format(e, train_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e-start_epoch)%args.log_step == 0:
            print('Epoch {:d} : Validating'.format(e))
            _, val_acc = validate(args, val_loader, model)

            print('Epoch {:d} : Saving results'.format(e))

            if e == start_epoch:
                os.makedirs(args.save_dir, exist_ok=True)
                save_expt_info(args, sha)

            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'best_acc': best_acc,
            }, os.path.join(args.save_dir, 'checkpoint.pth.tar'))

            log_str = get_log_str(
                args,
                log_info={
                    'epoch': e,
                    'train_loss': train_loss,
                    'val_acc': val_acc
                })

            print(log_str)
            write_to_log(args, log_str)

            if val_acc > best_acc:
                shutil.copyfile(
                    os.path.join(args.save_dir, 'checkpoint.pth.tar'),
                    os.path.join(args.save_dir, 'model-best.pth.tar'))
                best_acc = val_acc

if __name__ == "__main__":
    args = parse_args_vc()
    sha = get_sha()
    main(args, sha)
