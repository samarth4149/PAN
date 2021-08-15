import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets.dataset_imgs import SimpleDataset
from datasets.dataset_imgs import TripletDataset
from models.siamese import SiameseNet

parser = argparse.ArgumentParser(
    description='Feature extractor')
parser.add_argument('--model', type=str,
                    default='expts/w_data_aug/model-best.pth.tar',
                    help='Path to load model from')
parser.add_argument('--split', type=str,
                    choices=['base', 'novel', 'val'],
                    default='base',
                    help='split for which features are to be extracted')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch_size to use for extracting features')
args = parser.parse_args()

model = SiameseNet(pretrained=False)
if not os.path.isfile(args.model):
    raise Exception('No model found at {}'.format(args.model))

datafile = 'data/{}.json'.format(args.split)


tmp_dataset = TripletDataset(datafile, aug=False)
all_pos_pairs = torch.LongTensor(tmp_dataset.pos_pair_idxs)
pos_pair_idxs = torch.randperm(len(all_pos_pairs))
if args.split == 'base':
    n_train = len(pos_pair_idxs)//2
else:
    n_train = len(pos_pair_idxs)

# Half the positive pairs are used for message passing
pos_pairs_train = all_pos_pairs[pos_pair_idxs[:n_train]]
pos_pairs_train2 = pos_pairs_train[:, [1,0]] # edge in the other direction
pos_pairs_train = torch.cat([pos_pairs_train, pos_pairs_train2])
adj_idxs = pos_pairs_train

if args.split == 'base':
    # And the remaining half for prediction
    pos_pairs_eval = all_pos_pairs[pos_pair_idxs[n_train :]]
    neg_pairs_eval = []
    for i, (u, v) in enumerate(pos_pairs_eval):
        pos_cls = tmp_dataset.img_labels[u]
        w = tmp_dataset.sample_negative(pos_cls)
        # either u or v can be the pos example
        neg_pairs_eval.append((np.random.choice([u, v]), w))
    neg_pairs_eval = torch.LongTensor(neg_pairs_eval)

# For extracting features
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['model_state_dict'])
model.cuda()

dataset = SimpleDataset(datafile)

loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, num_workers=12,
    shuffle=False, pin_memory=True)

all_features = []

print('Extracting features for {} split'.format(args.split))

with torch.no_grad(), tqdm(total=len(loader)) as pbar:
    for i, imgs in enumerate(loader):
        imgs = imgs.cuda()
        feats = model.get_features(imgs)
        all_features.append(feats)
        pbar.update(1)

all_features = torch.cat(all_features).cpu()

save_dict = {
    'features' : all_features,
    'adj_idxs' : adj_idxs,
    'img_labels' : tmp_dataset.img_labels
}

if args.split == 'base':
    save_dict.update({
        'pos_idxs' : pos_pairs_eval,
        'neg_idxs' : neg_pairs_eval
    })

torch.save(save_dict, 'data/{}_feats.pt'.format(args.split))
