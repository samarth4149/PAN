import csv
import json
import os

import torch

cwd = os.getcwd()
img_name_file = os.path.join(cwd,'CUB_200_2011/images.txt')

name_idx_map = {}
with open(img_name_file, 'r') as f:
    rd = csv.reader(f, delimiter=' ')
    for row in rd:
        name_idx_map[row[1]] = int(row[0])-1

num_imgs = len(name_idx_map)
num_attr = 312
attr_labels = torch.zeros(num_imgs, num_attr)
certainty_mask = torch.zeros(num_imgs, num_attr)
conf_scores = torch.zeros(num_imgs, num_attr)
attr_file = os.path.join(
    cwd,'CUB_200_2011/attributes/image_attribute_labels.txt')

with open(attr_file, 'r') as f:
    rd = csv.reader(f, delimiter=' ')
    for row in rd:
        img_idx = int(row[0])-1
        attr_idx = int(row[1])-1
        conf = int(row[3])
        attr_labels[img_idx, attr_idx] = int(row[2])
        conf_scores[img_idx, attr_idx] = conf
        if conf > 2.:
            certainty_mask[img_idx, attr_idx] = 1

print('Getting image attributes...')
for split in ['base', 'novel', 'val']:
    print('Processing {}'.format(split))
    with open('{}.json'.format(split), 'r') as f:
        meta = json.load(f)
    img_idxs = []
    for img_name in meta['image_names']:
        curr_idx = name_idx_map['/'.join(img_name.split('/')[-2:])]
        img_idxs.append(curr_idx)
    img_idxs = torch.LongTensor(img_idxs)
    torch.save({
        'attr_labels' : attr_labels[img_idxs],
        'certainty_mask' : certainty_mask[img_idxs],
        'conf_scores' : conf_scores[img_idxs]
    }, '{}_attr_test.pt'.format(split))
