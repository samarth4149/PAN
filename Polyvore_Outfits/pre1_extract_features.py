import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from PIL import Image
import time
import pickle as pkl
import json

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', choices=['train', 'test', 'valid'])
args = parser.parse_args()

img_size=224

"""
    train.json: 53306 set ids --> 204679 item ids;
    valid.json: 5000 set ids --> 25132 item ids;
    test.json: 10000 set ids --> 47854 item ids;
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load net to extract features
model = models.resnet50(pretrained=True)
# skip last layer (the classifier)
model = nn.Sequential(*list(model.children())[:-1])
model = model.to(device)
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256), transforms.CenterCrop(img_size),
                transforms.ToTensor(), normalize
            ])

def process_image(im):
    im = transform(im)
    im = im.unsqueeze_(0)
    im = im.to(device)
    out = model(im)
    return out.squeeze()

dataset_path = 'data/polyvore_outfits/'
images_path = dataset_path + 'images/'
json_file = os.path.join(dataset_path, 'nondisjoint/{}.json'.format(args.phase))

with open(json_file) as f:
    data = json.load(f)

save_to = 'data/polyvore_outfits/dataset/'
if not os.path.exists(save_to):
    os.makedirs(save_to)
save_dict = os.path.join(save_to, 'imgs_featdict_{}.pkl'.format(args.phase))
## create a file name to save dictionary

ids = {}

## I use img ids, but the folders use outfit_id/index.png

for outfit in data:
    for item in outfit['items']:
        # get id from the image url
        item_id = item['item_id']
        # item_index = item['index']
        their_id = '{}_{}'.format(outfit['set_id'], item['index'])
        ids[item_id] = their_id



features = {}
count = {}

print('iterating through ids')
i = 0
n_items = len(ids.keys())
with torch.no_grad(): # it is the same as volatile=True for versions before 0.4
    for item_id in ids:
        set_id, index = ids[item_id].split('_') # outfitID_index
        #outfit_id: set_id, index: the indexth image of set_id

        image_path = images_path + '{}.jpg'.format(item_id)
        assert os.path.exists(image_path)

        im = skimage.io.imread(image_path)
        if len(im.shape) == 2:
            im = gray2rgb(im)
        if im.shape[2] == 4:
            im = rgba2rgb(im)

        im = resize(im, (img_size, img_size))#(3,256,256)
        im = img_as_ubyte(im)

        feats = process_image(im).cpu().numpy()

        if item_id not in features:
            features[item_id] = feats
            count[item_id] = 0
        else:
            features[item_id] += feats
        count[item_id] += 1

        if i % 1000 == 0 and i > 0:
            print('{}/{}'.format(i, n_items))
        i += 1

feat_dict = {}
for id in features:
    feats = features[id]
    feats = np.array(feats)/count[id]
    feat_dict[id] = feats

with open(save_dict, 'wb') as handle:
    pkl.dump(feat_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)











