from PIL import Image
from torchvision import transforms
import json
import numpy as np
import torch
import torch.utils.data
import os

class TripletDataset(torch.utils.data.Dataset):
    """
    Dataset to obtain image triplets for training the Siamese Network
    """
    def __init__(self, datafile, aug):
        with open(datafile, 'r') as f:
            self.meta = json.load(f)

        self.input_size = (224, 224)

        if aug:
            self.transform = transforms.Compose([
                # NOTE : range of scale seems too intense for RandomResizedCrop
                transforms.RandomResizedCrop(self.input_size),  
                transforms.ColorJitter(brightness=0.4, 
                                       contrast=0.4,
                                       saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(1.15*self.input_size[0]), 
                                   int(1.15*self.input_size[1]))),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        self.img_labels = np.array(self.meta['image_labels'])
        self.idxs = np.arange(len(self.img_labels))
        self.pos_pair_idxs = []
        self.populate_pos_pairs()

    def populate_pos_pairs(self):
        for i in range(len(self.img_labels)):
            for j in range(i+1, len(self.img_labels)):
                if self.img_labels[j] != self.img_labels[i]:
                    break
                self.pos_pair_idxs.append((i, j))

    def sample_negative(self, curr_label):
        tmp = self.idxs[self.img_labels!=curr_label]
        return tmp[torch.randint(high=len(tmp), size=(1, ))]

    def get_img(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.pos_pair_idxs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pos_pair_idxs[idx]
        curr_label = self.img_labels[idx1]
        idx3 = self.sample_negative(curr_label)

        img1 = self.get_img(idx1)
        img2 = self.get_img(idx2)
        img3 = self.get_img(idx3)

        return img1, img2, img3


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, num_ep, n_way, n_support, n_query,
                 transform=None):
        """
        Args:
            datafile: json file containing metadata
            num_ep: number of episodes
            n_way: number of classes for classification
            n_support: number of support examples
            n_query: number of query examples
        """
        self.num_ep = num_ep
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        self.input_size = (224, 224)

        with open(datafile, 'r') as f:
            self.meta = json.load(f)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.img_labels = np.array(self.meta['image_labels'])
        self.idxs = np.arange(len(self.img_labels))
        self.classes = np.unique(self.img_labels)

    def get_img(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __getitem__(self, idx):
        # sample n_way classes
        classes = self.classes[torch.randperm(len(self.classes))[:self.n_way]]
        examples = []

        for cl in classes:
            curr_idxs = self.idxs[self.img_labels == cl]
            curr_idxs = curr_idxs[torch.randperm(
                len(curr_idxs))[:(self.n_query+self.n_support)]]
            curr_examples = []
            for curr_idx in curr_idxs:
                curr_examples.append(self.get_img(curr_idx))
            curr_examples = torch.stack(curr_examples, axis=0)
            examples.append(curr_examples)

        examples = torch.stack(examples, axis=0)

        # oTODO : return classes to find original classes for examples
        return examples # n_way x (n_query+n_support) X C x H x W

    def __len__(self):
        return self.num_ep

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, datafile):
        with open(datafile, 'r') as f:
            self.meta = json.load(f)

        self.input_size = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize((int(1.15 * self.input_size[0]),
                               int(1.15 * self.input_size[1]))),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.img_labels = np.array(self.meta['image_labels'])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))


class TripletDatasetwAttr(torch.utils.data.Dataset):
    """
    Dataset returning triplets of images along with their attribute annotations
    for training the multitask model
    """
    def __init__(self, datafile, aug):
        with open(datafile, 'r') as f:
            self.meta = json.load(f)

        self.split = os.path.splitext(os.path.basename(datafile))[0]
        attr_file = os.path.join('data', '{}_attr.pt'.format(self.split))
        attr_data = torch.load(attr_file)
        self.attr_labels = attr_data['attr_labels']
        self.certainty_mask = attr_data['certainty_mask']
        self.input_size = (224, 224)

        if aug:
            self.transform = transforms.Compose([
                # NOTE : range of scale seems too intense for RandomResizedCrop
                transforms.RandomResizedCrop(self.input_size),  
                transforms.ColorJitter(brightness=0.4, 
                                       contrast=0.4,
                                       saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((int(1.15*self.input_size[0]), 
                                   int(1.15*self.input_size[1]))),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        self.img_labels = np.array(self.meta['image_labels'])
        self.idxs = np.arange(len(self.img_labels))
        self.pos_pair_idxs = []
        self.populate_pos_pairs()

    def populate_pos_pairs(self):
        for i in range(len(self.img_labels)):
            for j in range(i+1, len(self.img_labels)):
                if self.img_labels[j] != self.img_labels[i]:
                    break
                self.pos_pair_idxs.append((i, j))

    def sample_negative(self, curr_label):
        tmp = self.idxs[self.img_labels!=curr_label]
        return tmp[torch.randint(high=len(tmp), size=(1, ))]

    def get_img(self, idx):
        img_path = self.meta['image_names'][idx]
        return self.transform(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return len(self.pos_pair_idxs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pos_pair_idxs[idx]
        curr_label = self.img_labels[idx1]
        idx3 = self.sample_negative(curr_label)

        img1 = self.get_img(idx1)
        img2 = self.get_img(idx2)
        img3 = self.get_img(idx3)

        attr_lab1 = self.attr_labels[idx1]
        attr_lab2 = self.attr_labels[idx2]
        attr_lab3 = self.attr_labels[idx3]

        cm1 = self.certainty_mask[idx1]
        cm2 = self.certainty_mask[idx2]
        cm3 = self.certainty_mask[idx3]

        return img1, img2, img3, attr_lab1, attr_lab2, attr_lab3, cm1, cm2, cm3