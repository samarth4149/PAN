from utils.misc import get_degree_supports
from utils.misc import process_supports
import numpy as np
import torch

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, args, features, img_labels,
                 num_ep, n_way, n_support, n_query):
        # Note that n_support in the arguments is the number of support examples
        super(ValDataset, self).__init__()
        self.features = features
        self.img_labels = img_labels
        self.num_ep = num_ep
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.idxs = np.arange(len(self.img_labels))
        self.classes = np.unique(self.img_labels)

    def __len__(self):
        return self.num_ep

    def __getitem__(self, idx):
        # first support examples and then query examples
        classes = self.classes[torch.randperm(len(self.classes))[:self.n_way]]
        examples = []
        pos_idxs = np.array([(i, j) for i in range(self.n_support)
                             for j in range(i + 1, self.n_support)])
        all_pos_idxs = []
        for i, cl in enumerate(classes):
            curr_idxs = self.idxs[self.img_labels == cl]
            curr_idxs = curr_idxs[torch.randperm(
                len(curr_idxs))[:(self.n_query + self.n_support)]]
            examples.append(self.features[curr_idxs])
            # add edge pairs for current class
            all_pos_idxs.append(pos_idxs + i*(self.n_query + self.n_support))


        # examples : (n_way * (n_support + n_query)) x feat_size
        examples = torch.cat(examples, axis=0)
        all_pos_idxs = np.concatenate(all_pos_idxs)

        query_idxs = []
        for i in range(len(classes)):
            for j in range(self.n_support, self.n_query + self.n_support):
                for k in range(len(classes)):
                    for l in range(self.n_support):
                        query_idxs.append((i*(self.n_query+self.n_support)+j,
                                           k*(self.n_query+self.n_support)+l))

        query_idxs = np.array(query_idxs)

        return examples, all_pos_idxs, query_idxs
