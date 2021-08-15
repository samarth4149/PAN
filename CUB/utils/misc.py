import copy

import numpy as np
import scipy.sparse as sp
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='avg_meter_var', fmt=':f'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_degree_supports(adj_idxs, shape, k, adj_self_con=True, verbose=False):
    adj_idxs = np.array(adj_idxs)
    if len(adj_idxs) == 0:
        adj = sp.coo_matrix(([], ([], [])), shape=shape) 
    else:
        adj = sp.coo_matrix(
            (np.ones(len(adj_idxs)), (adj_idxs[:, 0], adj_idxs[:, 1])),
            shape=shape)
    if verbose:
        print('Computing adj matrices up to {}th degree'.format(k))
    supports = [sp.identity(adj.shape[0])]
    if k == 0: # return Identity matrix (no message passing)
        return supports
    assert k > 0
    supports = [sp.identity(adj.shape[0]), adj.astype(np.float64)
                + adj_self_con*sp.identity(adj.shape[0])]

    # Currently unused when k == 1
    prev_power = adj
    for i in range(k-1):
        pow = prev_power.dot(adj)
        new_adj = ((pow) == 1).astype(np.float64)
        new_adj.setdiag(0)
        new_adj.eliminate_zeros()
        supports.append(new_adj)
        prev_power = pow
    ###############################

    return supports

def normalize_nonsym_adj(adj):
    degree = np.asarray(adj.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv_sqrt = 1. / np.sqrt(degree)
    degree_inv_sqrt_mat = sp.diags([degree_inv_sqrt], [0])

    degree_inv = degree_inv_sqrt_mat.dot(degree_inv_sqrt_mat)

    adj_norm = degree_inv.dot(adj)

    return adj_norm

def support_dropout(sup, do, edge_drop=False):
    sup = sp.tril(sup)
    assert do > 0.0 and do < 1.0
    n_nodes = sup.shape[0]
    # nodes that I want to isolate
    isolate = np.random.choice(range(n_nodes), int(n_nodes*do), replace=False)
    nnz_rows, nnz_cols = sup.nonzero()

    # mask the nodes that have been selected
    mask = np.in1d(nnz_rows, isolate)
    mask += np.in1d(nnz_cols, isolate)
    assert mask.shape[0] == sup.data.shape[0]

    sup.data[mask] = 0
    sup.eliminate_zeros()

    if edge_drop:
        prob = np.random.uniform(0, 1, size=sup.data.shape)
        remove = prob < do
        sup.data[remove] = 0
        sup.eliminate_zeros()

    sup = sup + sup.transpose()
    return sup

def process_supports(supports, do, on_gpu=True, edge_drop=False):
    """
    Support dropout, normalize and convert to torch sparse tensors
    Note that do is the drop probability
    """
    # Make a copy so supports passed in does not get modified
    ret_supports = copy.deepcopy(supports)
    for i in range(len(ret_supports)):
        if i>=1:
            if do > 0.:
                ret_supports[i] = support_dropout(
                    ret_supports[i], do, edge_drop)
            ret_supports[i] = normalize_nonsym_adj(ret_supports[i])

        coo = sp.coo_matrix(ret_supports[i])
        indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
        values = torch.FloatTensor(coo.data)
        shape = coo.shape
        ret_supports[i] = torch.sparse.FloatTensor(
            indices, values, torch.Size(shape))
        if on_gpu:
            ret_supports[i] = ret_supports[i].cuda()
    return ret_supports