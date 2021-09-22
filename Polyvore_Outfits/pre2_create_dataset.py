import os
import json
import scipy as sp
from scipy.sparse import lil_matrix, save_npz, csr_matrix
import argparse
import pickle as pkl
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='test', choices=['train', 'valid', 'test'])
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                        help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
args = parser.parse_args()

save_path = 'data/polyvore_outfits/dataset/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

dataset_path = 'data/polyvore_outfits/{}/'.format(args.polyvore_split)
json_file = dataset_path + '{}.json'.format(args.phase)

train_file = dataset_path + 'train.json'
valid_file = dataset_path + 'valid.json'
test_file = dataset_path + 'test.json'
"""train_file = train_on_dup.json--> train.json"""
with open(json_file) as f:
    json_data = json.load(f)

# load the features extracted with 'extract_features.py'
feat_pkl = os.path.join(save_path, 'imgs_featdict_{}.pkl'.format(args.phase))
if os.path.exists(feat_pkl):
    with open(feat_pkl, 'rb') as handle:
        feat_dict = pkl.load(handle)
else:
    raise 'The extracted features file {} does not exist'.format(feat_pkl)

relations = {}
id2idx = {}
idx = 0
features = []

# map an image id to their ids with format 'OUTFIT-ID_INDEX'
map_id2their = {}

for outfit in json_data:
    outfit_ids = set()
    for item in outfit['items']:
        # get id from the image url
        item_id = item['item_id']
        item_id = int(item_id)
        outfit_ids.add(item_id)
        #add item_id(s) of EACH set_id to outfit_ids
        map_id2their[item_id] = '{}_{}'.format(outfit['set_id'], item['index'])
        #map_id2their: map item id to set_id with index
    for item_id in outfit_ids:
        if item_id not in relations:
            relations[item_id] = set()
            img_feats = feat_dict[str(item_id)] 
            # TODO, REMOVE
            #cat_vector = cat_vectors[cat_dict[id]]
            #feats = np.concatenate((cat_vector, img_feats))
            features.append(img_feats)
            # map this id to a sequential index
            id2idx[item_id] = idx#:id2idx: item_id --> idx in the feature array
            idx += 1

        relations[item_id].update(outfit_ids)
        relations[item_id].remove(item_id)
        """relations means the the specific item_id with the rest item_id(s), 
        all these ids belong to one set_id"""

map_file = save_path + 'id2idx_{}.json'.format(args.phase)#id2idx: id 2 index in array
with open(map_file, 'w') as f:
    json.dump(id2idx, f)
map_file = save_path + 'id2their_{}.json'.format(args.phase)#id2their: id 2 set_id with index
with open(map_file, 'w') as f:
    json.dump(map_id2their, f)

# create sparse matrix that will represent the adj matrix
sp_adj = lil_matrix((idx, idx))
features_mat = np.zeros((idx, 2048))
print('Filling the values of the sparse adj matrix')
for rel in relations:
    rel_list = relations[rel]
    from_idx = id2idx[rel]
    features_mat[from_idx] = features[from_idx]

    for related in rel_list:
        to_idx = id2idx[related]

        sp_adj[from_idx, to_idx] = 1
        sp_adj[to_idx, from_idx] = 1 # because it is symmetric

print('Done!')

density = sp_adj.sum() / (sp_adj.shape[0] * sp_adj.shape[1])
print('Sparse density: {}'.format(density))

# now save the adj matrix
save_adj_file = save_path + 'adj_{}.npz'.format(args.phase)
sp_adj = sp_adj.tocsr()
save_npz(save_adj_file, sp_adj)

save_feat_file = save_path + 'features_{}'.format(args.phase)
sp_feat = csr_matrix(features_mat)
save_npz(save_feat_file, sp_feat)



if args.phase == 'test':
    from pre3_test_functions import get_compats, get_questions
    def create_test(polyvore_split):
        """
        outfits[i]:
                list: ['102972440', '103394173', '91303250', '94989504', '103184729']
                float: 1 
        questions[i]:
                list: ['102972440', '91303250', '94989504', '103184729'] (query items)
                list: ['103394173', '127110314', '156949162', '96522232'] (answer items)
                list: [2, 1, 4, 1] (category of each answer)
                int: 2(right category)
        """
        
        # build the question indexes
        questions = get_questions(polyvore_split)
        for i in range(len(questions)): # for each question
            assert len(questions[i]) == 4
            for j in range(2): # questions list (j==0) or answers list (j==1)
                for z in range(len(questions[i][j])): # for each id in the list
                    item_id = int(questions[i][j][z])
                    questions[i][j][z] = id2idx[item_id] # map the id to the node index

        questions_file = save_path + 'questions_{}.json'.format(args.phase) 
        questions_file = save_path + 'questions_RESAMPLED_{}.json'.format(args.phase)
        with open(questions_file, 'w') as f:
            json.dump(questions, f)
    
        # outfit compat task
        outfits = get_compats(polyvore_split)
        for i in range(len(outfits)): # for each outfit
            for j in range(len(outfits[i][0])):
                id = int(outfits[i][0][j])
                outfits[i][0][j] = id2idx[id]
        compat_file = save_path + 'compatibility_{}.json'.format(args.phase)
        compat_file = save_path + 'compatibility_RESAMPLED_{}.json'.format(args.phase)
        print(compat_file)
        with open(compat_file, 'w') as f:
            json.dump(outfits, f)
    
    create_test(args.polyvore_split)








