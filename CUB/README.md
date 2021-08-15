# Few shot learning using PAN
## Requirements
To install the requirements for this project. (Virtual environment encouraged, e.g. conda) 

```
pip install -r requirements.txt
```

## Data
Copy the directory `CUB_filelists` to the location where you would like to download 
CUB data (\~1.2 GB)

```
cd CUB_filelists
bash download_CUB.sh
``` 

Then copy the files `base.json`, `val.json` and `novel.json` to a directory named `data` in the base folder of the project.

## To run the siamese model
The file main_siamese.py can be used to train the siamese model. 
```
usage: main_siamese.py [-h] [--eval_only] [--test_split] [--n_way N_WAY]
                       [--n_support N_SUPPORT] [--n_query N_QUERY]
                       [--num_val_ep NUM_VAL_EP] [--num_epoch NUM_EPOCH]
                       [--data_aug] [-b BATCH_SIZE] [--margin MARGIN]
                       [--resume PATH] [--lr LR] [--attr_pred] [--lam LAM]
                       [--fix_val] [--save_dir SAVE_DIR] [--log_step LOG_STEP]

Siamese Baseline for Few shot learning

optional arguments:
  -h, --help            show this help message and exit
  --eval_only           Do only evaluation (presumably using a loaded model)
  --test_split          Use the test split for validation
  --n_way N_WAY         class num to classify for testing (validation)
  --n_support N_SUPPORT
                        number of labeled data in each class
  --n_query N_QUERY     number of query examples in each class
  --num_val_ep NUM_VAL_EP
                        number of episodes for validation
  --num_epoch NUM_EPOCH
                        Number of epochs for training
  --data_aug            Train with data augmentation
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Minibatch size used
  --margin MARGIN       Margin to use for triplet loss
  --resume PATH         path to latest checkpoint (default: none)
  --lr LR, --learning-rate LR
                        initial learning rate
  --attr_pred           Use an auxiliary attribute predictor branch
  --lam LAM             Weight of attribute loss in the final loss func
  --fix_val             Each time validation is done, the same classes are
                        chosen. Numpy random seed is fixed.
  --save_dir SAVE_DIR   path to directory for saving results
  --log_step LOG_STEP   log each <log_step> epoch

```
Command to run training:
```
python main_siamese.py --save_dir path/to/results --num_epoch 200 --data_aug
```
Command to run inference:
```
python main_siamese.py --resume path/to/saved_model --eval_only --test_split --num_val_ep 600
```

## Extract features from the siamese model
Run
```
for split in base val novel;
do
    python extract_features.py --model path/to/model --split $split;
done;
```
where `path/to/model` is the path of the model file to use (saved from training the siamese model).

## Extract attribute labels for CUB images
Inside the directory `CUB_filelist`, run
```
for split in base val novel;
do
    python get_attributes.py --split $split;
done;
```

## To run the PAN model
To run the pan model, 
```
usage: main_pan.py [-h] [--eval_only] [--test_split] [--n_way N_WAY]
                   [--n_support N_SUPPORT] [--n_query N_QUERY]
                   [--num_val_ep NUM_VAL_EP] [--num_epoch NUM_EPOCH]
                   [--resume PATH] [--lr LR] [--fix_val]
                   [-hi HIDDEN [HIDDEN ...]] [-deg DEGREE] [-do DROPOUT]
                   [-sup_do SUPPORT_DROPOUT] [--hybrid] [--use_at_lab]
                   [--label_func [{OR,AND,XNOR,AND_XOR}]]
                   [--num_sup_at NUM_SUP_AT] [--nout NOUT] [--lam LAM]
                   [--no_ge] [--save_dir SAVE_DIR] [--log_step LOG_STEP]

PAN model for few shot learning

optional arguments:
  -h, --help            show this help message and exit
  --eval_only           Do only evaluation (presumably using a loaded model)
  --test_split          Use the test split for validation
  --n_way N_WAY         class num to classify for testing (validation)
  --n_support N_SUPPORT
                        number of labeled data in each class
  --n_query N_QUERY     number of query examples in each class
  --num_val_ep NUM_VAL_EP
                        number of episodes for validation
  --num_epoch NUM_EPOCH
                        Number of epochs for training
  --resume PATH         path to latest checkpoint (default: none)
  --lr LR, --learning-rate LR
                        initial learning rate
  --fix_val             Each time validation is done, the same classes are
                        chosen. Numpy random seed is fixed.
  -hi HIDDEN [HIDDEN ...], --hidden HIDDEN [HIDDEN ...]
                        Number of hidden units in the GCN layers.
  -deg DEGREE, --degree DEGREE
                        Degree of the convolution (Number of supports)
  -do DROPOUT, --dropout DROPOUT
                        Dropout fraction
  -sup_do SUPPORT_DROPOUT, --support_dropout SUPPORT_DROPOUT
                        Use dropout on the support matrices, dropping all the
                        connections from some nodes
  --hybrid              Whether to a hybrid model. In this case nout is set to
                        number of attributes + nout and use_at_lab is set to
                        True
  --use_at_lab          Whether to use attribute labels. In this case nout is
                        set to number of attributes
  --label_func [{OR,AND,XNOR,AND_XOR}]
                        Logical function to combine attribute labels
  --num_sup_at NUM_SUP_AT
                        Use only the first num_sup_at attribute labels
  --nout NOUT           Number of outputs of the MLP decoder, typically same
                        as number of attributes
  --lam LAM             Weight of attribute loss in the final loss func
  --no_ge               Whether to use graph image encoder
  --save_dir SAVE_DIR   path to directory for saving results
  --log_step LOG_STEP   log each <log_step> epoch

```

### Performance Comparison 

| Method           | 5-way 5-shot accuracy |
|------------------|-----------------------|
| Baseline++       | 83.58                 |
| ProtoNet         | 87.42                 |
| Trinet           | 84.10                 |
| TEAM             | 87.17                 |
| CGAE             | 88.00 $\pm$ 1.13      |
| PAN-Unsupervised | 92.69 $\pm$ 0.28      |
| PAN-Supervised   | 92.77 $\pm$ 0.30      |

Commands for running above models (runnable on our code):
- CGAE : `python main_pan.py --nout=1 --fix_val --save_dir=path/to/results`
- PAN-Unsupervised : `python main_pan.py --no_ge --nout=50 --fix_val --save_dir=path/to/results`
- PAN-Hybrid : `python main_pan.py --hybrid --no_ge --nout=10 --num_sup_at=10 --lam=1e-5 --fix_val --save_dir=path/to/results`
- PAN-Supervised : `python main_pan.py --use_at_lab --no_ge --fix_val --lam=1e-5 --save_dir=path/to/results`

Commands for running inference are similar to that for the Siamese Network (using `eval_only` and `test_split` options)

Options for number of similarity conditions for different model variants : 
- For the unsupervised model, set `NOUT` to the number of similarity conditions. 
- For a supervised model, set `use_at_lab` to `True` and set `num_sup_at` to the number of attributes to use.
- For a hybrid model, set `hybrid` to `True`. `NOUT` in this case would be the number of unsupervised similarity conditions.

 