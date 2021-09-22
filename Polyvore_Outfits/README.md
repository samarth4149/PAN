# Outfit compatibility using PAN

## Requirements

The model is implemented with Tensorflow. Relevant libraries can be installed by:

    pip install -r requirements.txt

## Data

### Polyvore Outfits

The Polyvore Outfits dataset can be downloaded from [google drive](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing). 

We include the re-sampled validation and testing set under path "data/polyvore_outfits/nondisjoint". The Polyvore Outfits data should also be put under "data/polyvore_outfits".
```
|-- data
    |-- polyvore_outfits
        |-- nondisjoint
```

## Preprocess

Extract ResNet50 features

    python pre1_extract_features.py --phase train
    python pre1_extract_features.py --phase valid
    python pre1_extract_features.py --phase test

Create the data for training and testing

    python pre2_create_dataset.py --phase train
    python pre2_create_dataset.py --phase valid
    python pre2_create_dataset.py --phase test

## Training

The model is trained with the following command:

    python train.py -sdir logs/206_10 -n_output 206 -c_loss True

The most relevant arguments are the following:

 - `-sdir `: Directory for saving tensorflow summaries
 - `-n_output` :number of output nodes
 - `-c_loss`: supervised or unsupervised model

## Evaluation:

A model can be evaluated for the FITB task and the Compatibility prediction task.
FITB task:

    python test_fitb.py -lf logs/206_10
    python test_fitb.py -lf logs/206_10 -resampled True

and for the compatibility task with:

    python test_compatibility.py -lf logs/206_10
    python test_compatibility.py -lf logs/206_10 -resampled True



Our project is based on [Context-Aware Visual Compatibility Prediction](https://github.com/gcucurull/visual-compatibility) .

The modifications are mainly reflected on "model/layers.py" and "model/CompatibilityGAE.py".