
# Antimicrobial GNN: Generalizing Peptide-Trained GNNs to Out-of-Distribution Antimicrobial Polymer Classification

### Overview:
This repository contains the code to train various graphical neural networks (AttentiveFP, MPNN, GAT, Weave, GCN) for antimicrobial property classification. 

* ```data_preprocessing/```: contains the database that will be used for train/validation/test split. All preprocessing scripts are included to modify this dataset and ensure that it is in the correct format. The subfolder "peptide_data" creates the peptide dataset, whereas the subfolder "polymer_data" creates the polymer dataset. They are merged together and assigned into classes in "create_db.ipynb". Each subfolder contains documented '.ipynb' files that explain the preprocessing steps.
* ```experiments/```: contains the databases used to create the results in the paper.
* ```inference/```: contains the trained model that will be used during inference
* ```model_hparams```: contains the hyperparameters used during training for each of the five GNN architectures explored.
* ```monomer_data/```: contains the list of RDKit descriptors used to featurize the nodes and edges.
* ```utils/```: contains all files related to training and inference.

### 
To set up the environment, run:
```
conda env create -f environment.yml
conda activate GLAMOUR
```

Alternatively, to set up this environment on a server :

```
chmod +x setup_glamour.sh
./setup_glamour.sh

```
To train the models, follow ```train_example.ipynb```. To run inference, follow ```inference_example.ipynb```.

To run hyperparameter_optimization: 
```
python hyperparameter_optimization.py --seed [SEED] --GPU [GPU #] --model [MODEL_NAME] --num_epochs [# EPOCHS] --num_trials [# TRIALS] --rand_samples [# randomly sampled hyperparameters]
```