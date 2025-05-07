
# Antimicrobial GNN:

To set up the environment to run this reposity run:
```
conda env create -f environment.yml
conda activate GLAMOUR
```

Alternatively, to set up this environment on a server :

```
chmod +x setup_glamour.sh
./setup_glamour.sh

```
To train the models, follow ```ex.ipynb```. The code has undergone significant revisions and re-organization in the past few days. Currently, the inference pipeline is down and the training is limited to training, validating, and testing on data mixed between peptides and polymers for multi-class classification. The code will be fixed soon to accommodate binary and multi-class classification, as well as training and validating strictly on peptides.