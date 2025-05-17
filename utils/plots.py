import pandas as pd
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, confusion_matrix


def plot_loss(self, df):

    epochs = df['epoch'].unique()

    train_loss = df[df['dataset'] == 'train']['loss']
    val_loss = df[df['dataset'] == 'val']['loss']

    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    
    # Add legend
    plt.legend()
    
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    
    plt.savefig(self._model_path + '/loss_fig.png')
    plt.close()

def cm_plot(self, df):

    if self._dataset._ntask == 1:
        lbs = [0,1]
    else:
        lbs = [ i for i in range(int(self._dataset._ntask))]

    if self._dataset._mixed == True:
        data1 = confusion_matrix(df['y_true'].to_numpy(), df['y_pred'].to_numpy(), labels=lbs)
        disp = ConfusionMatrixDisplay(confusion_matrix=data1)
        disp.plot()
        plt.savefig(self._model_path + '/CM_all.png')
        plt.close()

    polymers = df[df["ID"].str.contains("poly", na=False)].copy()

    # split once
    split_ID = polymers["ID"].str.split("_S", n=1, expand=True)
    
    # assign back
    polymers["ID"]     = split_ID[0]
    polymers["sample"] = split_ID[1].astype(int)
    data2 = confusion_matrix(polymers['y_true'].to_numpy(), polymers['y_pred'].to_numpy(), labels=lbs)

    disp = ConfusionMatrixDisplay(confusion_matrix=data2)
    disp.plot()
    plt.savefig(self._model_path + '/CM_all_poly.png')
    plt.close()

    grouped = (polymers
               .groupby("ID", sort=False) # group on ID, keep first-seen order
               .mean(numeric_only=True) # take the mean of all numeric columns
               .reset_index()           # turn ID back into a column
              )
    
    grouped = grouped.drop(columns="sample")
    
    # convert them all to int
    num_cols = grouped.select_dtypes(include="number").columns
    grouped[num_cols] = grouped[num_cols].round().astype(int)
    
    data3 = confusion_matrix(grouped['y_true'].to_numpy(), grouped['y_pred'].to_numpy(), labels=lbs)

    disp = ConfusionMatrixDisplay(confusion_matrix=data3)
    disp.plot()

    plt.savefig(self._model_path + '/CM_poly_avg.png')
    plt.close()

    return

def rocauc_plot(self, df):
    ''' Plots ROC-AUC curve for classification task
    
    Args:
    plottype : str, dataset to plot, 'val' for validation or 'test' for test
    fig_path : str, path to save figure
    '''

    probs = np.vstack(df['y_probs'].values)    # shape (n_samples, n_classes)
    truth = np.array(df['y_true'].tolist())    # shape (n_samples,)
    
    plt.figure()
    n_classes = probs.shape[1]
    
    for i in range(n_classes):
        # one-vs-rest truth vector
        bin_truth = (truth == i).astype(int)
        class_probs = probs[:, i]
    
        # skip if y_true never equals this class in your data
        if len(np.unique(bin_truth)) < 2:
            continue
    
        fpr, tpr, _ = roc_curve(bin_truth, class_probs)
        roc_auc    = auc(fpr, tpr)
        plt.plot(fpr, tpr,
                 lw=2,
                 label=f"OVR Class {i} (AUC = {roc_auc:.3f})")
    
    # plot the chance diagonal
    plt.plot([0, 1], [0, 1],
             linestyle="--",
             lw=2,
             color="#B2B2B2")
    
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate",  fontsize=14)
    plt.title("ROC Curves (One-vs-Rest)",    fontsize=16)
    plt.tick_params(labelsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(self._model_path + "/ROC_AUC.png")
    plt.close()

    return

def prauc_plot(self, df):

    # stack the per-sample probability lists into an (N × C) array
    y_probs = np.vstack(df['y_probs'].values)
    y_true  = np.array(df['y_true'].tolist())

    n_classes = y_probs.shape[1]
    plt.figure()

    for i in range(n_classes):
        # binarize: 1 for class i, else 0
        binary_targets = (y_true == i).astype(int)

        # skip if only one class present
        if len(np.unique(binary_targets)) < 2:
            continue

        probs_i = y_probs[:, i]
        precision, recall, _ = precision_recall_curve(binary_targets, probs_i)
        ap = average_precision_score(binary_targets, probs_i)

        plt.plot(
            recall, precision,
            lw=2,
            label=f'OVR Class {i} (AP={ap:.3f})'
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('One-vs-Rest Precision-Recall Curves', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(self._model_path + "/PR_Curve.png")
    plt.close()
    
    return
