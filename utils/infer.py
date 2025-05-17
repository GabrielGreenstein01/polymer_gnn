import os
import torch
import dgl
import json
import re
import joblib

from torch.utils.data import DataLoader

from utils.util_functions import seq_to_dgl

class infer:
    def __init__(self, DF, GPU, NUM_WORKERS, MODEL, MODEL_PATH, SMILES, DESCRIPTORS):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        self._exp_config = json.load(open(MODEL_PATH + 'configure.json'))
        self._num_workers = NUM_WORKERS
        
        self._model = MODEL
        self._model_path = MODEL_PATH

        self._smiles = SMILES
        self._descriptors = DESCRIPTORS
        self._features = joblib.load(MODEL_PATH + "features.pkl")

        self._df = DF

        self._df['dgl'] = self._df.apply(lambda row: 
                                           seq_to_dgl(row['ID'], row['sequence'], self._features, self._model),
                                           axis=1)

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        if self._device.type == "cpu":
            model = torch.load(
                "{}/fullmodel.pt".format(self._model_path),
                map_location=torch.device("cpu"),
            )
        elif self._device.type == "cuda":
            model = torch.load(
                "{}/fullmodel.pt".format(self._model_path),
                map_location=torch.device("cuda"),
            )

        self.model = model.to(self._device)

    def _run_an_eval_epoch(self, model, data_loader):
        """Utility function for running an evaluation (validation/test) epoch

        Args:
        model : dgllife model, Predictor with set hyperparameters
        data_loader : DataLoader, DataLoader for train, validation, or test

        Returns:
        metric_dict : dict, dictionary of metric names and corresponding evaluation values
        """
        all_preds = []
        all_IDs = []

        model.eval()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(data_loader):
                IDs, bg = batch_data
                logits = self._predict(model, bg)

                all_IDs.extend(IDs)
                all_preds.append(logits.detach().cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_preds = torch.sigmoid(all_preds).numpy().ravel()

        return [*zip(all_IDs, all_preds)]

    def _collate_molgraphs(self, data: list[tuple[str, dgl.DGLGraph]]):
        """
        Collate function for a list of tuples (ID, graph).
        """
        # seperate IDs and graphs
        IDs, graphs = map(list, zip(*data))
        bg = dgl.batch(graphs)
    
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
    
        return IDs, bg

    def _predict(self, model, bg):
        bg = bg.to(self._device)
        if self._exp_config["model"] in ["GCN", "GAT"]:
            node_feats = bg.ndata.pop("h").to(self._device)
            preds = model(bg, node_feats)
            node_feats.detach().cpu()
            del node_feats
        else:
            node_feats = bg.ndata.pop("h").to(self._device)
            edge_feats = bg.edata.pop("e").to(self._device)
            preds = model(bg, node_feats, edge_feats)
            node_feats.detach().cpu()
            edge_feats.detach().cpu()
            del edge_feats
            del node_feats

        bg.to("cpu")
        del bg

        return preds

    def run(self):

        # data = list(zip(self._df["ID"], self._df["sequence"].apply(lambda x: seq_to_dgl(x))))
        data = list(zip(self._df["ID"], self._df["dgl"]))

        data_loader = DataLoader(
            dataset=data, 
            batch_size= self._exp_config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_molgraphs, 
            num_workers=self._num_workers)
        
        preds = self._run_an_eval_epoch(self.model, data_loader)

        return preds
