import pandas as pd
import re
import joblib
import dgl
import torch
import pickle


class Preprocess:
    def __init__(
        self,
        df_polymers: pd.DataFrame,
        SCALERS,
        MODEL,
    ):
        """
        Will generate DGL graphs from a DataFrame of polymers.

        :param df_polymers: DataFrame with columns "ID", "sequence"
        :param MON_SMILES_POLY: Path to monomer SMILES file
        :param BOND_SMILES_POLY: Path to bond SMILES file
        :param DESCRIPTORS: Path to descriptors file
        :param SCALERS: Path to scalers file
        """
        if "ID" not in df_polymers.columns or "sequence" not in df_polymers.columns:
            raise ValueError(
                "df_polymers must have columns 'ID' and 'sequence'")

        self.model = MODEL
        self.poly_IDs = df_polymers["ID"].map(
            lambda x: "_".join(map(str, map(int, x.split("_")[:2])))
        )
        self.sequences = df_polymers["sequence"]

        with open("monomer_data/rdkit_features_synthetic_monomers.pkl", "rb") as f:
            self.monomer_features = pickle.load(
                f
            )
        self.scalers = joblib.load(SCALERS)
        self.scaled_feats = self.scale_features()

    def scale_features(self):
        """
        Get scaled_feats by scaling monomer features with scalers fit a
        training time.
        """
        # Get scaled_feats by scaling unscaled_feats with SCALERS
        res = {}
        for graph_component in self.monomer_features.keys():
            unscaled_df = (
                pd.DataFrame(self.monomer_features[graph_component])
                .T.reset_index()
                .rename(columns={"index": graph_component})
            )

            is_single_row = unscaled_df.shape[0] == 1

            data = unscaled_df.iloc[:, 1:].to_numpy()
            data = data.reshape(-1, 1) if is_single_row else data

            scaled_data = self.scalers[graph_component].transform(data)
            scaled_data = scaled_data.T if is_single_row else scaled_data

            unscaled_df.iloc[:, 1:] = scaled_data
            res[graph_component] = (
                unscaled_df.set_index(graph_component)
                .apply(lambda row: row.to_numpy(), axis=1)
                .to_dict()
            )

        return res

    def seq_to_dgl(self, sequence):
        monomers = re.findall(r"[A-Z][a-z]+", sequence)

        # Initialize DGL graph
        g = dgl.graph(([], []), num_nodes=len(monomers))

        # Featurize nodes
        node_features = [
            torch.tensor(self.scaled_feats["node"]
                         [monomer], dtype=torch.float32)
            for monomer in monomers
        ]
        g.ndata["h"] = torch.stack(node_features)

        # Edges are between sequential monomers, i.e., (0->1, 1->2, etc.)
        src_nodes = list(range(len(monomers) - 1))  # Start nodes of edges
        dst_nodes = list(range(1, len(monomers)))  # End nodes of edges
        g.add_edges(src_nodes, dst_nodes)

        # Featurize edges
        edge_features = [
            torch.tensor(self.scaled_feats["edge"]["CC"], dtype=torch.float32)
        ] * g.number_of_edges()
        g.edata["e"] = torch.stack(edge_features)

        if self.model == "GCN" or self.model == "GAT":
            g = dgl.add_self_loop(g)

        return g

    def get_dgl_graphs(self) -> list[tuple[str, dgl.DGLGraph]]:
        self.graphs = self.sequences.apply(lambda x: self.seq_to_dgl(x))

        return list(zip(self.poly_IDs, self.graphs))
