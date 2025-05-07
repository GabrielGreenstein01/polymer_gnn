import dgl
import torch

from torch.utils.data import DataLoader

def collate_molgraphs(data):

    IDs, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return IDs, bg

def load_data(DATA, BATCH_SIZE, NUM_WORKERS):

    return DataLoader(dataset=DATA, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=collate_molgraphs, num_workers=NUM_WORKERS)