from src.main.graph_generation import main

import numpy as numpy
import torch
from torch_geometric.data import Dataset
import numpy as np
import os
import argparse


class CustomGraphDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.file_list = [
            os.path.join(self.root, f)
            for f in os.listdir(self.root)
            if f.endswith(".pt") and "001" not in f
        ]

    @property
    def raw_file_names(self):
        # No raw files to process
        return []

    @property
    def processed_file_names(self):
        # This is actually not needed because we handle file paths in __init__
        return [f for f in os.listdir(self.processed_dir) if f.endswith(".pt")]

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        # Load only the requested graph
        return torch.load(self.file_list[idx])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The root directory to parse (str) e.g.:`../../data/processed/0_05__1000/`",
    )

    args = parser.parse_args()
    main(args.compactness, args.n_segments, args.file_id, args.downsample_factor)
    dataset = CustomGraphDataset(root=args.root_dir)
