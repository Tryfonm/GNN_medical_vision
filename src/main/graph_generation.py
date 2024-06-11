import numpy as numpy
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.preprocessing import MinMaxScaler
import argparse
import numpy as np
import csv
from skimage.segmentation import slic
from skimage.transform import rescale
import numpy as np
from tqdm import tqdm
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


from collections import Counter


class Nii_loader:
    def __init__(
        self,
        file_id="001",
        root_dir="./data/raw/MICCAI_BraTS2020_TrainingData/",
    ) -> None:
        self.root_dir = root_dir

        self.file_id = file_id

        t1_path = (
            self.root_dir
            + f"BraTS20_Training_{self.file_id}/BraTS20_Training_{self.file_id}_t1.nii"
        )
        t1ce_path = (
            self.root_dir
            + f"BraTS20_Training_{self.file_id}/BraTS20_Training_{self.file_id}_t1ce.nii"
        )
        t2_path = (
            self.root_dir
            + f"BraTS20_Training_{self.file_id}/BraTS20_Training_{self.file_id}_t2.nii"
        )
        flair_path = (
            self.root_dir
            + f"BraTS20_Training_{self.file_id}/BraTS20_Training_{self.file_id}_flair.nii"
        )
        mask_path = (
            self.root_dir
            + f"BraTS20_Training_{self.file_id}/BraTS20_Training_{self.file_id}_seg.nii"
        )
        self.paths_dict = {
            "t1": t1_path,
            "t1ce": t1ce_path,
            "t2": t2_path,
            "flair": flair_path,
            "mask": mask_path,
        }

        for key, value in self.paths_dict.items():
            setattr(self, key, nib.load(value).get_fdata())
        # self.t1 = [x for x in nib.load(list(self.paths_dict.values())).get_fdata()]
        self.mask[self.mask == 4] = 3

    def change_to_binary_mask(self):
        self.mask = np.where(self.mask == 0, 0, 1)

    @staticmethod
    def _normalize(img_raw):
        img_raw_shape = img_raw.shape

        img = img_raw.reshape(-1, 1)
        scaler = MinMaxScaler()
        img = scaler.fit_transform(img)
        img = img.reshape(img_raw_shape)

        return img

    def normalize_all(self):
        self.t1 = self._normalize(self.t1)
        self.t1ce = self._normalize(self.t1ce)
        self.t2 = self._normalize(self.t2)
        self.flair = self._normalize(self.flair)

    @staticmethod
    def get_slic_mask(img):
        return np.where(img != 0.0, 1, 0)

    def get_stacked_modalities(self):
        img_stacked = np.stack((self.t1, self.t1ce, self.t2, self.flair))

        if not (img_stacked[0] == self.t1).all():
            raise ValueError
        return img_stacked

    def get_slic_labels(self, n_segments=1000, compactness=0.05):
        # TODO: check all 4 convex shapes and get the intersection might be most robust
        slic_mask = Nii_loader.get_slic_mask(self.flair)

        slic_labels = slic(
            self.get_stacked_modalities(),
            n_segments=n_segments,
            compactness=compactness,
            channel_axis=0,
            mask=slic_mask,
        )
        # unique_labels, counts = np.unique(slic_sample_image, return_counts=True)
        # label_counts = dict(zip(unique_labels, counts))

        return slic_labels

    # TODO: delete -> meant for tersting multithreading
    def _downsample(self, img, downsample_factor, is_label=False):
        downsample_factor = (downsample_factor, downsample_factor, downsample_factor)

        # downsampled_data = rescale(
        #     img, downsample_factor, anti_aliasing=True, mode="reflect"
        # )
        if is_label:
            # print(f"before downsampling: {img.shape}")
            # Use nearest-neighbor interpolation for label data
            downsampled_img = rescale(
                img,
                downsample_factor,
                anti_aliasing=False,
                mode="reflect",
                order=0,
                preserve_range=True,
            )
            # print(f"after downsampling: {downsampled_img.shape}")
            # downsampled_data = downsampled_data.astype(img.dtype)
        else:
            # print(f"before downsampling: {img.shape}")
            # Use default anti-aliasing for image data
            downsampled_img = rescale(
                img,
                downsample_factor,
                anti_aliasing=False,  # Typically not needed for label data
                order=0,  # Nearest-neighbor interpolation
                preserve_range=True,  # Preserve the original data's range
                mode="edge",
            )
            # print(f"after downsampling: {downsampled_img.shape}")
            # print("--------------------------")
        ## If needed, save the downsampled data back to a memmap
        # downsampled_shape = downsampled_data.shape
        # downsampled_memmap = np.memmap(
        #     "downsampled_memmap.dat",
        #     dtype="float32",
        #     mode="w+",
        #     shape=downsampled_shape,
        # )
        return downsampled_img
        # downsampled_memmap.flush()

    # TODO: delete -> meant for tersting multithreading
    def downsample_all(self, downsample_factor):
        # print(f"Downsampling with factor: {downsample_factor}")
        self.t1 = self._downsample(self.t1, downsample_factor)
        self.t1ce = self._downsample(self.t1ce, downsample_factor)
        self.t2 = self._downsample(self.t2, downsample_factor)
        self.flair = self._downsample(self.flair, downsample_factor)
        self.mask = self._downsample(self.mask, downsample_factor, is_label=True)


class Voxel:
    def __init__(self, index, stacked_img, slic_labels) -> None:
        self.x = index[0]
        self.y = index[1]
        self.z = index[2]
        self.index = index
        self.stacked_img = stacked_img
        self.slic_labels = slic_labels

        self.values = self.stacked_img[(slice(None), *self.index)]
        self.label = slic_labels[tuple(index)]

    def get_neighors(self):

        # TODO: change to numpy slicing instead of iterating
        neighbors = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    neighbor = Voxel(
                        [self.x + i, self.y + j, self.z + k],
                        self.stacked_img,
                        self.slic_labels,
                    )
                    if neighbor.label != 0:
                        neighbors.append(neighbor)

        return neighbors

    def get_neighbor_labels(self, return_dict=False):
        neighbor_labels = []
        for n in self.get_neighors():
            neighbor_labels.append(n.label)

        if return_dict:
            return dict(Counter(neighbor_labels))
        else:
            return neighbor_labels

    def __repr__(self) -> str:
        return f"Voxel at {self.x}, {self.y}, {self.z} with values {[x for x in self.values]} and label {self.label}"


class SuperVoxel:
    def __init__(self, label, stacked_img, seg_mask, slic_labels) -> None:
        self.label = label  # slic_label that is assigned to this supervoxel
        self.stacked_img = stacked_img
        self.seg_mask = seg_mask
        self.slic_labels = slic_labels
        self.list_of_voxel_indices = list(zip(*np.where(slic_labels == label)))
        self.list_of_voxels = [
            Voxel(index, self.stacked_img, self.slic_labels)
            for index in self.list_of_voxel_indices
        ]

        self.supervoxel_mask = np.zeros(self.seg_mask.shape, dtype=np.float32)
        for idx in self.list_of_voxel_indices:
            self.supervoxel_mask[idx] = 1

    def get_all_neighbors(self, return_dict=True):
        neighbors = []
        for voxel in self.list_of_voxels:
            neighbors = neighbors + voxel.get_neighbor_labels(return_dict=False)

        if return_dict:
            return dict(Counter(neighbors))
        else:
            return neighbors

    def get_mean_value(self):
        # TODO: make this work with stacked image instead of image
        indices_tensor = torch.tensor(self.list_of_voxel_indices)
        t1_mean = self.stacked_img[
            0, indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]
        ].mean()
        t1ce_mean = self.stacked_img[
            1, indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]
        ].mean()
        t2_mean = self.stacked_img[
            2, indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]
        ].mean()
        flair_mean = self.stacked_img[
            3, indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]
        ].mean()

        return [t1_mean, t1ce_mean, t2_mean, flair_mean]

    def get_seg_label(self):
        indices_tensor = torch.tensor(self.list_of_voxel_indices)
        mode = torch.mode(
            torch.tensor(
                self.seg_mask[
                    indices_tensor[:, 0], indices_tensor[:, 1], indices_tensor[:, 2]
                ]
            )
        ).values.item()
        return mode

    def get_edges_for_networkx(self):
        edges = []
        all_neighbors_dict = self.get_all_neighbors()
        for target, weight in all_neighbors_dict.items():
            edges.append((self.label, target, weight))
        return edges

    def prepare_for_dgl(self):
        edges_nx = torch.tensor(self.get_edges_for_networkx())
        src_tensor, dst_tensor, weights = edges_nx[:, 0], edges_nx[:, 1], edges_nx[:, 2]
        # check tensors
        return src_tensor, dst_tensor, weights

    def prepare_for_pyg(self):
        dgl_tensor = self.prepare_for_dgl()
        edge_index = torch.concat(
            (dgl_tensor[0].reshape(-1, 1), dgl_tensor[1].reshape(-1, 1)), dim=1
        )
        return (
            torch.tensor(self.get_mean_value()),
            edge_index,
            torch.tensor(self.get_seg_label()),
            dgl_tensor[2].reshape(-1, 1),
        )

    def __len__(self):
        return len(self.list_of_voxel_indices)

    def __repr__(self) -> str:
        return f"SuperVoxel {self.label} with {len(self)} Voxels"


from datetime import datetime


def main(
    file_id: str,
    compactness: float,
    n_segments: int,
    save_as: str = "graph_test.pt",
    downsample_factor: float = None,
    change_to_binary: bool = False,
    return_results: bool = False,
) -> None:

    nii_loader = Nii_loader(file_id=file_id)
    nii_loader.normalize_all()
    if change_to_binary:
        nii_loader.change_to_binary_mask()
    if downsample_factor:
        nii_loader.downsample_all(downsample_factor=downsample_factor)

    img_stacked = nii_loader.get_stacked_modalities()
    slic_labels = nii_loader.get_slic_labels(
        compactness=compactness, n_segments=n_segments
    )

    list_of_labels_unique = list(np.unique(slic_labels))[1:]  # removing bakcground

    if 0 in list_of_labels_unique[1:]:  # check [1:] -> should be [:]
        raise ValueError

    list_of_supervoxels = []
    list_of_features = []
    list_of_edge_indices = []
    list_of_labels = []
    list_of_weights = []

    current_time = datetime.now().strftime("%H:%M:%S")
    for slic_label in tqdm(
        list_of_labels_unique,
        desc=f"INFO | {current_time} | Processing file_id {file_id}",
    ):  # TODO remove slice
        sv = SuperVoxel(
            label=slic_label,
            stacked_img=img_stacked,
            seg_mask=nii_loader.mask,
            slic_labels=slic_labels,
        )
        feature_temp, edge_index_temp, label_temp, weights_temp = sv.prepare_for_pyg()

        # if (src == 0).any().item():
        #     raise ValueError(f"slic_label (src): {slic_label} contains 0\n\t{src}")
        # if (dst == 0).any().item():
        #     raise ValueError(f"slic_label (dst): {slic_label} contains 0\n\t{dst}")

        list_of_supervoxels.append(sv)
        list_of_features.append(feature_temp.reshape(-1, 4))
        list_of_edge_indices.append(edge_index_temp)
        list_of_labels.append(label_temp.reshape(-1, 1))
        list_of_weights.append(weights_temp)

        # if slic_label == 9:
        #     break

        # if 0 in g.nodes():  # dgl starts indexing from 0 and thus a 0-node is introduced
        #     raise ValueError(f"failed at sv: {sv.label}")

    # g.remove_nodes(0)
    ## IMPORTANT: the supervoxels label need to be subtracted by 1 to match to dgl node IDs

    # ------------------------------------------------------------------------- #
    list_of_supervoxel_idx_with_overlap = []
    list_of_overlap_proportions = []
    list_of_dice_coefficients = []

    for sv_idx, sv in enumerate(list_of_supervoxels):
        overlap = np.logical_and(sv.seg_mask == 1, sv.supervoxel_mask == 1)

        # Count the number of overlapping elements
        overlap_count = np.sum(overlap)

        # Calculate the total number of elements in each mask
        total_mask1_elements = np.sum(sv.seg_mask == 1)
        total_mask2_elements = np.sum(sv.supervoxel_mask == 1)

        # Calculate the proportion of the second mask within the first one
        overlap_proportion = (
            overlap_count / total_mask2_elements if total_mask2_elements != 0 else 0
        )

        # Calculate Dice coefficient
        dice_coefficient = (
            (2 * overlap_count) / (total_mask1_elements + total_mask2_elements)
            if (total_mask1_elements + total_mask2_elements) != 0
            else 0
        )

        if overlap_proportion > 0.1:
            list_of_supervoxel_idx_with_overlap.append(sv_idx)
            list_of_overlap_proportions.append(overlap_proportion)
            list_of_dice_coefficients.append(dice_coefficient)

    overlap_proportion_mean = round(
        sum(list_of_overlap_proportions) / len(list_of_overlap_proportions), 6
    )
    dice_coefficient_mean = round(
        sum(list_of_dice_coefficients) / len(list_of_dice_coefficients), 6
    )

    # Save means to CSV file
    with open(save_as.replace("pt", "csv"), mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["overlap_proportion", "dice_coefficient"])
        writer.writerow([overlap_proportion_mean, dice_coefficient_mean])

    # ------------------------------------------------------------------------- #
    # Required to project findings back to the images
    # 1. method: if enough space on disk
    # with open(save_as.replace("pt", "pkl"), "wb") as f:
    #     pickle.dump(list_of_supervoxels, f)
    # print(f"list_of_supervoxels saved to {save_as}")
    # 2. method: just saves the absolute necessery

    import json

    sv_index_to_indices = {}

    for sv_index, sv in enumerate(list_of_supervoxels):
        indices = tuple(zip(*sv.list_of_voxel_indices))
        sv_index_to_indices[sv_index] = indices

    for key, value in sv_index_to_indices.items():
        sv_index_to_indices[key] = tuple(
            tuple(int(item) for item in inner_tuple) for inner_tuple in value
        )

    with open(save_as.replace("pt", "json"), "w") as json_file:
        json.dump(sv_index_to_indices, json_file)

    graph = Data(
        x=torch.cat(list_of_features, dim=0),
        edge_index=torch.cat(list_of_edge_indices, dim=0).to(dtype=torch.long).T
        - 1,  # long story for the -1
        edge_attr=torch.cat(list_of_weights, dim=0),
        y=torch.cat(list_of_labels, dim=0).reshape(-1),
    )

    torch.save(graph, save_as)

    if return_results:
        return (
            nii_loader,
            img_stacked,
            slic_labels,
            list_of_labels_unique,
            list_of_supervoxels,
            list_of_features,
            list_of_edge_indices,
            list_of_labels,
            list_of_weights,
            graph,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main function with compactness and number of segments."
    )

    # Adding arguments
    parser.add_argument(
        "--file_id",
        type=str,
        required=True,
        help="The file_id parameter (str).",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        required=True,
        help="The compactness parameter (float).",
    )
    parser.add_argument(
        "--n_segments",
        type=int,
        required=True,
        help="The number of segments (integer).",
    )
    parser.add_argument(
        "--save_as",
        type=str,
        required=True,
        help="The number of segments (integer).",
    )

    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function with parsed arguments
    main(args.file_id, args.compactness, args.n_segments, args.save_as)
