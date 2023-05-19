from typing import Tuple, List, Dict
from importlib_resources import files

import numpy as np
from importlib_resources.abc import Traversable
from torchvision.datasets import ImageNet


class DiscoverTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class ImageNetSplit(ImageNet):
    LABELED_SPLITS = ["l1", "l1.5", "l2"]
    UNLABELED_SPLITS = ["u1", "u2"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        labeled_set: str = "L1",
        unlabeled_set: str = "U1",
        **kwargs,
    ):
        labeled_lower = labeled_set.lower()
        unlabeled_lower = unlabeled_set.lower()

        assert (
            labeled_lower in self.LABELED_SPLITS
        ), f"Labeled split {labeled_set} is not supported!"
        assert (
            unlabeled_lower in self.UNLABELED_SPLITS
        ), f"Unlabeled split {unlabeled_set} is not supported!"

        self.labeled_set = labeled_lower
        self.unlabeled_set = unlabeled_lower
        super(ImageNetSplit, self).__init__(root, split, **kwargs)

    @staticmethod
    def _get_class_list_path(is_labeled: bool, split: str) -> Traversable:
        labeled = "labeled" if is_labeled else "unlabeled"
        file_name = f"imagenet_{labeled}_{split}.txt"
        return files("similaritybench.splits").joinpath(file_name)

    def _initialize_entity30(self, class_to_idx):
        labeled_path = self._get_class_list_path(True, self.labeled_set)
        unlabeled_path = self._get_class_list_path(False, self.unlabeled_set)

        labeled_classes = list(np.loadtxt(labeled_path, dtype=str))
        unlabeled_classes = list(np.loadtxt(unlabeled_path, dtype=str))

        labeled_class_idxs = np.array([class_to_idx[name] for name in labeled_classes])
        unlabeled_class_idxs = np.array(
            [class_to_idx[name] for name in unlabeled_classes]
        )

        # target transform
        all_class_idxs = np.concatenate((labeled_class_idxs, unlabeled_class_idxs))

        target_transform = DiscoverTargetTransform(
            {original: target for target, original in enumerate(all_class_idxs)}
        )
        self.target_transform = target_transform

        new_classes = labeled_classes + unlabeled_classes
        new_class_to_idx = {cls: idx for cls, idx in zip(new_classes, all_class_idxs)}

        return new_classes, new_class_to_idx

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        classes, class_to_idx = super(ImageNetSplit, self)._find_classes(dir)

        subset_classes, subset_class_to_idx = self._initialize_entity30(class_to_idx)

        return subset_classes, subset_class_to_idx
