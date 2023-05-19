from typing import Tuple, List, Dict

from PIL import Image
from importlib_resources import files

import numpy as np
from importlib_resources.abc import Traversable
from numpy.random import default_rng
from torchvision.datasets import ImageNet


class DiscoverTargetTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, y):
        y = self.mapping[y]
        return y


class ImageNetSplit(ImageNet):
    """
    ImageNet-based dataset for Novel Class Discovery (NCD) and
    Generalized Category Discovery (GCD). Introduced in the paper
    "Supervised Knowledge May Hurt Novel Class Discovery Performance".
    """
    LABELED_SPLITS = ["l1", "l1.5", "l2"]
    UNLABELED_SPLITS = ["u1", "u2"]
    NUM_LABELED_CLASSES = 90
    NUM_UNLABELED_CLASSES = 30

    def __init__(
        self,
        root: str,
        split: str = "train",
        setting: str = "NCD",
        labeled_set: str = "L1",
        unlabeled_set: str = "U1",
        gcd_labeled_proportion: float = 0.5,
        **kwargs,
    ):
        """

        :param root: Root directory of the ImageNet dataset.
        :param split: The split to use. Can be either "train" or "val".
        :param setting: The setting to use. Can be either "NCD" or "GCD".
        :param labeled_set: The labeled set to use. Can be either "L1", "L1.5" or "L2".
        :param unlabeled_set: The unlabeled set to use. Can be either "U1" or "U2".
        :param gcd_labeled_proportion: The proportion of labeled samples in the known classes (GCD setting only).
        :param kwargs: Additional arguments to pass to the ImageNet constructor.
        """
        labeled_lower = labeled_set.lower()
        unlabeled_lower = unlabeled_set.lower()

        assert setting in ["NCD", "GCD"], f"Setting can only be NCD or GCD!"
        assert (
            labeled_lower in self.LABELED_SPLITS
        ), f"Labeled split {labeled_set} is not supported!"
        assert (
            unlabeled_lower in self.UNLABELED_SPLITS
        ), f"Unlabeled split {unlabeled_set} is not supported!"

        self.setting = setting
        self.labeled_set = labeled_lower
        self.unlabeled_set = unlabeled_lower
        self.gcd_labeled_proportion = gcd_labeled_proportion
        super(ImageNetSplit, self).__init__(root, split, **kwargs)

        # For the GCD setting, we need to decide which samples from the known classes
        # are labeled and which are unlabeled.
        # We need to do this after the super constructor is called, because we need
        # the self.targets attribute.
        self.is_labeled = self._split_labeled_unlabeled()

    def _split_labeled_unlabeled(self) -> np.ndarray:
        transformed_targets = [self.target_transform(y) for y in self.targets]
        transformed_targets = np.array(transformed_targets)

        # For NCD, it is easy: all samples from the known classes are labeled.
        if self.setting == "NCD":
            return transformed_targets < self.NUM_LABELED_CLASSES

        # For GCD, we need to decide which samples from the known classes are labeled
        # Set the seed to make the split reproducible
        rng = default_rng(seed=0)
        is_labeled = np.zeros(len(self.targets), dtype=bool)

        for i in range(self.NUM_LABELED_CLASSES):
            indices = np.where(transformed_targets == i)[0]
            num_labeled = int(len(indices) * self.gcd_labeled_proportion)

            # Shuffle the indices
            rng.shuffle(indices)
            is_labeled[indices[:num_labeled]] = True

        return is_labeled

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

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, bool]:
        """
        Fetch one sample from the dataset.

        :param index: Index of the sample
        :return: (image, target, is_labeled) where target is class_index of the target class.
            is_labeled is a boolean indicating whether the sample is labeled or not.
        """
        img, target = super(ImageNetSplit, self).__getitem__(index)
        is_labeled = self.is_labeled[index]

        return img, target, is_labeled
