import typing
from typing import List, Tuple, Any, Union, Optional,Generic,TypeVar
import numpy as np
Label = Tuple[int, np.dtype]
LabelList = List[Label]
Sources = List[Any]


class DataManager:
    def __init__(
        self,
        features: np.ndarray,
        labels_dtype: Optional[np.dtype] = None,
        sources: Optional[Sources] = None,
    ):
        """

        When doing active learning we have our Original Data (OD) Labeled Data [LD] and Unlabeled Data [UD]
        where UD and LD are subsets of OD.
        The active learner operates on UD and returns indexes relative to it. We want to store those indices with respect
        to OD, and sometimes see the subset of labels of LD. (The subset of labels of UD is Null)

        That's a fancy way of saying there is a lot book keeping to be done and this class solves that by doing it for you

        The main idea is that we store a mask (labeeld_mask) of indices that have been labeled and then expose UD , LD
        and the labels by using fancy indexing with that mask. The manager exposes a an add_labels method which lets the
        user add labels indexed with respect to UD and it will adjust the indices so that they match OD.

        :param features: An array of the features that will be used for AL.
        :param labels: Any prexesiting labels. Each label is a tuple(idx,label)
        :param source: A list of the original data
        """
        self.features = features

        self._labels =np.empty(shape=self.features.shape[0],dtype=labels_dtype)
        self.labeled_mask = np.zeros(self.features.shape[0], dtype=bool)
        self.sources = np.array(sources if sources else [])

    @property
    def labels(self):
        '''

        Returns the labels indexed with respect to LD

        '''
        return self._labels[self.labeled_mask]

    @property
    def unlabeled_mask(self):
        '''

        Returns: a mask which is true for all unlabeled points

        '''
        return np.logical_not(self.labeled_mask)

    def _update_masks(self, labels: Union[LabelList, Label]):
        for label in labels:
            self.labeled_mask[label[0]] = True

    def _offset_new_labes(self, labels: LabelList):
        """
        This is where the magic happens.
        We take self.unlabeled_mask.nonzero()[0] which gives us an array of the indices that appear in the unlabeled
        data. So if the original label was at position 0 we look up the "real index" in the unlabeled_indices array to
        get it's true index
        :param labels:
        :return:
        """
        if len(self._labels) == 0:
            # Nothing to correct in this case
            return labels
        correctLabels: LabelList = []
        unlabeled_indices = self.unlabeled_mask.nonzero()[0]

        for label in labels:
            newIndex = unlabeled_indices[label[0]]
            newLabel: Label = (newIndex, label[1])
            correctLabels.append(newLabel)
        return correctLabels

    def add_labels(self, labels: LabelList,offset_to_unlabeled=True):
        if isinstance(labels, tuple):  # if this is a single example
            labels: LabelList = [labels]
        elif isinstance(labels, list):
            pass
        else:
            raise Exception(
                "Malformed input. Please add either a tuple (ix,label) or a list [(ix,label),..]"
            )
        if offset_to_unlabeled:
            labels = self._offset_new_labes(labels)
        self._update_masks(labels)
        for label in labels:
            self._labels[label[0]] = label[1]

    @property
    def unlabeld(self):
        """

        :return: Returns UD, all of the unlabeled data points
        """
        return self.features[self.unlabeled_mask]

    @property
    def labeled(self):
        """
                :return: Returns LD, all of the labeld data points
        """
        return self.features[self.labeled_mask]

    @property
    def remaining_sources(self):
        """

        :return: Returns the original data, as opposed to features, with respect to UD
        """
        return self.sources[self.unlabeled_mask]

    def get_original_index_from_unlabeled_index(self, ixs:Union[int, List[int]]):
        '''
        Utility function that takes as input indices from the unlabeled subset and returns the equivalent indices
        in the complete array.
        Useful for testing purposes, where we have the existing labels and want to take them in the order in which
        the active learner specifes.
        :param ixs:
        :return:
        '''
        unlabeled_indices = self.unlabeled_mask.nonzero()[0]
        if isinstance(ixs, np.int64):
            ixs = [ixs]
        return list(map(lambda x: unlabeled_indices[x], ixs))



__all__ = [Label, LabelList, DataManager]
