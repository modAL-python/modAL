from collections.abc import Mapping
from typing import Callable

import numpy as np
import torch
from scipy.special import entr
from sklearn.base import BaseEstimator
from skorch.utils import to_numpy

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax


def default_logits_adaptor(input_tensor: torch.tensor, samples: modALinput):
    # default Callable parameter for get_predictions
    return input_tensor


def mc_dropout_bald(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                    random_tie_break: bool = False, dropout_layer_indexes: list = [],
                    num_cycles: int = 50, sample_per_forward_pass: int = 1000,
                    logits_adaptor: Callable[[
                        torch.tensor, modALinput], torch.tensor] = default_logits_adaptor,
                    **mc_dropout_kwargs,) -> np.ndarray:
    """
        Mc-Dropout bald query strategy. Returns the indexes of the instances with the largest BALD 
        (Bayesian Active Learning by Disagreement) score calculated through the dropout cycles
        and the corresponding bald score. 

        Based on the work of: 
            Deep Bayesian Active Learning with Image Data.
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)
            Dropout as a Bayesian Approximation: Representing Model Uncer- tainty in Deep Learning.
            (Yarin Gal and Zoubin Ghahramani. 2016.)
            Bayesian Active Learning for Classification and Preference Learning.
            (NeilHoulsby,FerencHusza ́r,ZoubinGhahramani,andMa ́te ́Lengyel. 2011.) 

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
            random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())
            num_cycles: Number of forward passes with activated dropout
            sample_per_forward_pass: max. sample number for each forward pass. 
                The allocated RAM does mainly depend on this.
                Small number --> small RAM allocation
            logits_adaptor: Callable which can be used to adapt the output of a forward pass 
                to the required vector format for the vectorised metric functions 
            **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.

        Returns:
            The indices of the instances from X chosen to be labelled;
            The mc-dropout metric of the chosen instances; 
    """
    predictions = get_predictions(
        classifier, X, dropout_layer_indexes, num_cycles, sample_per_forward_pass, logits_adaptor)
    # calculate BALD (Bayesian active learning divergence))

    bald_scores = _bald_divergence(predictions)

    if not random_tie_break:
        return multi_argmax(bald_scores, n_instances=n_instances)

    return shuffled_argmax(bald_scores, n_instances=n_instances)


def mc_dropout_mean_st(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                       random_tie_break: bool = False, dropout_layer_indexes: list = [],
                       num_cycles: int = 50, sample_per_forward_pass: int = 1000,
                       logits_adaptor: Callable[[
                           torch.tensor, modALinput], torch.tensor] = default_logits_adaptor,
                       **mc_dropout_kwargs) -> np.ndarray:
    """
        Mc-Dropout mean standard deviation query strategy. Returns the indexes of the instances 
        with the largest mean of the per class calculated standard deviations over multiple dropout cycles
        and the corresponding metric.

        Based on the equations of: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
            random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())
            num_cycles: Number of forward passes with activated dropout
            sample_per_forward_pass: max. sample number for each forward pass. 
                The allocated RAM does mainly depend on this.
                Small number --> small RAM allocation
            logits_adaptor: Callable which can be used to adapt the output of a forward pass 
                to the required vector format for the vectorised metric functions 
            **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.

        Returns:
            The indices of the instances from X chosen to be labelled;
            The mc-dropout metric of the chosen instances; 
    """

    # set dropout layers to train mode
    predictions = get_predictions(
        classifier, X, dropout_layer_indexes, num_cycles, sample_per_forward_pass, logits_adaptor)

    mean_standard_deviations = _mean_standard_deviation(predictions)

    if not random_tie_break:
        return multi_argmax(mean_standard_deviations, n_instances=n_instances)

    return shuffled_argmax(mean_standard_deviations, n_instances=n_instances)


def mc_dropout_max_entropy(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                           random_tie_break: bool = False, dropout_layer_indexes: list = [],
                           num_cycles: int = 50, sample_per_forward_pass: int = 1000,
                           logits_adaptor: Callable[[
                               torch.tensor, modALinput], torch.tensor] = default_logits_adaptor,
                           **mc_dropout_kwargs) -> np.ndarray:
    """
        Mc-Dropout maximum entropy query strategy. Returns the indexes of the instances 
        with the largest entropy of the per class calculated entropies over multiple dropout cycles
        and the corresponding metric.

        Based on the equations of: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
            random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())
            num_cycles: Number of forward passes with activated dropout
            sample_per_forward_pass: max. sample number for each forward pass. 
                The allocated RAM does mainly depend on this.
                Small number --> small RAM allocation
            logits_adaptor: Callable which can be used to adapt the output of a forward pass 
                to the required vector format for the vectorised metric functions 
            **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.

        Returns:
            The indices of the instances from X chosen to be labelled;
            The mc-dropout metric of the chosen instances; 
    """
    predictions = get_predictions(
        classifier, X, dropout_layer_indexes, num_cycles, sample_per_forward_pass, logits_adaptor)

    # get entropy values for predictions
    entropy = _entropy(predictions)

    if not random_tie_break:
        return multi_argmax(entropy, n_instances=n_instances)

    return shuffled_argmax(entropy, n_instances=n_instances)


def mc_dropout_max_variationRatios(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                                   random_tie_break: bool = False, dropout_layer_indexes: list = [],
                                   num_cycles: int = 50, sample_per_forward_pass: int = 1000,
                                   logits_adaptor: Callable[[
                                       torch.tensor, modALinput], torch.tensor] = default_logits_adaptor,
                                   **mc_dropout_kwargs) -> np.ndarray:
    """
        Mc-Dropout maximum variation ratios query strategy. Returns the indexes of the instances 
        with the largest variation ratios over multiple dropout cycles
        and the corresponding metric.

        Based on the equations of: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            n_instances: Number of samples to be queried.
            random_tie_break: If True, shuffles utility scores to randomize the order. This
                can be used to break the tie when the highest utility score is not unique.
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())
            num_cycles: Number of forward passes with activated dropout
            sample_per_forward_pass: max. sample number for each forward pass. 
                The allocated RAM does mainly depend on this.
                Small number --> small RAM allocation
            logits_adaptor: Callable which can be used to adapt the output of a forward pass 
                to the required vector format for the vectorised metric functions 
            **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
                measure function.

        Returns:
            The indices of the instances from X chosen to be labelled;
            The mc-dropout metric of the chosen instances; 
    """
    predictions = get_predictions(
        classifier, X, dropout_layer_indexes, num_cycles, sample_per_forward_pass, logits_adaptor)

    # get variation ratios values for predictions
    variationRatios = _variation_ratios(predictions)

    if not random_tie_break:
        return multi_argmax(variationRatios, n_instances=n_instances)

    return shuffled_argmax(variationRatios, n_instances=n_instances)


def get_predictions(classifier: BaseEstimator, X: modALinput, dropout_layer_indexes: list = [],
                    num_predictions: int = 50, sample_per_forward_pass: int = 1000,
                    logits_adaptor: Callable[[torch.tensor, modALinput], torch.tensor] = default_logits_adaptor):
    """
        Runs num_predictions times the prediction of the classifier on the input X 
        and puts the predictions in a list.

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())
            num_predictions: Number of predictions which should be made
            sample_per_forward_pass: max. sample number for each forward pass. 
                The allocated RAM does mainly depend on this.
                Small number --> small RAM allocation
            logits_adaptor: Callable which can be used to adapt the output of a forward pass 
                to the required vector format for the vectorised metric functions 
        Return: 
            prediction: list with all predictions
    """

    assert num_predictions > 0, 'num_predictions must be larger than zero'
    assert sample_per_forward_pass > 0, 'sample_per_forward_pass must be larger than zero'

    predictions = []
    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_,
                     dropout_layer_indexes, train_mode=True)

    split_args = []

    if isinstance(X, Mapping):  # check for dict
        for k, v in X.items():

            v.detach()
            split_v = torch.split(v, sample_per_forward_pass)
            # create sub-dictionary split for each forward pass with same keys&values
            for split_idx, split in enumerate(split_v):
                if len(split_args) <= split_idx:
                    split_args.append({})
                split_args[split_idx][k] = split

    elif torch.is_tensor(X):  # check for tensor
        X.detach()
        split_args = torch.split(X, sample_per_forward_pass)
    else:
        raise RuntimeError(
            "Error in model data type, only dict or tensors supported")

    for i in range(num_predictions):

        probas = []

        for samples in split_args:
            # call Skorch infer function to perform model forward pass
            # In comparison to: predict(), predict_proba() the infer()
            # does not change train/eval mode of other layers
            with torch.no_grad():
                logits = classifier.estimator.infer(samples)
                prediction = logits_adaptor(logits, samples)
                mask = ~prediction.isnan()
                prediction[mask] = prediction[mask].softmax(-1)
                probas.append(prediction)

        probas = torch.cat(probas)
        predictions.append(to_numpy(probas))

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_,
                     dropout_layer_indexes, train_mode=False)

    return predictions


def entropy_sum(values: np.array, axis: int = -1):
    # sum Scipy basic entropy function: entr()
    entropy = entr(values)
    return np.sum(entropy, where=~np.isnan(entropy), axis=axis)


def _mean_standard_deviation(proba: list) -> np.ndarray:
    """
        Calculates the mean of the per class calculated standard deviations.

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
            mask: mask to detect the padded classes (must be of same shape as elements in proba)
        Return: 
            Returns the mean standard deviation of the dropout cycles over all classes. 
    """

    proba_stacked = np.stack(proba, axis=len(proba[0].shape))

    standard_deviation_class_vise = np.std(proba_stacked, axis=-1)
    mean_standard_deviation = np.mean(standard_deviation_class_vise, where=~np.isnan(
        standard_deviation_class_vise), axis=-1)

    return mean_standard_deviation


def _entropy(proba: list) -> np.ndarray:
    """
        Calculates the entropy per class over dropout cycles

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
            mask: mask to detect the padded classes (must be of same shape as elements in proba)
        Return: 
            Returns the entropy of the dropout cycles over all classes. 
    """

    proba_stacked = np.stack(proba, axis=len(proba[0].shape))

    # calculate entropy per class and sum along dropout cycles
    entropy_classes = entropy_sum(proba_stacked, axis=-1)
    entropy = np.mean(entropy_classes, where=~
                      np.isnan(entropy_classes), axis=-1)
    return entropy


def _variation_ratios(proba: list) -> np.ndarray:
    """
        Calculates the variation ratios over dropout cycles

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
            mask: mask to detect the padded classes (must be of same shape as elements in proba)
        Return: 
            Returns the variation ratios of the dropout cycles. 
    """
    proba_stacked = np.stack(proba, axis=len(proba[0].shape))

    # Calculate the variation ratios over the mean of dropout cycles
    valuesDCMean = np.mean(proba_stacked, axis=-1)
    return 1 - np.amax(valuesDCMean, initial=0, where=~np.isnan(valuesDCMean), axis=-1)


def _bald_divergence(proba: list) -> np.ndarray:
    """
        Calculates the bald divergence for each instance

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
            mask: mask to detect the padded classes (must be of same shape as elements in proba)
        Return: 
            Returns the mean standard deviation of the dropout cycles over all classes. 
    """
    proba_stacked = np.stack(proba, axis=len(proba[0].shape))

    # entropy along dropout cycles
    accumulated_entropy = entropy_sum(proba_stacked, axis=-1)
    f_x = accumulated_entropy/len(proba)

    # score sums along dropout cycles
    accumulated_score = np.sum(proba_stacked, axis=-1)
    average_score = accumulated_score/len(proba)
    # expand dimension w/o data for entropy calculation
    average_score = np.expand_dims(average_score, axis=-1)

    # entropy over average prediction score
    g_x = entropy_sum(average_score, axis=-1)

    # entropy differences
    diff = np.subtract(g_x, f_x)

    # sum all dimensions of diff besides first dim (instances)
    shaped = np.reshape(diff, (diff.shape[0], -1))

    bald = np.sum(shaped, where=~np.isnan(shaped), axis=-1)
    return bald


def set_dropout_mode(model, dropout_layer_indexes: list, train_mode: bool):
    """ 
        Function to change the mode of the dropout layers (bool: train_mode -> train or evaluation)

        Args: 
            model: Pytorch model
            dropout_layer_indexes: Indexes of the dropout layers which should be activated
                Choose indices from : list(torch_model.modules())    
            train_mode: boolean, true <=> train_mode, false <=> evaluation_mode 
    """

    modules = list(model.modules())  # list of all modules in the network.

    if len(dropout_layer_indexes) != 0:
        for index in dropout_layer_indexes:
            layer = modules[index]
            if layer.__class__.__name__.startswith('Dropout'):
                if True == train_mode:
                    layer.train()
                elif False == train_mode:
                    layer.eval()
            else:
                raise KeyError(
                    "The passed index: {} is not a Dropout layer".format(index))

    else:
        for module in modules:
            if module.__class__.__name__.startswith('Dropout'):
                if True == train_mode:
                    module.train()
                elif False == train_mode:
                    module.eval()
