import numpy as np
import logging
import sys

from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize

from scipy.special import entr

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax

from skorch.utils import to_numpy


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def KL_divergence(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                random_tie_break: bool = False, dropout_layer_indexes: list = [], 
                num_cycles : int = 50, **mc_dropout_kwargs) -> np.ndarray:
    """
    TODO: Work in progress 
    """
    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=True)

    predictions = get_predictions(classifier, X, num_cycles)

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=False)

    #KL_divergence = _KL_divergence(predictions)
    
    if not random_tie_break:
        return multi_argmax(KL_divergence, n_instances=n_instances)

    return shuffled_argmax(KL_divergence, n_instances=n_instances)


def mc_dropout_bald(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                random_tie_break: bool = False, dropout_layer_indexes: list = [], 
                num_cycles : int = 50, **mc_dropout_kwargs) -> np.ndarray:
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
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        The mc-dropout metric of the chosen instances; 
    """

    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=True)

    predictions = get_predictions(classifier, X, num_cycles)

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=False)

    #calculate BALD (Bayesian active learning divergence))
    bald_scores = _bald_divergence(predictions)

    if not random_tie_break:
        return multi_argmax(bald_scores, n_instances=n_instances)

    return shuffled_argmax(bald_scores, n_instances=n_instances)


def mc_dropout_mean_st(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                random_tie_break: bool = False, dropout_layer_indexes: list = [], 
                num_cycles : int = 50, **mc_dropout_kwargs) -> np.ndarray:
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
        **uncertainty_measure_kwargs: Keyword arguments to be passed for the uncertainty
            measure function.

    Returns:
        The indices of the instances from X chosen to be labelled;
        The mc-dropout metric of the chosen instances; 
    """

    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=True)

    predictions = get_predictions(classifier, X, num_cycles)

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=False)

    mean_standard_deviations = _mean_standard_deviation(predictions)

    if not random_tie_break:
        return multi_argmax(mean_standard_deviations, n_instances=n_instances)

    return shuffled_argmax(mean_standard_deviations, n_instances=n_instances)

def get_predictions(classifier: BaseEstimator, X: modALinput, num_predictions: int = 50):
    """
        Runs num_predictions times the prediction of the classifier on the input X 
        and puts the predictions in a list.

        Args:
            classifier: The classifier for which the labels are to be queried.
            X: The pool of samples to query from.
            num_predictions: Number of predictions which should be made
        Return: 
            prediction: list with all predictions
    """

    predictions = []
    for i in range(num_predictions):
        logging.getLogger().info("Dropout: start prediction forward pass")
        #call Skorch infer function to perform model forward pass
        #In comparison to: predict(), predict_proba() the infer() 
        # does not change train/eval mode of other layers 
        prediction = classifier.estimator.infer(X)
        predictions.append(to_numpy(prediction))
    return predictions


def entropy_sum(values, axis=-1):
    #sum Scipy basic entropy function: entr()
    return np.sum(entr(values), axis=axis)

def _mean_standard_deviation(proba: list) -> np.ndarray: 
    """
        Calculates the mean of the per class calculated standard deviations.

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
        Return: 
            Returns the mean standard deviation of the dropout cycles over all classes. 
    """

    proba_stacked = np.stack(proba, axis=len(proba[0].shape)) 
    mean_squared = np.mean(proba_stacked, axis=-1)**2
    squared_mean = np.mean(proba_stacked**2, axis=-1)
    standard_deviation_class_vise = np.sqrt(squared_mean - mean_squared)
    mean_standard_deviation = np.mean(standard_deviation_class_vise, axis=-1)

    return mean_standard_deviation


def _bald_divergence(proba: list) -> np.ndarray:
    """
        Calculates the bald divergence for each instance

        As it is explicitly formulated in: 
            Deep Bayesian Active Learning with Image Data. 
            (Yarin Gal, Riashat Islam, and Zoubin Ghahramani. 2017.)

        Args: 
            proba: list with the predictions over the dropout cycles
        Return: 
            Returns the mean standard deviation of the dropout cycles over all classes. 
    """
    proba_stacked = np.stack(proba, axis=len(proba[0].shape))

    #entropy along dropout cycles
    accumulated_entropy = entropy_sum(proba_stacked, axis=-1)
    f_x = accumulated_entropy/len(proba)

    #score sums along dropout cycles 
    accumulated_score = np.sum(proba_stacked, axis=-1)
    average_score = accumulated_score / len(proba) 
    #expand dimension w/o data for entropy calculation
    average_score = np.expand_dims(average_score, axis=-1)

    #entropy over average prediction score 
    g_x = entropy_sum(average_score, axis=-1)

    #entropy differences
    diff = np.subtract(g_x, f_x)

    #sum all dimensions of diff besides first dim (instances) 
    shaped = np.reshape(diff, (diff.shape[0], -1))
    bald = np.sum(shaped, axis=-1)

    return bald

def _KL_divergence(proba) -> np.ndarray:

    #create 3D or 4D array from prediction dim: (drop_cycles, proba.shape[0], proba.shape[1], opt:proba.shape[2])
    proba_stacked = np.stack(proba, axis=len(proba[0].shape))
    # TODO work in progress
    # TODO add dimensionality adaption
    #number_of_dimensions = proba_stacked.ndim
    #if proba_stacked.ndim > 2: 

    normalized_proba = normalize(proba_stacked, axis=0)


def set_dropout_mode(model, dropout_layer_indexes: list, train_mode: bool):
    """ 
        Function to enable the dropout layers by setting them to user specified mode (bool: train_mode)
        TODO: Reduce maybe complexity
        TODO: Keras support
    """

    modules = list(model.modules()) # list of all modules in the network.
    
    if len(dropout_layer_indexes) != 0:  
        for index in dropout_layer_indexes: 
            layer = modules[index]
            if layer.__class__.__name__.startswith('Dropout'): 
                if True == train_mode:
                    layer.train()
                elif False == train_mode:
                    layer.eval()
            else: 
                raise KeyError("The passed index: {} is not a Dropout layer".format(index))

    else: 
        for module in modules:
            if module.__class__.__name__.startswith('Dropout'):
                if True == train_mode:
                    module.train()
                    logging.getLogger().info("Dropout: set mode of " + str(module.__class__.__name__) + " to train")
                elif False == train_mode:
                    module.eval()
                    logging.getLogger().info("Dropout: set mode of " + str(module.__class__.__name__) + " to eval")
