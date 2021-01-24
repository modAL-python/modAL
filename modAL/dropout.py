import numpy as np
import logging
import sys

from sklearn.base import BaseEstimator
from scipy.special import entr

from modAL.utils.data import modALinput
from modAL.utils.selection import multi_argmax, shuffled_argmax

from skorch.utils import to_numpy


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def mc_dropout(classifier: BaseEstimator, X: modALinput, n_instances: int = 1,
                random_tie_break: bool = False, dropout_layer_indexes: list = [], 
                num_cycles : int = 50, **mc_dropout_kwargs) -> np.ndarray:
    """
    Mc-Dropout query strategy. Selects the instance with the largest change in their 
    values by multiple forward passes with enabled dropout. Change/ Disagrement is 
    the calculated BALD (Bayesian Active Learning by Disagreement) score. 

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
    """

    # set dropout layers to train mode
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=True)

    predictions = []

    #for each batch run num_cycles forward passes
    for i in range(num_cycles):
        logging.getLogger().info("Dropout: start prediction forward pass")
        #call Skorch infer function to perform model forward pass
        #In comparison to: predict(), predict_proba() the infer() 
        # does not change train/eval mode of other layers 
        prediction = classifier.estimator.infer(X)
        predictions.append(to_numpy(prediction))

    # set dropout layers to eval
    set_dropout_mode(classifier.estimator.module_, dropout_layer_indexes, train_mode=False)

    #calculate BALD (Bayesian active learning divergence))
    bald_scores = _bald_divergence(predictions)

    if not random_tie_break:
        return multi_argmax(bald_scores, n_instances=n_instances)

    return shuffled_argmax(bald_scores, n_instances=n_instances)

def entropy_sum(values, axis=-1):
    #sum Scipy basic entropy function: entr()
    return np.sum(entr(values), axis=axis)

def _bald_divergence(proba) -> np.ndarray:
    accumulated_score = np.zeros(shape=proba[0].shape)
    accumulated_entropy = np.zeros(shape=(proba[0].shape[0]))

    #create 3D or 4D array from prediction dim: (drop_cycles, proba.shape[0], proba.shape[1], opt:proba.shape[2])
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
    diff = g_x - f_x

    #sum all dimensions of diff besides first dim (instances) 
    shaped = np.reshape(diff, (diff.shape[0], -1))
    bald = np.sum(shaped, axis=-1)

    return bald

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
