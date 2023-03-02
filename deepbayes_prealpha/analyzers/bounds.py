# Author: Matthew Wicker

# This file contains the new organization of all of the bounds
# for my thesis :) each will have the same method signature for
# relative ease of use and will hopefully be further explained in
# the forthcoming documentation
import math
import numpy as np
import tensorflow as tf

from tqdm import trange
from tqdm import tqdm

from statsmodels.stats.proportion import proportion_confint

from .propagation import *
from .attacks import *
# Below are a list of the functions which need to be implimented in the propagation file.

# IBP - by default performs IBP exactly with fixed weights

def chernoff_model_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the model robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    if(model.det):
        raise ValueError("Error computing bounds: model robustness is strictly for stochastic networks!")
    # Property Parameter
    epsilon_ball = kwargs.get('eps', 0.0)
    # Bound Parameters
    epsilon = kwargs.get('epsilon', 0.1)
    delta = kwargs.get('delta', 0.05)

    # Optional Parameters
    regression = kwargs.get('regression', False)	# if the BNN is on a regression task
    output_l = kwargs.get('output_l', None)
    output_u = kwargs.get('output_u', None)
    output_cls = kwargs.get('output_cls', None)

    approx = kwargs.get('approx', False)		# if the user wants to perform approx IBP
    loss_fn = kwargs.get('loss', None)

    # if no upper bound is supplied then we assume the user wants epsilon ball
    if(input_1 is None):
        input_1 = input_0 + epsilon_ball
        input_0 = input_0 - epsilon_ball

    if(regression == True and (output_l is None or output_u is None)):
        raise ValueError("Error computing bounds: regression was selected but no safe set was specified.")
    elif(regression == False and output_cls is None):
        print("[deepbayes] Warning: no class supplied when checking property. Stability condition assumed.")
        output_cls = np.argmax(model.predict((input_0+input_1)/2.0))
    if(verify == False and loss_fn is None):
        raise ValueError("Error computing bounds: using attack to approx worst case but no loss function was specified.")
    if(regression == False):
        num_classes = model.model.get_weights()[-1].shape[-1]

    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    estimate = 0.0
    for i in trange(chernoff_bound, desc="Sampling for Chernoff Bound Satisfaction"):
        model.set_weights(model.sample())
        # Perform the testing here
        if(verify == True):
            out_l, out_u = IBP(model, input_0, input_1, model.model.get_weights(), predict=regression, approx=approx)
            if(not regression):
                v1 = tf.one_hot(output_cls, depth=num_classes)
                v2 = 1 - v1
                worst_case = tf.math.add(tf.math.multiply(v2, out_u), tf.math.multiply(v1, out_l))
                softmax = model.model.layers[-1].activation(tf.reshape(worst_case, (1,num_classes)))
                if(np.argmax(softmax) != output_cls):
                    pass
                else:
                    estimate += 1
            else:
                if((out_l >= output_l).all() and (out_u <= output_u).all()):
                    estimate += 1
        else:
            adv_example = FGSM(model, (input_0+input_1)/2.0, loss_fn=loss_fn, eps=epsilon_ball)
            worst_case = model.predict(adv_example)
            if(not regression and (np.argmax(worst_case) == output_cls)):
                estimate += 1
            elif(regression and ((worst_case >= output_l).all() and (worst_case <= output_u).all())):
                estimate += 1
    return float(estimate/chernoff_bound)



def chernoff_decision_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the decision robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    if(model.det):
        raise ValueError("Error computing bounds: model robustness is strictly for stochastic networks!")
    # Property Parameter
    epsilon_ball = kwargs.get('eps', 0.0)
    # Bound Parameters
    epsilon = kwargs.get('epsilon', 0.1)
    delta = kwargs.get('delta', 0.05)

    # Optional Parameters
    regression = kwargs.get('regression', False)	# if the BNN is on a regression task
    output_l = kwargs.get('output_l', None)
    output_u = kwargs.get('output_u', None)
    output_cls = kwargs.get('output_cls', None)

    approx = kwargs.get('approx', False)		# if the user wants to perform approx IBP
    loss_fn = kwargs.get('loss', None)

    # if no upper bound is supplied then we assume the user wants epsilon ball
    if(input_1 is None):
        input_1 = input_0 + epsilon_ball
        input_0 = input_0 - epsilon_ball

    if(regression == True and (output_l is None or output_u is None)):
        raise ValueError("Error computing bounds: regression was selected but no safe set was specified.")
    elif(regression == False and output_cls is None):
        print("[deepbayes] Warning: no class supplied when checking property. Stability condition assumed.")
        output_cls = np.argmax(model.predict((input_0+input_1)/2.0))
    if(verify == False and loss_fn is None):
        raise ValueError("Error computing bounds: using attack to approx worst case but no loss function was specified.")
    if(regression == False):
        num_classes = model.model.get_weights()[-1].shape[-1]

    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    estimate = 0.0
    worst_out = 0.0; worst_l = 0.0; worst_u = 0.0
    for i in trange(chernoff_bound, desc="Sampling for Chernoff Bound Satisfaction"):
        model.set_weights(model.sample())
        # Perform the testing here
        if(verify == True):
            out_l, out_u = IBP(model, input_0, input_1, model.model.get_weights(), predict=regression, approx=approx)
            if(not regression):
                v1 = tf.one_hot(output_cls, depth=num_classes)
                v2 = 1 - v1
                worst_case = tf.math.add(tf.math.multiply(v2, out_u), tf.math.multiply(v1, out_l))
                softmax = model.model.layers[-1].activation(tf.reshape(worst_case, (1,num_classes)))
                worst_out += softmax
            else:
                worst_l += out_l; worst_u += out_u
        else:
            adv_example = FGSM(model, (input_0+input_1)/2.0, loss_fn=loss_fn, eps=epsilon_ball)
            worst_case = model.predict(adv_example)
            worst_out += worst_case
    if(not regression and (np.argmax(worst_case) == output_cls)):
        return 1.0
    elif(regression and ((worst_case >= output_l).all() and (worst_case <= output_u).all())):
        return 1.0
    return 0.0


# This is h_a in the paper
def _absolute_massart_halting(succ, trials, I, epsilon, delta, alpha):
    gamma = float(succ)/trials
    if(I[0] < 0.5 and I[1] > 0.5):
        return -1
    elif(I[1] < 0.5):
        val = I[1]
        h = (9/2.0)*(((3*val + epsilon)*(3*(1-val)-epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))
    elif(I[0] >= 0.5):
        val = I[0]
        h = (9/2.0)*(((3*(1-val) + epsilon)*((3*val)+epsilon))**(-1))
        return math.ceil((h*(epsilon**2))**(-1) * math.log((delta - alpha)**(-1)))

def massart_model_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the model robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    
    if(model.det):
        raise ValueError("Error computing bounds: model robustness is strictly for stochastic networks!")
    # Property Parameter
    epsilon_ball = kwargs.get('eps', 0.0)
    # Bound Parameters
    epsilon = kwargs.get('epsilon', 0.1)
    delta = kwargs.get('delta', 0.05)
    alpha = kwargs.get('alpha', 0.01)

    # Optional Parameters
    regression = kwargs.get('regression', False)        # if the BNN is on a regression task
    output_l = kwargs.get('output_l', None)
    output_u = kwargs.get('output_u', None)
    output_cls = kwargs.get('output_cls', None)

    approx = kwargs.get('approx', False)                # if the user wants to perform approx IBP
    loss_fn = kwargs.get('loss', None)

    # if no upper bound is supplied then we assume the user wants epsilon ball
    if(input_1 is None):
        input_1 = input_0 + epsilon_ball
        input_0 = input_0 - epsilon_ball

    if(regression == True and (output_l is None or output_u is None)):
        raise ValueError("Error computing bounds: regression was selected but no safe set was specified.")
    elif(regression == False and output_cls is None):
        print("[deepbayes] Warning: no class supplied when checking property. Stability condition assumed.")
        output_cls = np.argmax(model.predict((input_0+input_1)/2.0))
    if(verify == False and loss_fn is None):
        raise ValueError("Error computing bounds: using attack to approx worst case but no loss function was specified.")
    if(regression == False):
        num_classes = model.model.get_weights()[-1].shape[-1]

    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    estimate, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    while(iterations <= halting_bound):
        model.set_weights(model.sample())
        if(verify == True):
            out_l, out_u = IBP(model, input_0, input_1, model.model.get_weights(), predict=regression, approx=approx)
            if(not regression):
                v1 = tf.one_hot(output_cls, depth=num_classes)
                v2 = 1 - v1
                worst_case = tf.math.add(tf.math.multiply(v2, out_u), tf.math.multiply(v1, out_l))
                softmax = model.model.layers[-1].activation(tf.reshape(worst_case, (1,num_classes)))
                if(np.argmax(softmax) != output_cls):
                    misses += 1
                else:
                    estimate += 1
            else:
                if((out_l >= output_l).all() and (out_u <= output_u).all()):
                    estimate += 1
        else:
            adv_example = FGSM(model, (input_0+input_1)/2.0, loss_fn=loss_fn, eps=epsilon_ball)
            worst_case = model.predict(adv_example)
            if(not regression and (np.argmax(worst_case) == output_cls)):
                estimate += 1
            elif(regression and ((worst_case >= output_l).all() and (worst_case <= output_u).all())):
                estimate += 1
        iterations += 1
        # Estimate the update to the bound
        lb, ub = proportion_confint(estimate, iterations, method='beta')
        if(math.isnan(lb)):
            lb = 0.0 # Setting lb to zero if it is Nans
        if(math.isnan(ub)):
            ub = 1.0 # Setting ub to one if it is Nans
        I = [lb, ub]
        hb = _absolute_massart_halting(estimate, iterations, I, epsilon, delta, alpha)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
        print("Finished iteration: %s \t Bound: %s \t Param: %s"%(iterations-1, halting_bound, estimate/iterations), end="\r", flush=True)
    return float(estimate/iterations)

def model_robustness_lower(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute a guarenteed probabalistic lower bound on the model robustness
    of a Bayesian posterior distribution
    """
    return None

def model_robustness_upper(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute a guarenteed probabalistic lower bound on the model robustness
    of a Bayesian posterior distribution
    """
    return None

def decision_robustness_lower(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute a guarenteed probabalistic lower bound on the model robustness
    of a Bayesian posterior distribution
    """
    return None


def decision_robustness_upper(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute a guarenteed probabalistic lower bound on the model robustness
    of a Bayesian posterior distribution
    """
    return None


def log_confidence_upper(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute an upper bound on the log confidence of a model decision from a
    BNN
    """
    return None

