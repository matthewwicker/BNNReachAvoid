# Author: Matthew Wicker

import math
import numpy as np
import tensorflow as tf
from tqdm import trange

def eps_LRP(model, input, epsilon=0.01, direction=-1):
    """
	LRP algorithm (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
	param model - a deepbayes posterior or optimizer object
	param input - an input to pass through the model

	returns np.array of the same shape as input with a 
		'relevance' attribution for each feature dimension
    """
    # We need to arrive at the ouput via manual propagation through the network
    # because we will need the activations to normalize the eps-LRP.
    
    layers = model.model.layers
    offset = 0
    weights = model.model.get_weights()
    ws = []
    biases = []
    h = input
    activations = [h]
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h = model.model.layers[i](h)
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        ws.append(w)
        pre_act = tf.add(tf.matmul(h, w),b)
        biases.append(b)
        h = model.model.layers[i].activation(pre_act)
        activations.append(h)
    output = h
    print("Output we wish to explain: ", output)
    if(direction == -1):
        direction = output
    # We zero out a copy of layer values to get the inital relevance
    #layers = model.model.get_weights()
    relevance = [np.asarray(i)*0.0 for i in activations] 
    relevance[-1] = direction
    # using https://arxiv.org/pdf/1711.06104.pdf as a guide here :) 
    print("================ BACKPROPPING DECISION RELEVANCE ============")
    for l in range(len(layers)-1, -1, -1):
        b = biases[l]
        print("Starting relevance prop from layer %s"%(l))
        print("Weight shape: ", (ws[l].shape))
        print("Iterating i (%s) ->  j (%s)"%(len(ws[l][:,0]), len(ws[l][0])))
        print("These should be the same %s == %s"%(len(b), len(ws[l][0])))
        for i in range(len(ws[l][:,0])):
            val = 0
            for j in range(len(ws[l][0])):
                denom = []
                for i_p in range(len(ws[l][:,0])):
                    help_me_pls = tf.cast(tf.squeeze(b)[j], dtype=tf.float64)
                    denom.append(ws[l][i_p][j]*tf.squeeze(activations[l])[i_p] + help_me_pls)
                denom_sum = tf.math.reduce_sum(denom)
                eps = epsilon * tf.sign(tf.math.reduce_sum(denom))
                rel = (ws[l][i][j]*tf.squeeze(activations[l])[i])/(denom_sum * eps)
                #print("value: ",tf.squeeze(relevance[l+1])[j], "backprop: ", rel)
                rel *= tf.squeeze(relevance[l+1])[j]
                val += rel
            relevance[l][0][i] = val

    return relevance[0]


def deeplift(model, input_ref, input):
    """
        DeepLift algorithm (http://proceedings.mlr.press/v70/shrikumar17a.html)
        param model - a deepbayes posterior or optimizer object
	param input_ref - a refence input to check against the passed input
        param input - an input to pass through the model

        returns np.array of the same shape as input with a
                'relevance' attribution for each feature dimension
    """
    return None

def shapely_values(model, input, samples=100):
    """
        Shapley value based on sampling algorithm (https://www.sciencedirect.com/science/article/pii/S0305054808000804)
        param model - a deepbayes posterior or optimizer object
        param input - an input to pass through the model
        param samples - a number of samples to use to compute the value

        returns np.array of the same shape as input with a 
                'relevance' attribution for each feature dimension
    """
    return None


def occlusion_attr(model, input):
    """
        Occlusion attribution algorithm (https://arxiv.org/abs/1311.2901)
        param model - a deepbayes posterior or optimizer object
        param input - an input to pass through the model

        returns np.array of the same shape as input with a 
                'relevance' attribution for each feature dimension
    """
    return None
