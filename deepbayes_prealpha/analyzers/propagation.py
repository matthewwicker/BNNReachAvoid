# Author: Matthew Wicker

# This is the new file that will contain all of the various form of
# over-approximate bound propagation that is used to analyze BNNs
import math
import numpy as np
import tensorflow as tf


def interval_prop(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    Function which does a fast but approximate interval propagation.
    See documentation for when this is bound is loose.
    """
    d = tf.float64
    x_l = tf.cast(x_l, dtype=d);x_u = tf.cast(x_u, dtype=d)
    W = tf.cast(W, dtype=d); b = tf.cast(b, dtype=d)
    marg = tf.cast(marg, dtype=d); b_marg = tf.cast(b_marg, dtype=d)
    x_mu = tf.divide(tf.math.add(x_u, x_l), 2.0)
    x_r =  tf.divide(tf.math.subtract(x_u, x_l), 2.0)
    W_mu = tf.cast(W, dtype=tf.float64)
    W_r =  tf.cast(marg, dtype=tf.float64)
    b_u =  tf.cast(b + b_marg, dtype=tf.float64)
    b_l =  tf.cast(b - b_marg, dtype=tf.float64)
    #h_mu = tf.math.add(tf.matmul(x_mu, W_mu), b_mu)
    h_mu = tf.matmul(x_mu, W_mu)
    x_rad = tf.matmul(x_r, tf.math.abs(W_mu))
    W_rad = tf.matmul(tf.abs(x_mu), W_r)
    Quad = tf.matmul(tf.abs(x_r), tf.abs(W_r))
    h_u = tf.add(tf.add(tf.add(tf.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = tf.add(tf.subtract(tf.subtract(tf.subtract(h_mu, x_rad), W_rad), Quad), b_l)
    return tf.cast(h_l, dtype=tf.float32), tf.cast(h_u, dtype=tf.float32)


def exact_interval_prop(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    Function which does matrix multiplication but with weight and
    input intervals.
    """
    x_l = tf.cast(x_l, dtype=tf.float32);x_u = tf.cast(x_u, dtype=tf.float32)
    W = tf.cast(W, dtype=tf.float32); b = tf.cast(b, dtype=tf.float32)
    marg = tf.cast(marg, dtype=tf.float32); b_marg = tf.cast(b_marg, dtype=tf.float32)
    x_l = tf.squeeze(x_l); x_u = tf.squeeze(x_u)
    W_l, W_u = W-marg, W+marg    	#Use eps as small symetric difference about the mean
    b_l, b_u = b-b_marg, b+b_marg   	#Use eps as small symetric difference about the mean 
    h_max = np.zeros(len(W[0])) 	#Placeholder variable for return value
    h_min = np.zeros(len(W[0])) 	#Placeholder variable for return value
    for i in range(len(W)):     	#This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])): 	# where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += np.min(out_arr)
            h_max[j] += np.max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         #Return the min and max of the intervals.
                                #(dont forget to apply activation function after)


def IBP(model, x_l, x_u, weights, gamma=0.0, predict=False, **kwargs):
    """
    Function which takes a BNN model and performs interval propagation
    """
    approx = kwargs.get('approx', False)
    h_u = x_u #tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    h_l = x_l #tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = model.posterior_var[2*(i-offset)]
        b_sigma = model.posterior_var[2*(i-offset)+1]
        marg = gamma*sigma; b_marg = gamma*b_sigma
        if(approx):
            h_l, h_u = approx_interval_prop(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        else:
            h_l, h_u = interval_prop(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        if(i < len(layers)-1 or predict):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

