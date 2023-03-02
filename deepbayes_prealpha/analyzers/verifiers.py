# Author: Matthew Wicker


from statsmodels.stats.proportion import proportion_confint
import math
import numpy as np
import tensorflow as tf
from tqdm import trange
from . import attacks

def propagate_interval(W, b, x_l, x_u, marg=0, b_marg=0):
    marg = tf.divide(marg, 2)
    b_marg = tf.divide(b_marg, 2)

    x_mu = tf.divide(tf.math.add(x_u, x_l), 2)
    x_r =  tf.divide(tf.math.subtract(x_u, x_l), 2)

    W_mu = tf.cast(W, dtype=tf.float64)
    W_r =  tf.cast(marg, dtype=tf.float64)

    b = tf.cast(b, dtype=tf.float64)
    b_marg = tf.cast(b_marg, dtype=tf.float64)
    b_u =  tf.cast(b + b_marg, dtype=tf.float64)
    b_l =  tf.cast(b - b_marg, dtype=tf.float64)

    h_mu = tf.matmul(x_mu, W_mu)
    x_rad = tf.matmul(x_r, tf.math.abs(W_mu))
    try:
        W_rad = tf.matmul(tf.abs(x_mu), W_r)
        Quad = tf.matmul(tf.abs(x_r), tf.abs(W_r))
    except:
        W_rad = 0 # This happens if margin stays zero
        Quad = 0

    h_u = tf.add(tf.add(tf.add(tf.add(h_mu, x_rad), W_rad), Quad), b_u)
    h_l = tf.add(tf.subtract(tf.subtract(tf.subtract(h_mu, x_rad), W_rad), Quad), b_l)

    return h_l, h_u

def propagate_interval_exact(W, b, x_l, x_u, marg=0, b_marg=0):
    """
    Function which does matrix multiplication but with weight and
    input intervals.
    """
    x_l = tf.cast(x_l, dtype=tf.float32);x_u = tf.cast(x_u, dtype=tf.float32)
    W = tf.cast(W, dtype=tf.float32); b = tf.cast(b, dtype=tf.float32)
    marg = tf.cast(marg, dtype=tf.float32); b_marg = tf.cast(b_marg, dtype=tf.float32)
    x_l = tf.squeeze(x_l); x_u = tf.squeeze(x_u)
    W_l, W_u = W-marg, W+marg           #Use eps as small symetric difference about the mean
    b_l, b_u = b-b_marg, b+b_marg       #Use eps as small symetric difference about the mean 
    h_max = np.zeros(len(W[0]))         #Placeholder variable for return value
    h_min = np.zeros(len(W[0]))         #Placeholder variable for return value
    for i in range(len(W)):             #This is literally just a step-by-step matrix multiplication
        for j in range(len(W[0])):      # where we are taking the min and max of the possibilities
            out_arr = [W_l[i][j]*x_l[i], W_l[i][j]*x_u[i],
                       W_u[i][j]*x_l[i], W_u[i][j]*x_u[i]]
            h_min[j] += min(out_arr)
            h_max[j] += max(out_arr)
    h_min = h_min + b_l
    h_max = h_max + b_u
    return h_min, h_max         #Return the min and max of the intervals.
                                #(dont forget to apply activation function after)

def IBP_state(model, s0, s1, weights, weight_margin=0, logits=True, exact=False):
    h_l = s0
    h_u = s1
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
        marg = weight_margin*sigma
        b_marg = weight_margin*b_sigma
        if(exact):
            h_l, h_u = propagate_interval_exact(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        else:
            h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        h_l = model.model.layers[i].activation(h_l)
        h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

def IBP_full_multiproc(args):
    actives, s0, s1, weights, weight_margin, predicate, posterior_var = args
    h_l = s0
    h_u = s1
    layers = int(len(posterior_var)/2)
    offset = 0
    for i in range(layers):
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = posterior_var[2*(i-offset)]
        b_sigma = posterior_var[2*(i-offset)+1]
        marg = weight_margin*sigma
        b_marg = weight_margin*b_sigma
        h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg, b_marg=b_marg)
        #print("Layer %s Bounds: "%(i), h_l, h_u)
        h_l = actives[i](h_l) #model.model.layers[i].activation(h_l)
        h_u = actives[i](h_u) #model.model.layers[i].activation(h_u)

    ol, ou = h_l, h_u
    lower = np.squeeze(s0)[0:len(ol)] + ol;
    upper = np.squeeze(s1)[0:len(ou)] + ou

    if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(lower), np.squeeze(upper))):
        ol = np.squeeze(ol); ou = np.squeeze(ou)
        return [lower, upper]
    else:
        return None


def IBP_conf(model, s0, s1, weights, weight_margin=0, logits=True):
    h_l = s0
    h_u = s1
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        if(i == len(layers)-1):
            # iterate over the number of classes:
            softmax_diffs = []
            num_classes = np.asarray(weights[(2*(i-offset))+1]).shape[-1]
            print("number of classes: ", num_classes)
            for k in range(num_classes):
                class_diffs = []
                for l in range(num_classes):
                    diff = weights[2*(i-offset)][:,k] - weights[2*(i-offset)][:,l]
                    max_diff = np.maximum(diff, 0)
                    min_diff = np.minimum(diff, 0)
                    bias_diff = weights[(2*(i-offset))+1][k] - weights[(2*(i-offset))+1][l]
                    #logit_diff = (max_diff * h_u) + (min_diff * h_l)  + bias_diff 
                    logit_diff = np.sum(max_diff * h_u) + np.sum(min_diff * h_l)  + bias_diff 
                    class_diff = logit_diff 
                    #print(class_diff)
                    #if(class_diff == 0.0):
                    #    print("These should be the same ", k,l)
                    class_diffs.append(class_diff)
                diff_val = -np.log(np.sum(np.exp(-1*np.asarray(class_diffs))))
                #diff_val = np.max(class_diffs)
                #diff_val = np.max(diff_val)
                print("Diff value for class ", k, " is ", diff_val)
                softmax_diffs.append(diff_val)
            return softmax_diffs
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        sigma = model.posterior_var[2*(i-offset)]
        marg = weight_margin*sigma
        h_l, h_u = propagate_interval(w, b, h_l, h_u, marg=marg)
    return None #error state if we hit this!

def IBP_learning(model, inp, weights, eps, predict=False):
    h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        h_l, h_u = propagate_interval(w, b, h_l, h_u)
        if(i < len(layers)-1):
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

def pIBP(model, inp_l, inp_u, weights, predict=False):
    #if(predict == False):
    #    h_u = tf.clip_by_value(tf.math.add(inp, eps), 0.0, 1.0)
    #    h_l = tf.clip_by_value(tf.math.subtract(inp, eps), 0.0, 1.0)
    #else:
    h_u = inp_u
    h_l = inp_l
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        if(len(layers[i].get_weights()) == 0):
            h_u = model.model.layers[i](h_u)
            h_l = model.model.layers[i](h_l)
            offset += 1
            continue
        w, b = weights[2*(i-offset)], weights[(2*(i-offset))+1]
        if(len(w.shape) == 2):
            h_l, h_u = propagate_interval(w, b, h_l, h_u)
            activate = True
        elif(len(w.shape) == 4):
            h_l, h_u = propagate_conv2d(w, b, h_l, h_u)
            activate = True
        if(predict == False and i >= len(layers)-1):
            continue
        else:
            h_l = model.model.layers[i].activation(h_l)
            h_u = model.model.layers[i].activation(h_u)
    return h_l, h_u

# Code for merging overlapping intervals. Taken from here: 
# https://stackoverflow.com/questions/49071081/merging-overlapping-intervals-in-python
# This function simple takes in a list of intervals and merges them into all 
# continuous intervals and returns that list 
def merge_intervals(intervals):
    sorted_intervals = sorted(intervals)
    interval_index = 0
    intervals = np.asarray(intervals)
    for  i in sorted_intervals:
        if i[0] > sorted_intervals[interval_index][1]:
            interval_index += 1
            sorted_intervals[interval_index] = i
        else:
            sorted_intervals[interval_index] = [sorted_intervals[interval_index][0], i[1]]
    return sorted_intervals[:interval_index+1] 


"""
Given a set of disjoint intervals, compute the probability of a random
sample from a guassian falling in these intervals. (Taken from lemma)
of the document
"""
import math
from scipy.special import erf
def compute_erf_prob(intervals, mean, var):
    prob = 0.0
    for interval in intervals:
        #val1 = erf((mean-interval[0])/(math.sqrt(2)*(var)))
        #val2 = erf((mean-interval[1])/(math.sqrt(2)*(var)))
        val1 = erf((mean-interval[0])/(math.sqrt(2*var)))
        val2 = erf((mean-interval[1])/(math.sqrt(2*var)))
        prob += 0.5*(val1-val2)
    return prob

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into maximum continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a weight matrix
"""
def compute_interval_probs_weight(vector_intervals, marg, mean, var):
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)

def compute_interval_probs_weight_m(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; # vars = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in trange(len(vector_intervals[0])):
        for j in range(len(vector_intervals[0][0])):
            intervals = []
            for num_found in range(len(vector_intervals)):
                interval = [vector_intervals[num_found][i][j]-(var[i][j]*marg), vector_intervals[num_found][i][j]+(var[i][j]*marg)]
                intervals.append(interval)
            p = compute_erf_prob(merge_intervals(intervals), means[i][j], var[i][j])
            prob_vec[i][j] = p
    return np.asarray(prob_vec)

"""
Given a set of possibly overlapping intervals:
    - Merge all intervals into continuous, disjoint intervals
    - compute probability of these disjoint intervals
    - do this for ALL values in a *flat* bias matrix (vector)
"""
def compute_interval_probs_bias(vector_intervals, marg, mean, var):
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec)

def compute_interval_probs_bias_m(arg):
    vector_intervals, marg, mean, var = arg
    means = mean; #stds = var
    prob_vec = np.zeros(vector_intervals[0].shape)
    for i in range(len(vector_intervals[0])):
        intervals = []
        for num_found in range(len(vector_intervals)):
            #!*! Need to correct and make sure you scale margin
            interval = [vector_intervals[num_found][i]-(var[i]*marg), vector_intervals[num_found][i]+(var[i]*marg)]
            intervals.append(interval)
        p = compute_erf_prob(merge_intervals(intervals), means[i], var[i])
        prob_vec[i] = p
    return np.asarray(prob_vec).tolist()

def compute_probability(model, weight_intervals, margin, verbose=True, n_proc=10):
    full_p = 1.0
    if(verbose == True):
        func = range
    else:
        func = range
    args_bias = []
    args_weights = []
    for i in func(len(model.posterior_mean)):
        if(i % 2 == 0): # then its a weight vector
            #p = compute_interval_probs_weight(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
            args_weights.append((weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))
        else: # else it is a bias vector
            #p = compute_interval_probs_bias(weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i]))
            args_bias.append((weight_intervals[i], margin, model.posterior_mean[i], np.asarray(model.posterior_var[i])))

    from multiprocessing import Pool
    proc_pool = Pool(n_proc)
    ps_bias = proc_pool.map(compute_interval_probs_bias_m, args_bias)
    ps_weight = proc_pool.map(compute_interval_probs_weight_m, args_weights)
    proc_pool.close()
    proc_pool.join()

    import itertools
    ps_bias = np.concatenate(ps_bias).ravel()
    ps_weight = np.asarray(list(itertools.chain(*(itertools.chain(*ps_weight)))))

    full_p *= np.prod(ps_bias)
    full_p *= np.prod(ps_weight)
    return full_p

def IBP_prob(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0):
    #w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []
    for i in range(samples):
        model.model.set_weights(model.sample(inflate=inflate))
        ol, ou = IBP_state(model, s0, s1, model.model.get_weights(), w_marg)
        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(ol), np.squeeze(ou))):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
            #lower = np.squeeze(s0)[0:len(ol)] + ol; upper = np.squeeze(s1)[0:len(ou)] + ou
            safe_outputs.append([-1,1]) # This is used ONLY for control loops which needs its own verification section
    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg**2)
    return p, np.squeeze(safe_outputs)


def propagate(model, s0, s1, w_marg=0.0):
    #model.model.set_weights(w)
    ol, ou = IBP_state(model, s0, s1, model.model.get_weights(), w_marg)
    lower = np.squeeze(s0)[0:len(ol)] + ol;
    upper = np.squeeze(s1)[0:len(ou)] + ou
    return lower, upper

def IBP_fixed_w(model, s0, s1, w_marg, w, predicate, i0=0):
    model.model.set_weights(w)
    ol, ou = IBP_state(model, s0, s1, w, w_marg)
    lower = np.squeeze(s0)[0:len(ol)] + ol;
    upper = np.squeeze(s1)[0:len(ou)] + ou
    if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(lower), np.squeeze(upper))):
        #p = compute_probability(model, np.swapaxes(np.asarray([w]),1,0), w_marg)
        return 1.0, -1
    else:
        return 0.0, -1


def IBP_prob_dyn(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0, full_state=False):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []

    for i in trange(samples):
        model.model.set_weights(model.sample(inflate=inflate))
        ol, ou = IBP_state(model, s0, s1, model.model.get_weights(), w_marg)

        lower = np.squeeze(s0)[0:len(ol)] + ol;
        upper = np.squeeze(s1)[0:len(ou)] + ou

        if(predicate(np.squeeze(s0), np.squeeze(s1), np.squeeze(lower), np.squeeze(upper))):
            safe_weights.append(model.model.get_weights())
            ol = np.squeeze(ol); ou = np.squeeze(ou)
            safe_outputs.append([lower, upper]) # This is used ONLY for control loops which needs its own verification section

    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)




def IBP_prob_dyn_m(model, s0, s1, w_marg, samples, predicate, i0=0, inflate=1.0, full_state=False, n_proc=20):
    w_marg = w_marg**2
    safe_weights = []
    safe_outputs = []

    actives = []
    layers = model.model.layers
    offset = 0
    for i in range(len(layers)):
        actives.append(model.model.layers[i].activation)

    #actives, s0, s1, weights, weight_margin
    args = []
    weights = []
    for i in range(samples):
        #model.model.set_weights(model.sample(inflate=inflate))
        w = model.sample(inflate=inflate)
        #actives, s0, s1, weights, weight_margin, predicate
        arg = (actives, s0, s1, w, w_marg, predicate, model.posterior_var)
        weights.append(w)
        args.append(arg)

    weights = np.asarray(weights)

    from multiprocessing import Pool
    proc_pool = Pool(n_proc)
    #IBP_full_multiproc
    safe_outputs = proc_pool.map(IBP_full_multiproc, args)
    proc_pool.close()
    proc_pool.join()

    none_indexes = []; ind = 0
    safe_outs = []
    for i in safe_outputs:
        if(i is not None):
            none_indexes.append(ind)
            safe_outs.append(i)
        ind += 1
    safe_weights = weights[none_indexes]
    safe_outputs = safe_outs #safe_outputs[none_indexes]

    print("Found %s safe intervals"%(len(safe_weights)))
    if(len(safe_weights) < 2):
        return 0.0, -1
    p = compute_probability(model, np.swapaxes(np.asarray(safe_weights),1,0), w_marg)
    return p, np.squeeze(safe_outputs)




def IBP_prob_w(model, s0, s1, w_marg, w, predicate, i0=0):
    model.model.set_weights(model.sample())
    ol, ou = IBP_state(model, s0, s1, w, w_marg)
    if(predicate(np.squeeze(s0)[i0:i0+2], np.squeeze(s1)[i0:i0+2], np.squeeze(ol)[i0:i0+2], np.squeeze(ou)[i0:i0+2])):
        p = compute_probability(model, np.swapaxes(np.asarray([w]),1,0), w_marg)
        return p, -1
    else:
        return 0.0, -1



# also known as the chernoff bound
def okamoto_bound(epsilon, delta):
    return (-1*.5) * math.log(float(delta)/2) * (1.0/(epsilon**2))

# This is h_a in the paper
def absolute_massart_halting(succ, trials, I, epsilon, delta, alpha):
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

"""

"""
def chernoff_bound_verification(model, inp, eps, cls, **kwargs):
    from tqdm import trange
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    softmax = 0
    for i in trange(chernoff_bound, desc="Sampling for Chernoff Bound Satisfaction"):
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        v1 = tf.squeeze(v1); v2 = tf.squeeze(v2)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(type(softmax) == int):
            softmax = model.model.layers[-1].activation(worst_case)
        else:
            softmax += model.model.layers[-1].activation(worst_case)
    return softmax
    #print("Not yet implimented")


"""
property - a function that takes a vector and returns a boolean if it was successful
"""
def massart_bound_check(model, inp, eps, cls, **kwargs):
    delta = kwargs.get('delta', 0.3)
    alpha = kwargs.get('alpha', 0.05)
    confidence = kwargs.get('confidence', 0.95)
    verbose = kwargs.get('verbose', False)
    
    atk_locs = []
    epsilon = 1-confidence
    chernoff_bound = math.ceil( (1/(2*epsilon**2)) * math.log(2/delta) )
    print("BayesKeras. Maximum sample bound = %s"%(chernoff_bound))
    successes, iterations, misses = 0.0, 0.0, 0.0
    halting_bound = chernoff_bound
    I = [0,1]
    while(iterations <= halting_bound):
        if(iterations > 0 and verbose):
            print("Working on iteration: %s \t Bound: %s \t Param: %s"%(iterations, halting_bound, successes/iterations))  
        model.set_weights(model.sample())
        logit_l, logit_u = IBP(model, inp, model.model.get_weights(), eps, predict=False)
        v1 = tf.one_hot(cls, depth=10)
        v2 = 1 - tf.one_hot(cls, depth=10)
        worst_case = tf.math.add(tf.math.multiply(v2, logit_u), tf.math.multiply(v1, logit_l))
        if(np.argmax(np.squeeze(worst_case)) != cls):
            misses += 1
            result = 0
        else:
            result = 1
        successes += result
        iterations += 1
        # Final bounds computation below
        lb, ub = proportion_confint(successes, iterations, method='beta')
        if(math.isnan(lb)):
            lb = 0.0 # Setting lb to zero if it is Nans
        if(math.isnan(ub)):
            ub = 1.0 # Setting ub to one if it is Nans
        I = [lb, ub]
        hb = absolute_massart_halting(successes, iterations, I, epsilon, delta, alpha)
        if(hb == -1):
            halting_bound = chernoff_bound
        else:
            halting_bound = min(hb, chernoff_bound)
    if(verbose):
        print("Exited becuase %s >= %s"%(iterations, halting_bound))
    return successes/iterations
    #return None





def chernoff_model_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the model robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    return None

def chernoff_decision_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the decision robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    return None

def massart_model_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the model robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    return None

def massart_decision_robustness(model, input_0, input_1=None, verify=True, **kwargs):
    """
    Function to compute the decision robustness of a given Bayesian posterior
    with statistically guarenteed precision via Chernoff concentration
    """
    return None

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
