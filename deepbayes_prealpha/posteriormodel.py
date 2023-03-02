#Author: Matthew Wicker
import os
import math
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from tqdm import tqdm
from tqdm import trange

class PosteriorModel():
    """
    This is a class that allows users to reload saved posterior distributions.
    after saving with any optimizer this calss should represent the posterior
    in order to perform analysis of the learned model.

    Attributes
    ----------
    path_to_model : str
                   A relative or absolute path to a model saved with deepbayes
                   see the save_model function of the abstract base class `Optimizer`
                   for more info.
    deterministic : bool, optional
		   A boolean value which determines future behavior of the posterior
                   when using analysis functions. This will turn off sampling behavior
		   when computing different properties.
    """
    def __init__(self, path_to_model, deterministic=False):
        """Constructor for Class. Reads in model, will throw error if unable
        to load in a component of the posterior
        """
        if(os.path.exists(path_to_model+"/var.npy")):
            self.model = tf.keras.models.load_model(path_to_model+'/model.h5')
            print("deepbayes: detected the above model \n", self.model.summary())
            self.det = deterministic
            self.posterior_mean = np.load(path_to_model+"/mean.npy", allow_pickle=True)
            self.posterior_var = np.load(path_to_model+"/var.npy", allow_pickle=True)
            if(self.det):
                self.model.set_weights(self.posterior_mean)
            self.sample_based = False
        else:
            self.path_to_model = path_to_model
            self.sample_based = True
            self.det = False
            print("deepbayes: Attempting to load a sample based posterior")
            self.model = tf.keras.models.load_model(path_to_model+'/model.h5')
            print("deepbayes: detected the above model \n", self.model.summary())
            self.frequency = np.load(path_to_model + "/freq.npy", allow_pickle=True)
            self.frequency = self.frequency/np.sum(self.frequency)
            self.num_post_samps = len(self.frequency)
        # Read in serialized information from the posterior run: including some domain information
        #info_dict = np.load(path_to_model+'/info.pkl',allow_pickle='TRUE')
        #self.input_upper = info_dict['input_upper']
        #self.input_lower = info_dict['input_lower']
        self.presampled = False

    def sample(self, inflate=1.0):
        """Returns a list of size 2x`n_layers` (a weight followed by a bias for each layer).
        this class does not set the weight of the posterior to this sample. Follow this call
        with the .set_weights() command for that action. 
        :return: A list of the form [weight, bias, ...] for every layer in the architecture.
        :rtype: list
        """
        if(self.det):
            return self.model.get_weights()
        elif(self.sample_based == False):
            sampled_weights = []
            for i in range(len(self.posterior_mean)):
                sampled_weights.append(np.random.normal(loc=self.posterior_mean[i],
                                                    scale=self.posterior_var[i]))
        elif(self.sample_based and self.presampled):
            index = np.random.choice(range(self.num_post_samps), p=self.frequency)
            sampled_weights = self.posterior_samples[index]
        elif(self.sample_based and not self.presampled):
            index = np.random.choice(range(self.num_post_samps), p=self.frequency)
            sampled_weights = np.load(self.path_to_model+"/samples/sample_%s.npy"%(index), allow_pickle=True)
        return sampled_weights
    
    def presample(self):
        self.posterior_samples = []
        for index in range(self.num_post_samps):
            sampled_weights = np.load(self.path_to_model+"/samples/sample_%s.npy"%(index), allow_pickle=True)
            self.posterior_samples.append(sampled_weights)
        self.presampled = True
        print("Presampled into memory :)")

    def predict(self, input, n=35):
        """Return the mean of the posterior predictive distribution: this function
        samples the posterior `n` times and returns the mean softmax value from each
        sample. 
        :param input: the input to the keras model that one would like to perform inference on.
        :type input: numpy n-d array
        :param n: the number of samples to use in the posterior predictive distribution.
        :type n: int, optional
        :return: A numpy array of size (len(input),output.shape)
        :rtype: numpy ndarray
        """
        if(self.det):
            return self.model(input)
        out = -1
        for i in range(n):
            self.model.set_weights(self.sample())
            if(type(out) == int):
                out = self.model(input)
            else:
                out += self.model(input)
        return out/float(n)
    
    def _predict_logits(self, input, n=35):
        num_layers = len(self.model.layers)
        weight_mats = len(self.model.get_weights())
        last_layer = input
        for i in range(num_layers-1):
            last_layer = self.model.layers[i](last_layer)
        logits = tf.matmul(last_layer, self.model.get_weights()[weight_mats-2]) 
        return logits

    def _predict(self, input):
        return self.model(input)

    def predict_logits(self, input, n=35):
        """Return the mean of the posterior predictive distribution wrt the logits
        we sample the posterior `n` times and returns the mean logit (e.g. pre-softmax) value from each
        sample. 
        :param input: the input to the keras model that one would like to perform inference on.
        :type input: numpy n-d array
        :param n: the number of samples to use in the posterior predictive distribution.
        :type n: int, optional
        :return: A numpy array of size (len(input),output.shape)
        :rtype: numpy ndarray
        """
        out = -1
        for i in range(n):
            self.model.set_weights(self.sample())
            num_layers = len(self.model.layers)
            weight_mats = len(self.model.get_weights())
            last_layer = input
            for i in range(num_layers-1):
                last_layer = self.model.layers[i](last_layer)
            logits = tf.matmul(last_layer, self.model.get_weights()[weight_mats-2]) 
            logits += self.model.get_weights()[weight_mats-1]
            if(type(out) == int):
                out = logits
            else:
                out += logits
        return out/float(n)
        
    def set_weights(self, weights):
        self.model.set_weights(weights)

    # First order gradient
    def gradient(self, inp, loss_fn, direction, num_models=10):
        gradient_sum = tf.zeros(inp.shape)
        inp = tf.convert_to_tensor(inp)
        val = num_models
        if(num_models < 1):
            num_models = 1
        for i in range(num_models):
            if(self.det or val == -1):
                no_op = 0
            else:
                self.model.set_weights(model.sample())
            # Establish Gradient Tape Context (for input this time)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inp)
                # Get the output
                predictions = self._predict(inp)
                loss = loss_fn(direction, predictions)
            # Get the gradients
            inp_gradient = tape.gradient(loss, inp)
            try:
                gradient_sum += inp_gradient
            except:
                gradient_sum += tf.cast(inp_gradient, 'float32')
            if(model.det or val == -1):
                break
        return gradient_sum

    # Special case of gradient
    def _gradient(self, inp, loss_fn, direction):
        return self.gradient(inp, loss_fn, direction, -1)
