# Author: Matthew Wicker
import numpy as np
import sklearn
from sklearn import datasets

X_train, y_train = datasets.make_moons(n_samples=1000, noise=0.075)
X_test, y_test = datasets.make_moons(n_samples=1000, noise=0.075)

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# Imports for Bayesian deep learning
import sys
sys.path.append('../..')
import deepbayesHF
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel
from deepbayesHF import analyzers

chernoff_tests = 111
massart_tests = 111
problower_tests = 111
probupper_tests = 111
probexpect_tests = 111
probexpect_tests = 111

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

f = open('build_status.log', 'a')

print("\n\n"); f.write("\n\n")
print("BOUNDS UNIT TEST RESULTS:"); f.write("BOUNDS UNIT TEST RESULTS:\n")
print("======================================= \n"); f.write("======================================= \n")

# Pre-condition: running of the classification test.
WIDTH = 64
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Load & test sample-based posterior
bayes_model = PosteriorModel("save_dir/HMC_MOONS_%s"%(WIDTH), deterministic=False)
key = "Sample-based, Model, Chernoff"
estimate = analyzers.bounds.chernoff_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=False, loss=loss_fn, epsilon=0.25)
print(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
f.write(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))

key = "Sample-based, Decision, Chernoff"
estimate = analyzers.bounds.chernoff_decision_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=False, loss=loss_fn, epsilon=0.25)
print(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
f.write(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))

# Load variational posterior
bayes_model = PosteriorModel("save_dir/VOGN_MOONS_%s"%(WIDTH), deterministic=False)
key = "Variational, Model, Chernoff"
estimate = analyzers.bounds.chernoff_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, epsilon=0.25)
estimate = analyzers.bounds.chernoff_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, approx=True, epsilon=0.25)
estimate = analyzers.bounds.chernoff_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=False, loss=loss_fn, epsilon=0.25)
print(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
f.write(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))

key = "Variational, Decision, Chernoff"
estimate = analyzers.bounds.chernoff_decision_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, epsilon=0.25)
estimate = analyzers.bounds.chernoff_decision_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, approx=True, epsilon=0.25)
estimate = analyzers.bounds.chernoff_decision_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=False, loss=loss_fn, epsilon=0.25)
print(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
f.write(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))

key = "Variational, Model, Massart"
estimate = analyzers.bounds.massart_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, epsilon=0.25)
estimate = analyzers.bounds.massart_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, approx=True, epsilon=0.25)
estimate = analyzers.bounds.massart_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=False, loss=loss_fn, epsilon=0.25)
print(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
f.write(f"BOUND METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))


# Load deterministic network
bayes_model = PosteriorModel("save_dir/SGD_MOONS_%s"%(WIDTH), deterministic=True)
try:
    estimate = analyzers.bounds.chernoff_model_robustness(bayes_model, np.asarray([X_train[0]]), eps=0.01, verify=True, epsilon=0.5)
except:
    print("Correctly threw error")

print("\n\n"); f.write("\n\n")

f.close()
