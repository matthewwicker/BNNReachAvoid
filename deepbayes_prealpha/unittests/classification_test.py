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

likelihood = tf.keras.losses.SparseCategoricalCrossentropy()

WIDTH = 64
model = Sequential()
model.add(Dense(WIDTH, activation="relu", input_shape=(1, 2)))
model.add(Dense(2, activation="softmax"))


inference_dict = {"VOGN":optimizers.VOGN(), "BBB":optimizers.BBB(), 
                  "NA":optimizers.NA(), "SWAG":optimizers.SWAG(),
                  "HMC":optimizers.HMC(), "SGHMC":optimizers.SGHMC(),
                  "SGLD":optimizers.SGLD(), "CSGLD":optimizers.CSGLD(),
                  "SGD":optimizers.SGD(), "ADAM":optimizers.Adam()}

#inference_dict = {"CSGLD":optimizers.CSGLD()}

pass_fail = []

for key, inference in inference_dict.items():
    print("~~~~~~~~~~~~~ TESTING %s INFERENCE ~~~~~~~~~~~~~~~"%(key))
    try:
        bayes_model = inference.compile(model, loss_fn=likelihood,
                          epochs=10, batch_size=64,
                          inflate_prior=1.0, mode='classification')
        bayes_model.train(X_train, y_train, X_train, y_train)
        bayes_model.save("save_dir/%s_MOONS_%s"%(key, WIDTH))

        bayes_model = None
        det = False
        if(key == "SGD" or key == "ADAM"):
            det = True
        bayes_model = PosteriorModel("save_dir/%s_MOONS_%s"%(key, WIDTH), deterministic=det)

        accuracy = tf.keras.metrics.Accuracy()
        preds = bayes_model.predict(X_test)
        accuracy.update_state(np.argmax(preds, axis=1), y_test)
        acc = accuracy.result()
        print(acc)
        if(accuracy.result() > 0.8):
            print("Inference Test: PASSED")
            pass_fail.append("PASSED")
        else:
            print("Inference Test: FAILED")
            pass_fail.append("FAILEDT")
    except Exception as e:
        print(e)
        print("Inference Test: FAILED")
        pass_fail.append("FAILED")



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
print("CLASSIFICATION INFERENCE TEST RESULTS:"); f.write("CLASSIFICATION INFERENCE TEST RESULTS:\n")
print("======================================= \n"); f.write("======================================= \n")
index = 0
for key, inference in inference_dict.items():
    if(pass_fail[index] == "PASSED"):
        print(f"INFERENCE METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC}"%(key))
        f.write(f"INFERENCE METHOD {bcolors.OKGREEN}\t [ %s ] \t \t PASSED {bcolors.ENDC} \n"%(key))
    elif(pass_fail[index] == "FAILED"):
        print(f"INFERENCE METHOD {bcolors.FAIL}\t [ %s ] \t \t FAILED (Error) {bcolors.ENDC}"%(key))
        f.write(f"INFERENCE METHOD {bcolors.FAIL}\t [ %s ] \t \t FAILED (Error) {bcolors.ENDC}\n"%(key))
    elif(pass_fail[index] == "FAILEDT"):
        print(f"INFERENCE METHOD {bcolors.WARNING}\t [ %s ] \t \t WORKING (Warning: Low Test Acc) {bcolors.ENDC}"%(key))
        f.write(f"INFERENCE METHOD {bcolors.WARNING}\t [ %s ] \t \t WORKING (Warning: Low Test Acc) {bcolors.ENDC} \n"%(key))
    index += 1
print("\n\n"); f.write("\n\n")

f.close() 

