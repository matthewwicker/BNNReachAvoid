import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

X_train, y_train = sklearn.datasets.make_regression(n_samples=500, n_features=10)
X_test, y_test = X_train[0:300], y_train[0:300]
#X_train, y_train = datasets.load_diabetes(return_X_y=True)
#X_test, y_test = datasets.load_diabetes(return_X_y=True)

y_train, y_test = y_train.reshape(-1,1), y_test.reshape(-1,1)

X_scaler = StandardScaler()
X_scaler.fit(X_train)

y_scaler = StandardScaler()
y_scaler.fit(y_train)

X_train, X_test = X_scaler.transform(X_train), X_scaler.transform(X_test)
y_train, y_test = y_scaler.transform(y_train.reshape(-1,1)), y_scaler.transform(y_test.reshape(-1,1))

print(X_train.shape)
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

# Imports for Bayesian deep learning
import sys
sys.path.append('../..')
import deepbayesHF
from deepbayesHF import optimizers
from deepbayesHF import PosteriorModel

likelihood = tf.keras.losses.MeanSquaredError()

WIDTH = 128
model = Sequential()
model.add(Dense(WIDTH, activation="linear", input_shape=(1, 10)))
#model.add(Dense(WIDTH, activation="linear"))
#model.add(Dense(WIDTH, activation="tanh"))
model.add(Dense(1, activation="linear"))


inference_dict = {"VOGN":optimizers.VOGN(), "BBB":optimizers.BBB(), 
                  "NA":optimizers.NA(), "SWAG":optimizers.SWAG(),
                  "HMC":optimizers.HMC(), "SGHMC":optimizers.SGHMC(),
                  "SGLD":optimizers.SGLD(), "CSGLD":optimizers.CSGLD(),
                  "SGD":optimizers.SGD(), "ADAM":optimizers.Adam()}

#inference_dict = {"SGD":optimizers.SGD()}

pass_fail = []

for key, inference in inference_dict.items():
    print("~~~~~~~~~~~~~ TESTING %s INFERENCE ~~~~~~~~~~~~~~~"%(key))
    try:
        bayes_model = inference.compile(model, loss_fn=likelihood,
                          epochs=25, batch_size=32, decay=0.15,
                          inflate_prior=0.1, mode='regression')
        bayes_model.learning_rate *= 0.5
        bayes_model.train(X_train, y_train, X_train, y_train)
        bayes_model.save("save_dir/%s_BOSTON_%s"%(key, WIDTH))
        accuracy = tf.keras.metrics.MeanSquaredError()
        preds = bayes_model.predict(X_test)
        accuracy.update_state(preds, y_test)
        acc = accuracy.result()
        print("SAVED MODEL WITH ERROR: ", acc)


        bayes_model = None
        det = False
        if(key == "SGD" or key == "ADAM"):
            det = True
        bayes_model = PosteriorModel("save_dir/%s_BOSTON_%s"%(key, WIDTH), deterministic=det)

        accuracy = tf.keras.metrics.MeanSquaredError()
        preds = bayes_model.predict(X_test)
        accuracy.update_state(preds, y_test)
        acc = accuracy.result()
        print("LOADED MODEL WITH ERROR: ", acc)
        if(accuracy.result() < 0.6):
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

f = open('build_status.log', 'w+')
print("\n\n"); f.write("\n\n")
print("REGRESSION INFERENCE TEST RESULTS:"); f.write("REGRESSION INFERENCE TEST RESULTS:\n")
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
        print(f"INFERENCE METHOD {bcolors.WARNING}\t [ %s ] \t \t WORKING (Warning: High Test Loss) {bcolors.ENDC}"%(key))
        f.write(f"INFERENCE METHOD {bcolors.WARNING}\t [ %s ] \t \t WORKING (Warning: High Test Loss) {bcolors.ENDC} \n"%(key))
    index += 1
print("\n\n"); f.write("\n\n")

f.close() 
