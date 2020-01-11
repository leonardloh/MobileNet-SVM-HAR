# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:17:01 2019

@author: LohJZ
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from mobilenetv2_model_WISDM import MobileNetv2
#from mobileHAR import mobileHAR
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, f1_score
from numpy import dstack
from pandas import read_csv

##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DATA_PATH = 'WISDMData/WISDM_ar_v1.1_raw.txt'

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 64
# Hyperparameters optimized
SEGMENT_TIME_SIZE = 128 #180

######################################
###Helper function
######################################

def total_num_param(model): # Compute number of params in a model (the actual number of floats)
    trainable_count = int(np.sum([K.count_params(p) for p in list(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in list(model.non_trainable_weights)]))
    return trainable_count + non_trainable_count

 # LOAD DATA
data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES)
data['z-axis'].replace({';': ''}, regex=True, inplace=True)
data = data.dropna()

# DATA PREPROCESSING
data_convoluted = []
labels = []

# Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
    x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
    y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
    z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
    data_convoluted.append([x, y, z])

    # Label for a data window is the label that appears most commonly
    label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
    labels.append(label)

data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)
# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values
 
# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded
 
# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
#	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
#	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y
 
# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train
	trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
	print(trainX.shape, trainy.shape)
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

######################################
###Helper function
######################################

def total_num_param(model): # Compute number of params in a model (the actual number of floats)
    trainable_count = int(np.sum([K.count_params(p) for p in list(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in list(model.non_trainable_weights)]))
    return trainable_count + non_trainable_count

# One-hot encoding
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
print("Convoluted data shape: ", data_convoluted.shape)
print("Labels shape:", labels.shape)

# SPLIT INTO TRAINING AND TEST SETS
trainX, _ , trainy, _ = train_test_split(data_convoluted, labels, test_size=0.3, random_state=RANDOM_SEED) #wisdm
_, _ , testX, testy = load_dataset()
testX = np.divide(testX, 10)
trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)
print("X train size: ", len(trainX))
print("X test size: ", len(testX))
print("y train size: ", len(trainy))
print("y test size: ", len(testy))


nb_classes = 6
nb_epochs = 10

from collections import defaultdict
final_results = defaultdict(dict)

for filter_num in [64]:
    print("====================================================================================")
    print("Filter size: ", filter_num)
    model = MobileNetv2(trainX.shape[1:],filter_num, nb_classes, alpha=1.0)
    model_checkpoint = callbacks.ModelCheckpoint("crosseval_weight.h5", monitor="val_loss", verbose=0, mode='auto', save_best_only=True, save_weights_only='True')
    reduce_lr = callbacks.ReduceLROnPlateau(patience=100, mode='auto', factor=0.7071, min_lr=1e-4, verbose=2) 
    callback_ls = [model_checkpoint, reduce_lr]
    
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("total parameters: ", total_num_param(model))
    final_results[filter_num]['num_parameters'] = total_num_param(model)  #save results
        
    hist = model.fit(trainX, trainy,validation_data=(testX,testy) ,batch_size=64, epochs=nb_epochs, verbose=0, callbacks=callback_ls)
    
    
    y_pred = model.predict(testX , batch_size=32, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)
    true_label = np.argmax(testy, axis=1)
    print(classification_report(true_label, y_pred_bool))
    weighted_f1_score = f1_score(true_label, y_pred_bool, average='weighted' )
    print("f1 score: ", weighted_f1_score)
    final_results[filter_num]['vanilla_f1'] = weighted_f1_score  #save results
    
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    model.save_weights("mobilenet-crosseval.h5")
    
    #############################Machine Learning########################################################
    
    model.load_weights("mobilenet-crosseval.h5")
    model_feat = Model(inputs=model.input,outputs=model.get_layer("GAP").output)
    feat_train = model_feat.predict(trainX)
    print("Shape of the  training feature extracted from model: " , feat_train.shape)
    feat_val = model_feat.predict(testX)
    print("Shape of the  test feature extracted from model: " , feat_val.shape)
    
    print("SVM")
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf')
    svm.fit(feat_train,np.argmax(trainy,axis=1))
    print('fitting done !!!')
    training_score = svm.score(feat_train,np.argmax(trainy,axis=1))
    print("Training score for SVM is " + str(training_score))
    validation_score = svm.score(feat_val,np.argmax(testy,axis=1))
    print("Validation score for SVM is " + str(validation_score))
    prediction = svm.predict(feat_val)
    true_label = np.argmax(testy, axis=1)
    print(classification_report(true_label, prediction))
    weighted_f1_score = f1_score(true_label, prediction, average='weighted')
    print("f1 score: ", weighted_f1_score)
    final_results[filter_num]['svm_f1'] = weighted_f1_score  #save results
    
    
    print("Random Forest")
    from sklearn.ensemble import RandomForestClassifier
    random_forest = RandomForestClassifier()
    random_forest.fit(feat_train,np.argmax(trainy,axis=1))
    print('fitting done !!!')
    training_score = random_forest.score(feat_train,np.argmax(trainy,axis=1))
    print("Training score for Random Forest is " + str(training_score))
    validation_score = random_forest.score(feat_val,np.argmax(testy,axis=1))
    print("Validation score for Random Forest is " + str(validation_score))
    prediction = random_forest.predict(feat_val)
    true_label = np.argmax(testy, axis=1)
    print(classification_report(true_label, prediction))
    weighted_f1_score = f1_score(true_label, prediction, average='weighted')
    print("f1 score: ", weighted_f1_score)
    final_results[filter_num]['rf_f1'] = weighted_f1_score  #save results

    print("====================================================================================")

######################## Print final results #####################################################
print("Final results", final_results)
final_result_matrix = []
for kernel_size, kernel_info in final_results.items():
    final_result_matrix.append([kernel_size, kernel_info['num_parameters'], kernel_info['vanilla_f1'], kernel_info['svm_f1'], kernel_info['rf_f1']])

import pandas as pd
df = pd.DataFrame(data = final_result_matrix, columns = ['kernel_size', 'num_parameters','vanilla_f1', 'svm_f1', 'rf_f1'])    
print(df.to_string)