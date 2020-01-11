# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 10:20:09 2019

@author: LohJZ
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import dstack
from pandas import read_csv
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from mobilenetv2_model_UCI import MobileNetv2
from tensorflow.keras.models import Model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from sklearn.metrics import classification_report, f1_score


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
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
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
    # Sitting - 0, Standing -1, laying - 2, Ascending stairs -3, Descending stairs -4, Walking -5
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	return trainX, trainy, testX, testy

######################################
###Helper function
######################################

def total_num_param(model): # Compute number of params in a model (the actual number of floats)
    trainable_count = int(np.sum([K.count_params(p) for p in list(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in list(model.non_trainable_weights)]))
    return trainable_count + non_trainable_count

# load dataset
trainX, trainy, testX, testy = load_dataset()
trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)

nb_classes = 6
nb_epochs = 10

from collections import defaultdict
final_results = defaultdict(dict)

for filter_num in [4, 8, 16, 32, 64, 128, 256]:
    print("====================================================================================")
    print("Filter size: ", filter_num)
    model = MobileNetv2(trainX.shape[1:], filter_num, nb_classes, alpha=1.0)
    model_checkpoint = callbacks.ModelCheckpoint("ucihar_weight.h5", monitor="val_loss", verbose=0, mode='auto', save_best_only=True, save_weights_only='True')
    reduce_lr = callbacks.ReduceLROnPlateau(patience=100, mode='auto', factor=0.7071, min_lr=1e-3, verbose=0) 
    callback_ls = [model_checkpoint, reduce_lr]
    
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
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
    
    model.save_weights("mobilenet-SVM_UCIHAR.h5")
       
    #############################Machine Learning########################################################
    
    model.load_weights("mobilenet-SVM_UCIHAR.h5")
    model_feat = Model(inputs=model.input,outputs=model.get_layer("GAP").output)
    feat_train = model_feat.predict(trainX)
    feat_val = model_feat.predict(testX)
    
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
    weighted_f1_score = f1_score(true_label, prediction, average='weighted' )
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
    weighted_f1_score = f1_score(true_label, prediction, average='weighted' )
    final_results[filter_num]['rf_f1'] = weighted_f1_score #save results
    print("====================================================================================")   
    
######################## Print final results #####################################################
print("Final results", final_results)
final_result_matrix = []
for kernel_size, kernel_info in final_results.items():
    final_result_matrix.append([kernel_size, kernel_info['num_parameters'], kernel_info['vanilla_f1'], kernel_info['svm_f1'], kernel_info['rf_f1']])

import pandas as pd
df = pd.DataFrame(data = final_result_matrix, columns = ['kernel_size', 'num_parameters','vanilla_f1', 'svm_f1', 'rf_f1'])    
print(df.to_string)

####################### 3 fold cross validation ##################################################
from sklearn.model_selection import KFold
from sklearn.externals import joblib 
#combine all training set and test set
trainX, trainy, testX, testy = load_dataset()
trainX = np.expand_dims(trainX, axis=2)
testX = np.expand_dims(testX, axis=2)
total_X = np.vstack([trainX, testX])
total_y = np.vstack([trainy, testy])

#perform k fold cross validation
kf = KFold(3)
best_filter_num = 8
final_results = defaultdict(dict)
kfold_index = 1
for train_index, test_index in kf.split(total_X):
    
    print("Train: ", train_index, "Test: ", test_index)
    trainX, testX = total_X[train_index], total_X[test_index]
    trainy, testy = total_y[train_index], total_y[test_index]

    print("=====================K Fold Cross Validation {}=============================".format(kfold_index))
    print("Optimal Filter size: ", best_filter_num)
    model = MobileNetv2(trainX.shape[1:],best_filter_num, nb_classes, alpha=1.0)
    model_checkpoint = callbacks.ModelCheckpoint("UCIHAR_weight.h5", monitor="val_loss", verbose=0, mode='auto', save_best_only=True, save_weights_only='True')
    reduce_lr = callbacks.ReduceLROnPlateau(patience=100, mode='auto', factor=0.7071, min_lr=1e-4, verbose=0) 
    callback_ls = [model_checkpoint, reduce_lr]
    
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print("total parameters: ", total_num_param(model))
    final_results[kfold_index]['num_parameters'] = total_num_param(model)  #save results
        
    hist = model.fit(trainX, trainy,validation_data=(testX,testy) ,batch_size=64, epochs=nb_epochs, verbose=0, callbacks=callback_ls)
    
    
    y_pred = model.predict(testX , batch_size=32, verbose=0)
    y_pred_bool = np.argmax(y_pred, axis=1)
    true_label = np.argmax(testy, axis=1)
    print(classification_report(true_label, y_pred_bool))
    
    weighted_f1_score = f1_score(true_label, y_pred_bool, average='weighted' )
    print("f1 score: ", weighted_f1_score)
    final_results[kfold_index]['vanilla_f1'] = weighted_f1_score  #save results
    
    # summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    model.save_weights("mobilenet-SVM_UCIHAR.h5")
    
    #############################Machine Learning########################################################
    
    model.load_weights("mobilenet-SVM_UCIHAR.h5")
    model_feat = Model(inputs=model.input,outputs=model.get_layer("GAP").output)
    feat_train = model_feat.predict(trainX)
    print("Shape of the  training feature extracted from model: " , feat_train.shape)
    feat_val = model_feat.predict(testX)
    print("Shape of the  test feature extracted from model: " , feat_val.shape)
    
    print("SVM")
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
    final_results[kfold_index]['svm_f1'] = weighted_f1_score  #save results
    
    
    print("Random Forest")
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
    final_results[kfold_index]['rf_f1'] = weighted_f1_score  #save results

    kfold_index += 1
    print("====================================================================================")    

##Save SVM and RF model ###
joblib.dump(svm, 'svm_UCIHAR.pkl')
joblib.dump(random_forest, 'random_forest_UCIHAR.pkl')
    
######################## Print final results #####################################################
print("Final kfold results", final_results)
final_result_matrix = []
for kfold_index, kfold_results in final_results.items():
    final_result_matrix.append([kfold_index, kfold_results['num_parameters'], kfold_results['vanilla_f1'], kfold_results['svm_f1'], kfold_results['rf_f1']])

df = pd.DataFrame(data = final_result_matrix, columns = ['kfold_index', 'num_parameters','vanilla_f1', 'svm_f1', 'rf_f1'])    
print("K fold cross validation results: ")
print(df.to_string)