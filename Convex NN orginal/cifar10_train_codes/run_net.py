import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Flatten,Lambda,Reshape,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,multiply,concatenate, Input, Activation
from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.constraints import Constraint
from keras.callbacks import callbacks 
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, LearningRateScheduler,CSVLogger

import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
from progressbar import ProgressBar
from keras.utils import np_utils
import cifar10_networks as networks 
import os
import sys


model_name = sys.argv[1]
model_type = sys.argv[2]

run_num = str(model_name)+'_'+str(model_type)


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
mean = np.mean(x_train,axis=(0,1,2))
std = np.std(x_train,axis=(0,1,2))

x_train = (x_train - mean)/ std
x_test = (x_test - mean)/ std

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


def pad(x_train = x_train):
    x_train_padded = []
    for i in x_train:
        j = cv2.copyMakeBorder( i,4,4, 4, 4, borderType = cv2.BORDER_CONSTANT)
        x_train_padded.append(j)
    return np.array(x_train_padded)

if model_name == 'mlp':
    genNorm = ImageDataGenerator(rotation_range=5,shear_range=0.2, horizontal_flip=True, fill_mode='constant')
    nb_epochs = 1300
elif model_name == 'conv':
    genNorm = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,shear_range=0.2, horizontal_flip=True, fill_mode='constant')
    nb_epochs = 1300
elif model_name == 'densenet':
    genNorm = ImageDataGenerator(width_shift_range=0.1,zoom_range=0.10,height_shift_range=0.1,rotation_range=5,shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
    nb_epochs = 450

else:
	print ("unrecognize arguments")
	exit()

x_train_padded = pad()


def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_length):
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield (batch_crops, batch_y)


batches = genNorm.flow(x_train_padded, y_train,batch_size=64)


class exp_const(tf.keras.constraints.Constraint):
    def __call__(self, W):
        rep = K.switch(W<0 , K.exp(W), W)
        return rep


def calc_acc(model,x_test = x_test, y_test = y_test):
    s1 = np.argmax(model.predict(x_test),axis=1)
    s2 = np.argmax(y_test,axis=1)
    c = 0
    for i in range(len(s1)):
        if s1[i] == s2[i]:
            c +=1
    return (c/np.shape(x_test)[0])*100


def pos_reg(weight_matrix):
    return 0.5 * K.max((K.zeros(K.shape(weight_matrix)),-weight_matrix))



architectures = ['mlp','conv','densenet']
constraint_type = ['ioc','nn','constrained_ioc']

if str(model_name) == architectures[0] and str(model_type) == constraint_type[0]:
    model = networks.mlp.get_ioc_model()
elif str(model_name) == architectures[0] and str(model_type) == constraint_type[1]:
    model = networks.mlp.get_nn_model()
elif str(model_name) == architectures[0] and str(model_type) == constraint_type[2]:
    model = networks.mlp.get_ioc_constrained_model()
elif str(model_name) == architectures[1] and str(model_type) == constraint_type[0]:
    model = networks.all_conv.get_ioc()
elif str(model_name) == architectures[1] and str(model_type) == constraint_type[1]:
    model = networks.all_conv.get_nn()
elif str(model_name) == architectures[1] and str(model_type) == constraint_type[2]:
    model = networks.all_conv.get_ioc_constrained()
elif str(model_name) == architectures[2] and str(model_type) == constraint_type[0]:
    model = networks.get_ioc()
elif str(model_name) == architectures[2] and str(model_type) == constraint_type[1]:
    model = networks.get_nn()
elif str(model_name) == architectures[2] and str(model_type) == constraint_type[2]:
    model = networks.get_ioc_constrained()
else:
	print ("unrecognize arguments")
	exit()
print (run_num)
model.summary()

def compile_bin_model(model):
    model.compile(optimizer=keras.optimizers.Adam(lr = 1e-3) ,loss=keras.losses.categorical_crossentropy,metrics = ['acc'])
    return model

model = compile_bin_model(model)


path = './'+str(run_num)+'/'
if not os.path.exists(path):
    os.makedirs(path)  
    
csv_logger = CSVLogger(path+'training.log')
filepath = path+"model_ioc.hdf5"

class save_model(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = [0]
    def on_epoch_end(self, batch, logs={}):
        if logs.get('val_acc') > np.max(self.acc):
            model.save(filepath)
        self.acc.append(logs.get('val_acc'))

#ckpts = [csv_logger,save_model()]
ckpts = [csv_logger]

history_c = model.fit_generator(crop_generator(batches,32),steps_per_epoch= int(len(x_train)/32) ,epochs=nb_epochs,validation_data=(x_test, y_test), verbose=0,callbacks = ckpts)

train_acc = calc_acc(model)
test_acc =  calc_acc(model=model,x_test=x_train,y_test=y_train)        

with open(path+'training.log','a') as fd:
        fd.write(str( train_acc))
        fd.write('\n')
with open(path+'training.log','a') as fd:
        fd.write(str(test_acc))
        fd.write('\n')


b_l = history_c.history['val_loss']
v_l = history_c.history['loss']
acc_t = history_c.history['val_acc']
acc_tr = history_c.history['acc']

fig = plt.figure(figsize=(8,10))
plt.xlim(0, nb_epochs)
plt.ylim(0, 100)

ax = fig.gca()
ax.set_xticks(np.arange(0, nb_epochs, 10))
ax.set_yticks(np.arange(0, 100, 10))

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(5) 

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(20) 

plt.plot(np.array(acc_t)*100,'r-s' ,linewidth=4)
plt.plot(np.array(acc_tr)*100,'g-o',linewidth=4)
plt.grid()

plt.title('Model Acc',fontsize=30)
plt.ylabel('accuracy',fontsize=30)
plt.xlabel('epoch',fontsize=30)
plt.legend(['test', 'train'], loc='upper left', prop={"size":30})
plt.savefig(path+'acc_plot.jpg')
