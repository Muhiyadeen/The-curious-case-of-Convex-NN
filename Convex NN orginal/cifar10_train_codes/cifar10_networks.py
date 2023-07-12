import tensorflow as tf
import keras
from keras.layers import Dense, Dropout, Flatten,Lambda,Reshape,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,multiply,concatenate,Input, Activation
from keras import backend as K
from keras.models import Model
from keras import optimizers
from keras.constraints import Constraint
from keras.layers import Convolution2D,GlobalAveragePooling2D,AveragePooling2D,Concatenate



class exp_const(tf.keras.constraints.Constraint):
	def __call__(self, W):
	    rep = K.switch(W<0 , K.exp(W-5), W)
	    return rep

class mlp():
	def get_ioc_constrained_model():
		inp1 = Input(shape=(32,32,3,))
		inp = Reshape((-1,))(inp1)
		l11 = Dense(800, activation='linear',kernel_initializer='glorot_uniform')(inp)
		l12 = BatchNormalization(scale=True)(l11)
		l13 = Activation('elu')(l12)

		l21 = Dense(800, activation='linear',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l13)
		l22 = BatchNormalization(scale=True,gamma_constraint = exp_const())(l21)
		l23 = Activation('elu')(l22)

		l31 = Dense(800, activation='linear',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l23)
		l32 = BatchNormalization(scale=True,gamma_constraint = exp_const())(l31)
		l33 = Activation('elu')(l32)

		out = Dense(10, activation='softmax',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l33)
		model = Model(inputs=[inp1], outputs=[out])
		return model

	def get_ioc_model():
		inp1 = Input(shape=(32,32,3,))
		inp = Reshape((-1,))(inp1)
		l11 = Dense(800, activation='linear',kernel_initializer='glorot_uniform')(inp)
		l12 = BatchNormalization(scale=True)(l11)
		l13 = Activation('elu')(l12)

		l21 = Dense(800, activation='linear',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l13)
		l22 = BatchNormalization(scale=True)(l21)
		l23 = Activation('elu')(l22)

		l31 = Dense(800, activation='linear',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l23)
		l32 = BatchNormalization(scale=True)(l31)
		l33 = Activation('elu')(l32)

		out = Dense(10, activation='softmax',kernel_initializer='glorot_uniform',W_constraint = exp_const())(l33)
		model = Model(inputs=[inp1], outputs=[out])
		return model
	
	def get_nn_model():
		inp1 = Input(shape=(32,32,3,))
		inp = Reshape((-1,))(inp1)
		l11 = Dense(800, activation='linear',kernel_initializer='glorot_uniform')(inp)
		l12 = BatchNormalization(scale=True)(l11)
		l13 = Activation('relu')(l12)

		l21 = Dense(800, activation='linear',kernel_initializer='glorot_uniform')(l13)
		l22 = BatchNormalization(scale=True)(l21)
		l23 = Activation('relu')(l22)

		l31 = Dense(800, activation='linear',kernel_initializer='glorot_uniform')(l23)
		l32 = BatchNormalization(scale=True)(l31)
		l33 = Activation('relu')(l32)

		out = Dense(10, activation='softmax',kernel_initializer='glorot_uniform')(l33)
		model = Model(inputs=[inp1], outputs=[out])
		return model

class all_conv():
	def get_ioc():
		input_l = Input(shape=(32, 32, 3,))
		lay1_1 = Convolution2D(96*2, (3, 3), padding = 'same',activation='elu')(input_l)
		lay1_2 = Convolution2D(96, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay1_1)
		norm_1 = BatchNormalization()(lay1_2)
		lay1_3 = Convolution2D(96, (3, 3), padding = 'same',activation='elu', subsample = (2,2),W_constraint = exp_const())(norm_1)
		lay1 = Dropout(0.5)(lay1_3)


		lay2_1 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay1)
		norm_2_1 = BatchNormalization()(lay2_1)
		lay2_2 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(norm_2_1)
		lay2_3 = Convolution2D(192, (3, 3), padding = 'same',activation='elu', subsample = (2,2),W_constraint = exp_const())(lay2_2)
		norm_2_2 = BatchNormalization()(lay2_3)
		lay2 = Dropout(0.5)(norm_2_2)

		lay3_1 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay2)
		lay3_2 = Convolution2D(192, (1, 1), padding = 'valid',activation='elu',W_constraint = exp_const())(lay3_1)
		norm_3 = BatchNormalization()(lay3_2)
		lay3_3 = Convolution2D(10, (1, 1), padding = 'valid',activation='elu', subsample = (2,2),W_constraint = exp_const())(norm_3)

		pooled_value =  GlobalAveragePooling2D()(lay3_3)
		y_out = Activation('softmax')(pooled_value)
		model = Model(inputs = [input_l],outputs = [y_out])
		return model

	def get_ioc_constrained():
		input_l = Input(shape=(32, 32, 3,))
		lay1_1 = Convolution2D(96*2, (3, 3), padding = 'same',activation='elu')(input_l)
		lay1_2 = Convolution2D(96, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay1_1)
		norm_1 = BatchNormalization(gamma_constraint = exp_const())(lay1_2)
		lay1_3 = Convolution2D(96, (3, 3), padding = 'same',activation='elu', subsample = (2,2),W_constraint = exp_const())(norm_1)
		lay1 = Dropout(0.5)(lay1_3)


		lay2_1 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay1)
		norm_2_1 = BatchNormalization(gamma_constraint = exp_const())(lay2_1)
		lay2_2 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(norm_2_1)
		lay2_3 = Convolution2D(192, (3, 3), padding = 'same',activation='elu', subsample = (2,2),W_constraint = exp_const())(lay2_2)
		norm_2_2 = BatchNormalization(gamma_constraint = exp_const())(lay2_3)
		lay2 = Dropout(0.5)(norm_2_2)

		lay3_1 = Convolution2D(192, (3, 3), padding = 'same',activation='elu',W_constraint = exp_const())(lay2)
		lay3_2 = Convolution2D(192, (1, 1), padding = 'valid',activation='elu',W_constraint = exp_const())(lay3_1)
		norm_3 = BatchNormalization(gamma_constraint = exp_const())(lay3_2)
		lay3_3 = Convolution2D(10, (1, 1), padding = 'valid',activation='elu', subsample = (2,2),W_constraint = exp_const())(norm_3)

		pooled_value =  GlobalAveragePooling2D()(lay3_3)
		y_out = Activation('softmax')(pooled_value)
		model = Model(inputs = [input_l],outputs = [y_out])
		return model

	def get_nn():
		input_l = Input(shape=(32, 32, 3,))
		lay1_1 = Convolution2D(96, (3, 3), padding = 'same',activation='relu')(input_l)
		lay1_2 = Convolution2D(96, (3, 3), padding = 'same',activation='relu')(lay1_1)
		norm_1 = BatchNormalization()(lay1_2)
		lay1_3 = Convolution2D(96, (3, 3), padding = 'same',activation='relu', subsample = (2,2))(norm_1)
		lay1 = Dropout(0.5)(lay1_3)


		lay2_1 = Convolution2D(192, (3, 3), padding = 'same',activation='relu')(lay1)
		norm_2_1 = BatchNormalization()(lay2_1)
		lay2_2 = Convolution2D(192, (3, 3), padding = 'same',activation='relu')(norm_2_1)
		lay2_3 = Convolution2D(192, (3, 3), padding = 'same',activation='relu', subsample = (2,2))(lay2_2)
		norm_2_2 = BatchNormalization()(lay2_3)
		lay2 = Dropout(0.5)(norm_2_2)

		lay3_1 = Convolution2D(192, (3, 3), padding = 'same',activation='relu')(lay2)
		lay3_2 = Convolution2D(192, (1, 1), padding = 'valid',activation='relu')(lay3_1)
		norm_3 = BatchNormalization()(lay3_2)
		lay3_3 = Convolution2D(10, (1, 1), padding = 'valid',activation='relu', subsample = (2,2))(norm_3)

		pooled_value =  GlobalAveragePooling2D()(lay3_3)
		y_out = Activation('softmax')(pooled_value)
		model = Model(inputs = [input_l],outputs = [y_out])
		return model


img_height = 32
img_width = 32
channel = 3
batch_size = 64
num_classes = 10
epochs = 300
l = 6 
num_filter = 40 
compression = 1.0
dropout_rate = 0.20




def add_denseblock_ioc(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('elu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',W_constraint = exp_const())(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

def add_transition_ioc(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('elu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',W_constraint = exp_const())(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg

def output_layer_ioc(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('elu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax',W_constraint = exp_const())(flat)
    return output


def add_denseblock_constrained(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization(gamma_constraint = exp_const())(temp)
        relu = Activation('elu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same',W_constraint = exp_const())(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

def add_transition_constrained(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization(gamma_constraint = exp_const())(input)
    relu = Activation('elu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same',W_constraint = exp_const())(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg

def output_layer_constrained(input):
    global compression
    BatchNorm = BatchNormalization(gamma_constraint = exp_const())(input)
    relu = Activation('elu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax',W_constraint = exp_const())(flat)
    return output




def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = BatchNormalization()(temp)
        relu = Activation('relu')(BatchNorm)
        Conv2D_3_3 = Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
            Conv2D_3_3 = Dropout(dropout_rate)(Conv2D_3_3)
        concat = Concatenate(axis=-1)([temp,Conv2D_3_3])
        temp = concat
    return temp

def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
        Conv2D_BottleNeck = Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    return avg

def output_layer(input):
    global compression
    BatchNorm = BatchNormalization()(input)
    relu = Activation('relu')(BatchNorm)
    AvgPooling = AveragePooling2D(pool_size=(2,2))(relu)
    flat = Flatten()(AvgPooling)
    output = Dense(num_classes, activation='softmax')(flat)
    return output



def get_ioc():
	input = Input(shape=(img_height, img_width, channel,))
	First_Conv2D = Conv2D(num_filter*2, (3,3), use_bias=False ,padding='same')(input)
	First_Block = add_denseblock_ioc(First_Conv2D, num_filter, dropout_rate)
	First_Transition = add_transition_ioc(First_Block, num_filter, dropout_rate)

	Second_Block = add_denseblock_ioc(First_Transition, num_filter, dropout_rate)
	Second_Transition = add_transition_ioc(Second_Block, num_filter, dropout_rate)

	Third_Block = add_denseblock_ioc(Second_Transition, num_filter, dropout_rate)
	Third_Transition = add_transition_ioc(Third_Block, num_filter, dropout_rate)

	Last_Block = add_denseblock_ioc(Third_Transition,  num_filter, dropout_rate)
	output = output_layer_ioc(Last_Block)

	model = Model(inputs=[input], outputs=[output])
	return model


def get_nn():
	input = Input(shape=(img_height, img_width, channel,))
	First_Conv2D = Conv2D(num_filter*2, (3,3), use_bias=False ,padding='same')(input)
	First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
	First_Transition = add_transition(First_Block, num_filter, dropout_rate)

	Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
	Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

	Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
	Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

	Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
	output = output_layer(Last_Block)
	model = Model(inputs=[input], outputs=[output])
	return model


def get_ioc_constrained():
	input = Input(shape=(img_height, img_width, channel,))
	First_Conv2D = Conv2D(num_filter*2, (3,3), use_bias=False ,padding='same')(input)
	First_Block = add_denseblock_constrained(First_Conv2D, num_filter, dropout_rate)
	First_Transition = add_transition_constrained(First_Block, num_filter, dropout_rate)

	Second_Block = add_denseblock_constrained(First_Transition, num_filter, dropout_rate)
	Second_Transition = add_transition_constrained(Second_Block, num_filter, dropout_rate)

	Third_Block = add_denseblock_constrained(Second_Transition, num_filter, dropout_rate)
	Third_Transition = add_transition_constrained(Third_Block, num_filter, dropout_rate)

	Last_Block = add_denseblock_constrained(Third_Transition,  num_filter, dropout_rate)
	output = output_layer_constrained(Last_Block)
	model = Model(inputs=[input], outputs=[output])
	return model

