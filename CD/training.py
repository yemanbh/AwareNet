# -*- coding: utf-8 -*-
"""
Created on 17/03/2020

@author: yhagos
"""
# fundamental libraries
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

# output
import csv
import json

# deep learning libraries
import tensorflow as tf
tf.keras.backend.clear_session()


class UnetModels(object):
	
	def __init__(self,
				 optimizer_type='adam',
				 learning_rate=1e-4,
				 base=32,
				 epochs=500,
				 input_shape=256,
				 depth=4
				 ):
		self.optimizer_type = optimizer_type
		self.base = base
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.depth = depth
		self.input_shape = input_shape
	
	@staticmethod
	def inception_block(input_tensor, num_filters):
		
		p1 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
		p1 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p1)
		
		p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(input_tensor)
		p2 = tf.keras.layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(p2)
		
		p3 = tf.keras.layers.Conv2D(num_filters, (1, 1), padding='same', activation='relu')(input_tensor)
		
		# return tf.keras.layers.concatenate([p1, p2, p3], axis=3)
		o = tf.keras.layers.Add()([p1, p2, p3])
		
		return o
	
	def get_inception_backend_unet(self, input_tensor):
		# input_shape = (self.patch_size, self.patch_size, self.CH)
		en = self.inception_block(input_tensor, self.base)
		en = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(en)
		features = self.base
		for i in range(2, self.depth):
			features = 2 * features
			en = self.inception_block(en, features)
			en = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(en)
		
		features = 2 * features
		
		en = self.inception_block(en, features)
		encoder_model = tf.keras.Model(inputs=[input_tensor], outputs=[en])
		
		# DECODER
		features = int(features / 2)
		d = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(features, (2, 2), strides=(2, 2),
																		 padding='same')(
			encoder_model.layers[-1].output), encoder_model.layers[-8].output], axis=3)
		d = self.inception_block(d, features)
		
		a = 2
		for j in range(2, self.depth):
			features = int(features / 2)
			d = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(features, (2, 2), strides=(2, 2),
																			 padding='same')(d),
											 encoder_model.layers[-7 * a - 1].output], axis=3)
			d = self.inception_block(d, features)
			
			a += 1
		
		d = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(d)
		
		return d
	
	def get_model(self):
		input_tensor = tf.keras.layers.Input(shape=self.input_shape)
		out = self.get_inception_backend_unet(input_tensor)
		return tf.keras.models.Model(inputs=[input_tensor], outputs=[out])


def weighted_dice_loss(p, g, w=None, eps=1):
	
	p_and_g = tf.multiply(p, g)
	p_union_g = tf.add(p, g)
	
	if w is None:
		# print('no weighted applied')
		
		dice_overlap = 2 * tf.reduce_sum(p_and_g) / (eps + tf.reduce_sum(p_union_g))
	else:
		num = 2 * tf.reduce_sum(tf.multiply(w, p_and_g))
		
		denom = eps + tf.reduce_sum(tf.multiply(w, p_union_g))
		
		dice_overlap = num / denom
	
	dice_overlap_loss = 1 - dice_overlap
	
	return dice_overlap_loss

def create_log_csv_file(training_history):
	with open(training_history, 'w', newline='') as f:
		w = csv.writer(f)
		w.writerow(['epoch', 'time(m)', 'train_loss', 'val_loss'])


def update_log_file(training_history, row_value):
	with open(training_history, 'a', newline='') as f:
		w = csv.writer(f)
		w.writerow(row_value)


def plot_training_history(save_dir, log_file):
	
	train_hist_df = pd.read_csv(log_file)
	train_hist_df.drop(columns=['time(m)'], inplace=True)
	x = train_hist_df.columns.values[0]
	y = train_hist_df.columns.values[1:]
	train_hist_df.plot(x=x, y=y)
	plt.tight_layout()
	plt.savefig(os.path.join(save_dir, 'training_history.pdf'))

def weight_mapping(mapping_func, w_values, weight_matrix):
	weight_matrix_new = weight_matrix.copy()
	
	# value change mapping
	for k, v in w_values.items():
		weight_matrix_new[np.where(weight_matrix_new == int(k))] = v
		
	# due to precision of interpolation the values might not be exact much , and make these pixels background
	weight_matrix_new[~np.isin(weight_matrix_new, list(w_values.values()))] = 1
	
	if mapping_func == 'linear':
		return weight_matrix_new.astype('float32')
	elif mapping_func == 'exp-x2':
		return np.exp(-(weight_matrix_new.astype('float32') ** -2))
	elif mapping_func == 'exp-x':
		return np.exp(-(weight_matrix_new.astype('float32') ** -1))
	else:
		raise Exception('uknown method, {}'.format(mapping_func))

def run(train_data_dir,
		val_data_dir,
		output_dir,
		input_shape,
		ntk_depth,
		w_name=None,
		w_values=None,
		mapping_func=None,
		weighted=True,
		configuration_num=1,
		learning_rate=1e-4,
		init_num_neuron=16,
		batch_size=32,
		patience=50,
		epochs=400,
		all_params=None):

	
	model_params_hash = [
						 str(batch_size),
						 str(init_num_neuron),
						 str(learning_rate),
						 str(ntk_depth),
						 str(configuration_num)]
	
	if weighted is True:
		model_params_hash = ['weighted', mapping_func, w_name] + model_params_hash
	else:
		model_params_hash.insert(0, 'normal')
	
	output_dir = os.path.join(output_dir, '_'.join(model_params_hash))
	print('output directory:{}'.format(output_dir))
	
	if os.path.exists(output_dir) is False:
		os.makedirs(output_dir, exist_ok=True)
	else:
		print('The output folder already exists and continue to the next configuration')
		return
	
	if all_params is not None:
		with open(os.path.join(output_dir, 'config.json'), 'w') as j_file:
			json.dump(all_params, j_file, indent=5)
	
	# training story logging file
	training_history = os.path.join(output_dir, 'train_hist.csv')
	
	# load training data
	train_npz = np.load(train_data_dir)
	train_img = train_npz['images']
	train_weight = train_npz['weights']
	train_label = train_npz['labels']
	
	# transform weight
	if weighted is True:
		train_weight = weight_mapping(mapping_func, w_values, train_weight)
	else:
		pass
	
	# change dimension of the arrays to 4-d and
	# change data types in to float32 for tensorlow operations
	if train_img.ndim == 3:
		train_img = np.expand_dims(train_img, axis=-1)
	train_img = (train_img * 1.0 / 255).astype('float32')
	
	if train_weight.ndim == 3:
		train_weight = np.expand_dims(train_weight, axis=-1)
	train_weight = train_weight.astype('float32')
	
	if train_label.ndim == 3:
		train_label = np.expand_dims(train_label, axis=-1)
	train_label = (train_label > 200).astype('float32')
	
	# load training data
	val_npz = np.load(val_data_dir)
	val_img = val_npz['images']
	val_weight = val_npz['weights']
	val_label = val_npz['labels']
	
	# transform weight
	if weighted is True:
		val_weight = weight_mapping(mapping_func, w_values, val_weight)
	else:
		pass
	
	# change dimension of the arrays to 4-d and
	# change data types in to float32 for tensorlow operations
	if val_img.ndim == 3:
		val_img = np.expand_dims(val_img, axis=-1)
	val_img = (val_img * 1.0 / 255).astype('float32')
	
	if val_weight.ndim == 3:
		val_weight = np.expand_dims(val_weight, axis=-1)
	val_weight = val_weight.astype('float32')
	
	if val_label.ndim == 3:
		val_label = np.expand_dims(val_label, axis=-1)
	val_label = (val_label > 200).astype('float32')
	
	# prepare training data
	n_train = train_img.shape[0]
	train_data = tf.data.Dataset.from_tensor_slices((train_img, train_label, train_weight))
	train_data = train_data.shuffle(buffer_size=n_train).batch(batch_size=batch_size)
	
	# prepare validation data
	# n_val = val_img.shape[0]
	val_data = tf.data.Dataset.from_tensor_slices((val_img, val_label, val_weight))
	val_data = val_data.batch(batch_size=batch_size)
	
	# cnn architecture and params
	model_params = dict(
		learning_rate=learning_rate,
		base=init_num_neuron,
		epochs=epochs,
		input_shape=input_shape,
		depth=ntk_depth
	)
	model_obj = UnetModels(**model_params)
	model = model_obj.get_model()
	model.summary()
	
	optimizer = tf.keras.optimizers.Adam(learning_rate)
	# empty log file/csv file
	create_log_csv_file(training_history)
	
	counter = 0
	min_val_loss = np.infty  # initial minimum validation loss
	
	train_iter = int(np.ceil(n_train / batch_size))
	
	@tf.function
	def compute_loss(pred, label, weight=None):
		print('computing dice loss')
		loss = weighted_dice_loss(p=pred, g=label, w=weight)
		return loss
	
	@tf.function
	def train_step(model, image, label, weight=None):
		with tf.GradientTape() as tape:
			# Run the forward pass of the layer.
			# The operations that the layer applies
			# to its inputs are going to be recorded
			# on the GradientTape.
			pred = model(image, training=True)  # pred for this minibatch
			
			# Compute the loss value for this minibatch.
			train_loss = compute_loss(pred=pred, label=label, weight=weight)
		
		# Use the gradient tape to automatically retrieve
		# the gradients of the trainable variables with respect to the loss.
		grads = tape.gradient(train_loss, model.trainable_weights)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		
		return train_loss
	
	@tf.function
	def val_step(model, image, label, weight=None):
		pred = model(image, training=False)
		
		# compute loss
		val_loss = compute_loss(pred=pred, label=label, weight=weight)
		
		return val_loss
	
	for epoch in range(epochs):
		
		# reset validation and training loss at the start of every epoch
		train_loss = []
		val_loss = []
		start_time = time.time()
		
		# train model on training data
		for step, (train_batch_image, train_batch_label, train_batch_weight) in enumerate(train_data):
			
			if weighted is False:
				train_batch_weight = None
			
			loss = train_step(model,
							  train_batch_image,
							  train_batch_label,
							  weight=train_batch_weight)
			train_loss.append(loss)
			
			# Log every 10 batches.
			if step % 10 == 0:
				print('iter:{}/{}, epochs: {}, training dice loss:{}'.format(step,
																		   train_iter,
																		   epoch,
																		   np.mean(train_loss)
																		   ))
		
		# run validation loop at the end of every epoch
		for step, (val_batch_image, val_batch_label, val_batch_weight) in enumerate(val_data):
			if weighted is False:
				val_batch_weight = None
			loss = val_step(model,
							val_batch_image,
							val_batch_label,
							weight=val_batch_weight)
			val_loss.append(loss)
		
		# end of an epoch
		end_time = time.time()
		t = np.round((end_time - start_time) * 1.0 / 60, decimals=3)
		print('*' * 50)
		print('epochs: {}, validation dice loss:{}, time:{}m'.format(epoch,
																   np.mean(val_loss),
																   t))
		print('*' * 50)
		
		# epoch evaluation and update log file
		new_eval_list = [np.mean(train_loss), np.mean(val_loss)]
		update_log_file(training_history, [epoch, t] + new_eval_list)
		
		if np.mean(val_loss) < min_val_loss:
			
			# save model
			print('*' * 60)
			print('evaluation on validation data')
			print('Validation loss improved from {} to {}'.format(min_val_loss,
			                                                      np.mean(val_loss)))
			print('*' * 60)
			
			model.save(os.path.join(output_dir, 'best_model.h5'))
			
			counter = 0
			
			min_val_loss = np.mean(val_loss)  # update minimum loss
		
		else:
			print('validation loss did not improve')
			counter += 1
		
		if counter >= patience:
			break
		
	plot_training_history(save_dir=output_dir, log_file=training_history)
	
	# remove variable and objects
	del optimizer
	del model_obj
	del model


if __name__ == '__main__':
	model_params = dict(
		learning_rate=1e-4,
		base=32,
		epochs=10,
		input_shape=(256, 256, 3),
		depth=6
	)
	model_obj = UnetModels(**model_params)
	model = model_obj.get_model()
	model.summary()
