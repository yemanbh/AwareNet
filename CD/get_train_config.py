# -*- coding: utf-8 -*-
"""
Created on 07/04/2020

@author: yhagos
"""
import os
import argparse


def training_configs():
	args = get_parsed_arguments()
	data_dir = args.data_dir
	output_dir = args.output_dir
	params_dict = dict()
	common_params = dict(
		train_data_dir=os.path.join(data_dir, 'train.npz'),
		val_data_dir=os.path.join(data_dir, 'val.npz'),
		output_dir=output_dir,
		input_shape=(256, 256, 3),
		epochs=300
	)
	weighted_loss_values = [True]
	learning_rates = [1e-4, 1e-5]
	init_num_neurons = [16]
	batch_sizes = [16]
	ntk_depths = [3, 4, 5]
	configuration_num = 1
	
	# weight parameters
	weights = get_weight_mapping()
	weight_mapping_funcs = ['linear', 'exp-x2', 'exp-x']
	for w_name, w_values in weights.items():
		for mapping_func in weight_mapping_funcs:
			for learning_rate in learning_rates:
				for batch_size in batch_sizes:
					for init_num_neuron in init_num_neurons:
						for weighted in weighted_loss_values:
							for ntk_depth in ntk_depths:
								param = dict()
								param['learning_rate'] = learning_rate
								param['batch_size'] = batch_size
								param['init_num_neuron'] = init_num_neuron
								param['configuration_num'] = configuration_num
								param['weighted'] = weighted
								param['ntk_depth'] = ntk_depth
								param['mapping_func'] = mapping_func  # s[int(args.config_id) - 1]
								param['w_name'] = w_name
								param['w_values'] = w_values
								
								params_dict[str(configuration_num)] = {**common_params, **param}
								
								configuration_num += 1
	
	return params_dict, args


def get_parsed_arguments():

	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data_dir', dest='data_dir', help='directory of training images and csv file')
	parser.add_argument('-o', '--output', dest='output_dir', help='directory to save training output')
	parser.add_argument('-m', '--cnn', dest='cell_detection_model', help='path to cell detection model')
	parser.add_argument('-n', '--config_id', dest='config_id', help='path to cell detection model')
	parser.add_argument('-c', '--cluster', dest='cluster', help='flag to check if running on server or local',
						action='store_true', default=False)
	args = parser.parse_args()
	
	return args

def get_weight_mapping():
	weight = dict(
		linear1={'1': 1, '2': 2, '3': 9}
	)
	return weight
