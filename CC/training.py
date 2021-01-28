import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

import classifiermodels


class TrainClassifierModel(object):

    def __init__(self,
                 output_dir,
                 training_data_dir,
                 val_data_dir,
                 learning_rate,
                 backend='vgg',
                 base=16,
                 patch_size=(32, 32, 3),
                 depth=3,
                 patience=30,
                 batch_size=32,
                 ratio_weight=None,
                 mapping_func=None,
                 epochs=1000):

        self.output_dir = output_dir
        self.training_data_dir = training_data_dir
        self.val_data_dir = val_data_dir
        self.patch_size = patch_size
        self.W = patch_size[1]
        self.H = patch_size[0]
        self.channels = patch_size[2]
        self.batch_size = batch_size
        self.epochs = epochs
        self.depth = depth
        self.learning_rate = learning_rate
        self.base = base
        self.backend = backend
        self.patience = patience
        self.ratio_weight = ratio_weight
        self.mapping_func = mapping_func
    
    @staticmethod
    def plot_training_history(save_dir, log_file):

        train_hist_df = pd.read_csv(log_file)
        # train_hist_df.set_index("epochs", inplace=True)
        train_hist_df.plot(x='epoch', y=train_hist_df.columns.values[1:])
        min_loss = np.min(train_hist_df['val_loss'])
        x = train_hist_df['epoch'].mean()
        y = train_hist_df['val_loss'].mean()
        plt.text(x, y, 'min val loss = {:.3f}'.format(min_loss))
        plt.savefig(os.path.join(save_dir, 'training_history.pdf'), dpi=800)

    def get_generators(self):

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           zoom_range=0.25)

        val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.training_data_dir,
                target_size=(self.H, self.W),
                batch_size=self.batch_size,
                class_mode='categorical')

        validation_generator = val_generator.flow_from_directory(
                                        self.val_data_dir,
                                        target_size=(self.H, self.W),
                                        batch_size=self.batch_size,
                                        class_mode='categorical')
        return validation_generator, train_generator

    def compute_val_train_iterations(self):
        N = 0
        for f in os.listdir(self.training_data_dir):

            N += len(os.listdir(os.path.join(self.training_data_dir, f)))

        train_iter = int(np.ceil(N/self.batch_size))

        M = 0
        for f in os.listdir(self.val_data_dir):

            M += len(os.listdir(os.path.join(self.val_data_dir, f)))

        val_iter = int(np.ceil(M/self.batch_size))

        print("**"*30)
        print('validation iteration :{}'.format(val_iter))
        print('training iteration :{}'.format(train_iter))
        print("**"*30)

        return val_iter, train_iter
        
    def run(self, all_params):
        
        model_hash = [self.mapping_func, self.backend, str(self.batch_size), str(self.base),
                      str(self.depth), str(self.patch_size[0]), str(self.learning_rate)]
        
        output_dir = os.path.join(self.output_dir, '_'.join(model_hash))
        if os.path.exists(output_dir):
            print('{} already exists'.format(output_dir))
            return
        else:
            os.makedirs(output_dir, exist_ok=True)
    
        with open(os.path.join(output_dir, 'config.json'), 'w') as j_file:
            json.dump(all_params, fp=j_file, indent=4)

        val_iter, train_iter = self.compute_val_train_iterations()

        params = dict(depth=self.depth,
                      learning_rate=self.learning_rate,
                      base=self.base,
                      backend=self.backend,
                      patch_size=self.patch_size,
                      num_classes=len(os.listdir(self.training_data_dir))
                      )

        model = classifiermodels.get_model(**params)
        model.summary()

        log_filename = os.path.join(output_dir, 'train_hist.csv')
        csv_log = tf.keras.callbacks.CSVLogger(log_filename,
                                               separator=',',
                                               append=True)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                          patience=self.patience,
                                                          verbose=0,
                                                          mode='min')
        checkpoint_filepath = os.path.join(output_dir, 'best_model.h5')

        checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_filepath,
                                                        monitor='val_loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='min')

        callbacks_list = [csv_log, early_stopping, checkpoint]
        validation_generator, train_generator = self.get_generators()
        if all([self.mapping_func is not None, self.ratio_weight is not None]):
            # get weight
            class_weight = self.weight_mapping(self.mapping_func, self.ratio_weight)
        else:
            class_weight = None
        print(class_weight)
        model.fit_generator(
            train_generator,
            steps_per_epoch=train_iter,
            epochs=self.epochs,
            callbacks=callbacks_list,
            validation_data=validation_generator,
            validation_steps=val_iter,
            class_weight=class_weight,
            shuffle=True)

        self.plot_training_history(output_dir, log_filename)


    def weight_mapping(self, mapping_func, class_2_w_mapping):
        """
        implementation of different weighting strategies
        :param mapping_func: str, possiblible values
        :param original_w_values: dict of weight of cell classes
        :return:
        """
        assert mapping_func in ['linear', 'exp-x', 'exp-x2'], f"{mapping_func} not in ['linear', 'exp-x', 'exp-x2']"
        assert isinstance(class_2_w_mapping, dict)
        mapped_class_2_w_mapping = class_2_w_mapping.copy()
        print(f"mapping function:{mapping_func}, original weight:{class_2_w_mapping}")
        if mapping_func == 'linear':
            for key, val in class_2_w_mapping.items():
                mapped_class_2_w_mapping[key] = tf.cast(val, 'float32')
        elif mapping_func == 'exp-x2':
            for key, val in class_2_w_mapping.items():
                mapped_class_2_w_mapping[key] = tf.exp(-(tf.cast(val, 'float32') ** -2))
            # return np.exp(-(weight_matrix_new.astype('float32') ** -2))
        elif mapping_func == 'exp-x':
            for key, val in class_2_w_mapping.items():
                mapped_class_2_w_mapping[key] = tf.exp(-(tf.cast(val, 'float32') ** -1))
        else:
            raise Exception('uknown method, {}'.format(mapping_func))
        print(f"mapped weight:{mapped_class_2_w_mapping}")
        return mapped_class_2_w_mapping

if __name__ == '__main__':
    pass
