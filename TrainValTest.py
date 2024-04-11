import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from matplotlib import pyplot as plt
import cv2
from pprint import pprint

from datetime import datetime
import sys
import os
import glob
from tensorflow.keras.losses import *
import numpy as np
from pprint import pprint
from knockknock import teams_sender

class TrainValTest:
    '''
    This class hepls in training.

    Args ::
        model -- tf.keras.nn.Model        
    '''
    def __init__(self, model, model_name=None) -> None:
        self.model = model
        self.OPTIMIZER_DICT = {
                    "Adadelta" : tf.keras.optimizers.Adadelta, 
                    "Adagrad" : tf.keras.optimizers.Adagrad, 
                    "Adam" : tf.keras.optimizers.Adam, 
                    "Adamax" : tf.keras.optimizers.Adamax, 
                    "Ftrl" : tf.keras.optimizers.Ftrl, 
                    "Nadam" : tf.keras.optimizers.Nadam, 
                    "SGD" : tf.keras.optimizers.SGD 
        }
        self.OPTIMIZER_ARG = {
                            "Adadelta" : {'rho': 0.95, 'epsilon' : 1e-07, 'name' : 'Adadelta'}, 
                            "Adagrad" : tf.keras.optimizers.Adagrad, 
                            "Adam" : {'beta_1' : 0.9, 'beta_2' : 0.999, 'epsilon' : 1e-07, 'amsgrad' : False, 'name' : 'Adam'},
                            "Adamax" : tf.keras.optimizers.Adamax, 
                            "Ftrl" : tf.keras.optimizers.Ftrl, 
                            "Nadam" : tf.keras.optimizers.Nadam, 
                            "SGD" : tf.keras.optimizers.SGD 
        }
        self.is_compiled = False
        self.is_implemented_train_iter = True
        self.is_implemented_val_iter = True
        self.model_name = model_name
        
        ## allow gpu growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        gpus = ['gpu:'+gpu.name[-1] for gpu in gpus]
        print(f'GPUs : {gpus}')
        
        ## to be used in train_iter
        self.train_model = self.model

    def get_optimizer_arg(self, optimizer_name):
        '''
        Returns all the optimizer arguments
        Arguments --
            optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
        '''
        return self.OPTIMIZER_ARG[optimizer_name]
    
    def create_optimizer(self, optimizer_fnc, lr, other_params = {}):
        '''
        Creates one of the optimizers present in tf.keras.optimizers and returns it.

        Args --
            optimizer_func -- function | optimizer's creation function
            lr -- float or float tensor or learning_rate_function | learning rate
            other_params -- dict | default {} | contains all the arguments needed to create the optimizer
        Return --
            other_params -- tf.keras.optimizers.*
        '''
        other_params = other_params.values()
        optimizer = optimizer_fnc(lr) #, *other_params)

        return optimizer
    
    def compile(self, lr = 0.0001, optimizer_name = 'Adam', eval_metrics = {}, train_step = None, test_step = None, compute_cost = None):
        '''
        Prepares the model for training and returns True on success.

        Args ::            
            lr -- float or float tensor or learning_rate_function | learning rate
            optimizer_name -- string | one of ["Adadelta", "Adagrad", "Adam", "Adamax", "Ftrl", "Nadam", "SGD"]
            eval_metrics -- python dict | default {} | dict of metrics to be evaluated |
                                            the dict should contain eval step function as values and 
                                            metric name as keys
            train_step -- function | train for 1 iteration | takes 6 arguments 
                                            minibatch_X -- feature minibatch
                                            minibatch_Y -- label minibatch
                                            model -- CV model
                                            compute_cost -- cost function
                                            optimizer -- optimizer
            test_step -- function | test for 1 iteration | takes 5 arguments 
                                            minibatch_X -- feature minibatch
                                            minibatch_Y -- label minibatch
                                            model -- CV model
                                            compute_cost -- cost function
            compute_cost - function | calculates loss | takes 3 arguments
                                            Y - gt labels
                                            Y_pred - predicted labels
                                            vgg -- vgg model if use_vgg is true
        '''
        if train_step is not None:
            self.train_step = train_step
        if test_step is not None:
            self.test_step = test_step
        if compute_cost is not None:
            self.compute_cost = compute_cost

        ## creating optimizer
        self.optimizer_name = optimizer_name
        optimizer_arg_default = self.OPTIMIZER_ARG[optimizer_name] if optimizer_name in self.OPTIMIZER_ARG.keys() else {}

        optimizer_arg = self.get_optimizer_arg(optimizer_name)
        pprint(optimizer_arg)

        for k, v in optimizer_arg.items():
            if k in optimizer_arg_default.keys():
                optimizer_arg_default[k] = v
        
        self.optimizer_arg = optimizer_arg_default
    
        if optimizer_name not in self.OPTIMIZER_DICT.keys():
            print(f'Invalid optimizer option')
            print(f'Optimizer should be one of  : {self.OPTIMIZER_DICT.keys()}')


        self.lr = lr
        self.optimizer = self.create_optimizer(self.OPTIMIZER_DICT[optimizer_name], lr, optimizer_arg)
        self.eval_metrics = eval_metrics
        self.is_compiled = True

    def load_checkpoint(self, checkpoint_path, optimizer = False):
        '''
        checkpoint_path - str | default '' | specifies the saved checkpoint path to restore
                                            the model from
        '''
        checkpoint = None
        if self.is_compiled and optimizer:
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        else:
            checkpoint = tf.train.Checkpoint(net=self.model)
        if checkpoint_path != '':
            if os.path.exists(checkpoint_path):
                print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    def load_checkpoint_train(self, checkpoint_path, optimizer = True):
        '''
        checkpoint_path - str | default '' | specifies the saved checkpoint path to restore
                                            the model from
        '''
        checkpoint = None
        if self.is_compiled and optimizer:
            checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        else:
            checkpoint = tf.train.Checkpoint(net=self.model)
        if checkpoint_path != '':
            if os.path.exists(checkpoint_path):
                print(f'Restoring checkpoint from {checkpoint_path}', end='\r')
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()
    
    def load_h5(self, h5_path):
        self.model.load_weights(h5_path)

    def train_iter(self, data_dict,):
        '''
        This function is what's executed every train epoch
        '''
        minibatch_X = data_dict['images']
        minibatch_Y = data_dict['labels']

        ## train the model for one iteration
        temp_cost, Y_pred = self.train_step(minibatch_X, minibatch_Y)            

        return temp_cost

    def val_iter(self, data_dict, eval_only):
        '''
        This function is what's executed every val epoch
        '''
        iter_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}
        minibatch_X = data_dict['images']
        minibatch_Y = data_dict['labels']

        ## calculate cost and Y_pred
        temp_cost = None
        Y_pred = None
        if not eval_only:
            temp_cost, Y_pred = self.test_step(minibatch_X, minibatch_Y)
        else:
            Y_pred = self.forward_train(minibatch_X)
                        
        ## calculate metrics
        for kmetric, vmetric in self.eval_metrics.items():
            vmetric.update_state(Y_pred, minibatch_Y)
        
        return temp_cost

    def forward_train(self, X):
        Y_pred = self.train_model(X)
        return Y_pred

    def train_step(self, X, Y):
        '''
        Train one minibatch
        '''
        with tf.GradientTape() as tape:
            Y_pred = self.forward_train(X)
            cost = self.compute_cost(Y, Y_pred)    
        gradient = tape.gradient(cost, self.train_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.train_model.trainable_variables))

        return cost, Y_pred

    def test_step(self, X, Y):
        '''
        Infer one minibatch
        '''
        Y_pred = self.forward_train(X)
        cost = self.compute_cost(Y, Y_pred)

        return cost, Y_pred
 
    def infer_step(self, input_image):
        '''
        Infer one sample 
        '''
        print(f'[INFO] self.infer_step() not overridden!!')
        return None

    def post_process(self, output):
        '''
        Infer one sample 
        '''
        print(f'[INFO] self.post_process(output) not overridden!!')
        return None

    def normalise(self, image):
        print(f'[INFO] using default normalisation, overide self.normalise(image) to use different normalisation')
        return image / 255.

    def pre_process(self, image):
        print(f'[INFO] using default pre_process, overide self.pre_process(image) to use different normalisation')
        return self.normalise(image)[None, :, :, :]
    
    def read_RGB(self, image_path):
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
        return input_image
    
    def save_raw(self, output_path, output):
        if isinstance(output, np.ndarray):
            output.tofile(output_path)
        elif isinstance(output, list):
            if os.path.exists(output_path):
                os.system(f'rm -rf {output_path}')
            os.makedirs(output_path)
            for i in range(len(output)):
                if isinstance(output[i], np.ndarray):
                    output[i].tofile(os.path.join(image_path, f'{i}', img))
                else:
                    output[i].numpy().tofile(os.path.join(image_path, f'{i}', img))
        else:
            output.numpy().tofile(output_path)

    def save_RGB(self, image_path, image):
        if len(image.shape) == 4:
            if image.shape[0] == 1:
                image = image[0]
                cv2.imwrite(image_path, image)
            else:
                if os.path.exists(image_path):
                    os.system(f'rm -rf {image_path}')
                os.makedirs(image_path)
                for i in range(image.shape[0]):
                    img = image[i]                    
                    cv2.imwrite(os.path.join(image_path, f'{i}.png', img))
        elif len(image.shape) == 3:
            cv2.imwrite(image_path, image)
        elif len(image.shape) == 2:
            cv2.imwrite(image_path, image)

    def read_and_normalise_RGB(self, image_path):
        return self.normalise(self.read_RGB(image_path))

    def read_and_preprocess_RGB(self, image_path):
        return self.read_and_preprocess(self.read_RGB(image_path))

    def train(self, log_dir, ex_name, train_dataset, dev_dataset, batch_size, epochs,                             
                    callbacks = [], steps_per_epoch = None, validation_steps = None,
                    start_epoch=0, dataset_name = '', loss_name = '',
                    model_name = '', show_results=-1, teams_webhook_url=''):
        '''
        This function gets executed on executing the script.
        
        Args ::
            log_dir -- str | path where to write logs
            ex_name -- str | name of the experiment
            train_dataset -- Training set
            dev_dataset -- Dev set            
            batch_size -- int | number of batches used
            epochs -- int or int tensor | number of epochs            
            callbacks - list | default [] | list of callbacks. keras.callbacks.Callback instances
            steps_per_epoch - int | default None | when given, runs given number of iterations per epoch
            validation_steps - int | default None | when given, runs given number of val iterations per epoch
            start_epoch -- int | default 0 | starting epoch            
            dataset_name -- str | default '' | name of the dataset used in trainiing
            loss_name -- str | default '' | name of the loss(es) used in trainiing
            model_name -- str | default '' | name of the model used in trainiing
            show_results -- int | default -1 | if set between 0 to epochs; computes
                                        metrics and displayes results from dev set
                                        in that intervals
            teams_webhook_url -- str | uri containing ms teams link
        '''  
        @teams_sender(webhook_url=teams_webhook_url) 
        # @email_sender(recipient_emails=["<your_email@address.com>", "<your_second_email@address.com>"], sender_email="<grandma's_email@gmail.com>")
        # @email_sender(recipient_emails=mail_list, sender_email="<grandma's_email@gmail.com>")
        def _fit(train_dataset, dev_dataset, batch_size, epochs,                            
                    callbacks = [], steps_per_epoch = None, validation_steps = None,
                    start_epoch=0, dataset_name = '', loss_name = '',
                    model_name = '', show_results=-1):
            '''
            This function gets executed on executing the script.
            
            Args ::
                train_dataset -- Training set
                dev_dataset -- Dev set            
                batch_size -- int | number of batches used
                epochs -- int or int tensor | number of epochs            
                callbacks - list | default [] | list of callbacks. keras.callbacks.Callback instances
                steps_per_epoch - int | default None | when given, runs given number of iterations per epoch
                validation_steps - int | default None | when given, runs given number of val iterations per epoch
                start_epoch -- int | default 0 | starting epoch            
                dataset_name -- str | default '' | name of the dataset used in trainiing
                loss_name -- str | default '' | name of the loss(es) used in trainiing
                model_name -- str | default '' | name of the model used in trainiing
                show_results -- int | default -1 | if set between 0 to epochs; computes
                                            metrics and displayes results from dev set
                                            in that intervals
            '''    
            if not self.is_compiled:
                print('[INFO] Please call TrainInfer.compile() before calling TrainInfer.train()')
                return
            if not self.is_implemented_train_iter:
                print('[INFO] Please override TrainInfer.train_iter before calling TrainInfer.train()')
                return
            if not self.is_implemented_val_iter:
                print('[INFO] Please override TrainInfer.val_iter before calling TrainInfer.train()')
                return
            ## hyperparameters
            lr = self.lr
            PARAMS = {
                    'epochs' : epochs,
                    'steps_per_epoch' : steps_per_epoch,
                    'validation_steps' : validation_steps,
                    'start_epoch' : start_epoch,
                    'start-lr' : lr,
                    'batch-size' : batch_size,
                    'dataset-name' : dataset_name,
                    'loss-name' : loss_name,
                    'model-name' : model_name,
                    'optimizer_name' : self.optimizer_name,
                    'ex_name' : ex_name
            }

            print(f'Log Dir - {log_dir}')
            print('Hyperparameters - ')
            pprint(PARAMS)

            with open(f'{os.path.join(log_dir, ex_name)}.log', 'w') as f:
                f.write('HYPER-PARAMETERS\n')
                f.write('----------------------\n')
                for key, value in PARAMS.items(): 
                    f.write('%s:%s\n' % (key, value))
                f.write('\n')

            ## initialize data loader
            n_minibatches = train_dataset.count_minibatches()
            n_minibatches_dev = dev_dataset.count_minibatches() if dev_dataset is not None else 0

            print(f'Total number of training examples = {train_dataset.m}')         
            print(f'Start epoch - {start_epoch} | End epoch - {start_epoch + epochs}')
            print(f'Number of minibatches in training set - {n_minibatches}')
            print('Starting training...')

            logs = {}
            if steps_per_epoch is None or steps_per_epoch == 0:
                steps_per_epoch = n_minibatches
            elif steps_per_epoch < 0:
                print(f'[INFO] steps_per_epoch can not be -ve, but found {steps_per_epoch}')  
                return
            if validation_steps is None or validation_steps == 0:
                validation_steps = n_minibatches_dev  
            elif validation_steps < 0:
                print(f'[INFO] validation_steps can not be -ve, but found {validation_steps}')  
                return
            costs = []
            dev_costs = []
            dev_metric = {f'val_{k}' : [] for k, v in self.eval_metrics.items()}
            for epoch in range(start_epoch, start_epoch+epochs):                
                minibatch_cost = 0
                dev_minibatch_cost = 0
                dev_minibatch_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}
                dev_epoch_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}

                ## iterate over minibatches
                for iteration in range(steps_per_epoch):
                    step = (iteration + 1) + (epoch * steps_per_epoch)

                    ## fetch one minibatch
                    data_dict = train_dataset.get_data()            

                    temp_cost = self.train_iter(data_dict)

                    if isinstance(temp_cost, tuple) or isinstance(temp_cost, list):
                        if iteration == 0:
                            minibatch_cost = []
                            for tc in temp_cost:
                                minibatch_cost.append([tc])
                        else:
                            for i in range(len(temp_cost)):
                                minibatch_cost[i] += temp_cost[i]
                        if iteration > 0:
                            sys.stdout.write("\033[K")
                        op_string = 'costs - '
                        for i in range(len(temp_cost)):
                            op_string += f'{i} = {temp_cost[i]} |'
                        print(f'minibatches - {iteration + 1}/{steps_per_epoch} | {step} iter | {op_string}', end='\r')
                    elif isinstance(temp_cost, dict):
                        if iteration == 0:
                            minibatch_cost = {}
                            for key, value in temp_cost.items():
                                minibatch_cost[key] = value
                        else:
                            for key, value in temp_cost.items():
                                minibatch_cost[key] += value
                        if iteration > 0:
                            sys.stdout.write("\033[K")
                        op_string = ''
                        for key, value in temp_cost.items():
                            op_string += f'{key} = {value} |'
                        print(f'minibatches - {iteration + 1}/{steps_per_epoch} | {step} iter | {op_string}', end='\r')
                    else:    
                        minibatch_cost += temp_cost            
                        if iteration > 0:
                            sys.stdout.write("\033[K")
                        print(f'{iteration + 1}/{steps_per_epoch} minibatches processed | {step} iterations | cost - {temp_cost}', end='\r')
                    
                    step_lr = lr(step) if not isinstance(lr, float) else lr            

                    ## update model in callbacks
                    for callback in callbacks:
                        callback.model = self.model   
                    

                ## track cost
                if isinstance(minibatch_cost, tuple) or isinstance(minibatch_cost, list):
                    if len(costs) == 0:
                        for mc in minibatch_cost:
                            costs.append([mc])
                    else:
                        for i in range(len(minibatch_cost)):
                            costs[i].append(minibatch_cost[i])
                    sys.stdout.write("\033[K")
                    print(f'Training set cost after {epoch} epochs = ')
                    for i in range(len(minibatch_cost)):
                        print(f'{i} = {minibatch_cost[i]}')
                elif isinstance(minibatch_cost, dict):
                    if not isinstance(costs, dict):
                        costs = {}
                        for key, value in minibatch_cost.items():
                            costs[key] = [value]
                    else:
                        for key, value in minibatch_cost.items():
                            costs[key].append(value)
                    sys.stdout.write("\033[K")
                    print(f'Training set cost after {epoch} epochs = ')
                    for key, value in minibatch_cost.items():
                        print(f'{key} = {value}')
                else:    
                    costs.append(minibatch_cost) # /len(minibatch_cost))
                    minibatch_cost = 0
                    sys.stdout.write("\033[K")
                    print(f'Training set cost after {epoch} epochs =  {costs[-1]}')

                ## evaluate if show_result in greater than 0 and after every show_result epochs
                if show_results > 0 and epoch % show_results == 0:
                    ## iterate over dev set
                    for iteration in range(validation_steps):
                        ## fetch one minibatch
                        data_dict = dev_dataset.get_data()     
                        temp_cost = self.val_iter(data_dict, False)

                        if isinstance(temp_cost, tuple) or isinstance(temp_cost, list):
                            if iteration == 0:
                                dev_minibatch_cost = []
                                for tc in temp_cost:
                                    dev_minibatch_cost.append([tc])
                            else:
                                for i in range(len(temp_cost)):
                                    dev_minibatch_cost[i] += temp_cost[i]
                            if iteration > 0:
                                sys.stdout.write("\033[K")
                            op_string = 'costs - '
                            for i in range(len(temp_cost)):
                                op_string += f'{i} = {temp_cost[i]} |'
                            print(f'[DEV] minibatches - {iteration + 1}/{validation_steps} | {op_string}', end='\r')
                        elif isinstance(temp_cost, dict):
                            if iteration == 0:
                                dev_minibatch_cost = {}
                                for key, value in temp_cost.items():
                                    dev_minibatch_cost[key] = value
                            else:
                                for key, value in temp_cost.items():
                                    dev_minibatch_cost[key] += value
                            if iteration > 0:
                                sys.stdout.write("\033[K")
                            op_string = 'costs - '
                            for key, value in temp_cost.items():
                                op_string += f'{key} = {value} |'
                            print(f'[DEV] minibatches - {iteration + 1}/{validation_steps} | {op_string}', end='\r')
                        else:    
                            dev_minibatch_cost += temp_cost
                            if iteration > 0:
                                sys.stdout.write("\033[K")
                            print(f'[DEV] {iteration + 1}/{validation_steps} minibatches processed | cost - {temp_cost}', end='\r')
                    
                    ## track cost and PSNR
                    old_logs = {}
                    if epoch > start_epoch:
                        old_logs = logs
                    logs = {}
                    if isinstance(dev_minibatch_cost, tuple) or isinstance(dev_minibatch_cost, list):
                        if len(dev_costs) == 0:
                            for mc in dev_minibatch_cost:
                                dev_costs.append([mc])
                        else:
                            for i in range(len(dev_minibatch_cost)):
                                dev_costs[i].append(dev_minibatch_cost[i])
                        
                        for i in range(len(minibatch_cost)):
                            logs[f'loss_{i}'] = minibatch_cost[i]
                            if epoch == start_epoch:
                                logs[f'loss_{i}_min_val'] = logs[f'loss_{i}']
                                logs[f'loss_{i}_min_epoch'] = epoch
                                logs[f'loss_{i}_max_val'] = logs[f'loss_{i}']
                                logs[f'loss_{i}_max_epoch'] = epoch
                            else:
                                if old_logs[f'loss_{i}_min_val'] >= logs[f'loss_{i}']:
                                    logs[f'loss_{i}_min_val'] = logs[f'loss_{i}']
                                    logs[f'loss_{i}_min_epoch'] = epoch
                                else:
                                    logs[f'loss_{i}_min_val'] = old_logs[f'loss_{i}_min_val']
                                    logs[f'loss_{i}_min_epoch'] = old_logs[f'loss_{i}_min_epoch']

                                if old_logs[f'loss_{i}_max_val'] <= logs[f'loss_{i}']:
                                    logs[f'loss_{i}_max_val'] = logs[f'loss_{i}']
                                    logs[f'loss_{i}_max_epoch'] = epoch
                                else:
                                    logs[f'loss_{i}_max_val'] = old_logs[f'loss_{i}_max_val']
                                    logs[f'loss_{i}_max_epoch'] = old_logs[f'loss_{i}_max_epoch']

                        sys.stdout.write("\033[K")
                        print(f'[DEV] cost after {epoch} epochs = ')                        
                        for i in range(len(dev_minibatch_cost)):
                            print(f'[DEV] {i} = {dev_minibatch_cost[i]}')
                            logs[f'val_loss_{i}'] = dev_minibatch_cost[i]
                            if epoch == start_epoch:
                                logs[f'val_loss_{i}_min_val'] = logs[f'val_loss_{i}']
                                logs[f'val_loss_{i}_min_epoch'] = epoch
                                logs[f'val_loss_{i}_max_val'] = logs[f'val_loss_{i}']
                                logs[f'val_loss_{i}_max_epoch'] = epoch
                            else:
                                if old_logs[f'val_loss_{i}_min_val'] >= logs[f'val_loss_{i}']:
                                    logs[f'val_loss_{i}_min_val'] = logs[f'val_loss_{i}']
                                    logs[f'val_loss_{i}_min_epoch'] = epoch
                                else:
                                    logs[f'val_loss_{i}_min_val'] = old_logs[f'val_loss_{i}_min_val']
                                    logs[f'val_loss_{i}_min_epoch'] = old_logs[f'val_loss_{i}_min_epoch']
                                    
                                if old_logs[f'val_loss_{i}_max_val'] <= logs[f'val_loss_{i}']:
                                    logs[f'val_loss_{i}_max_val'] = logs[f'val_loss_{i}']
                                    logs[f'val_loss_{i}_max_epoch'] = epoch
                                else:
                                    logs[f'val_loss_{i}_max_val'] = old_logs[f'val_loss_{i}_max_val']
                                    logs[f'val_loss_{i}_max_epoch'] = old_logs[f'val_loss_{i}_max_epoch']

                    elif isinstance(dev_minibatch_cost, dict):
                        if not isinstance(dev_costs, dict):
                            dev_costs = {}
                            for key, value in dev_minibatch_cost.items():
                                dev_costs[key] = [value]
                        else:
                            for key, value in dev_minibatch_cost.items():
                                dev_costs[key].append(value)

                        for key, value in minibatch_cost.items():
                            logs[f'{key}'] = value
                            if epoch == start_epoch:
                                logs[f'{key}_min_val'] = logs[f'{key}']
                                logs[f'{key}_min_epoch'] = epoch
                                logs[f'{key}_max_val'] = logs[f'{key}']
                                logs[f'{key}_max_epoch'] = epoch
                            else:
                                if old_logs[f'{key}_min_val'] >= logs[f'{key}']:
                                    logs[f'{key}_min_val'] = logs[f'{key}']
                                    logs[f'{key}_min_epoch'] = epoch
                                else:
                                    logs[f'{key}_min_val']  = old_logs[f'{key}_min_val'] 
                                    logs[f'{key}_min_epoch']  = old_logs[f'{key}_min_epoch']

                                if old_logs[f'{key}_max_val'] <= logs[f'{key}']:
                                    logs[f'{key}_max_val'] = logs[f'{key}']
                                    logs[f'{key}_max_epoch'] = epoch
                                else:
                                    logs[f'{key}_max_val'] = old_logs[f'{key}_max_val']
                                    logs[f'{key}_max_epoch'] = old_logs[f'{key}_max_epoch']                                                               

                        sys.stdout.write("\033[K")
                        print(f'[DEV] cost after {epoch} epochs = ')
                        for key, value in dev_minibatch_cost.items():
                            print(f'[DEV] {key} = {value}')
                            logs[f'val_{key}'] = value
                            if epoch == start_epoch:
                                logs[f'val_{key}_min_val'] = logs[f'val_{key}']
                                logs[f'val_{key}_min_epoch'] = epoch
                                logs[f'val_{key}_max_val'] = logs[f'val_{key}']
                                logs[f'val_{key}_max_epoch'] = epoch
                            else:
                                if old_logs[f'val_{key}_min_val'] >= logs[f'val_{key}']:
                                    logs[f'val_{key}_min_val']  = logs[f'val_{key}']
                                    logs[f'val_{key}_min_epoch'] = epoch
                                else:
                                    logs[f'val_{key}_min_val'] = old_logs[f'val_{key}_min_val'] 
                                    logs[f'val_{key}_min_epoch'] = old_logs[f'val_{key}_min_epoch']
                                    
                                if old_logs[f'val_{key}_max_val'] <= logs[f'val_{key}']:
                                    logs[f'val_{key}_max_val'] = logs[f'val_{key}']
                                    logs[f'val_{key}_max_epoch']  = epoch
                                else:
                                    logs[f'val_{key}_max_val'] = old_logs[f'val_{key}_max_val']
                                    logs[f'val_{key}_max_epoch']  = old_logs[f'val_{key}_max_epoch'] 
                    else:    
                        dev_costs.append(dev_minibatch_cost) # /len(minibatch_cost))
                        sys.stdout.write("\033[K")
                        print(f'[DEV] cost after {epoch} epochs =  {dev_costs[-1]}')
                        ## epoch end callbacks                                        

                        logs = {'loss' : costs[-1],
                            'val_loss' : dev_costs[-1]}
                        if epoch == start_epoch:
                            logs[f'loss_min_val'] = logs[f'loss']
                            logs[f'loss_min_epoch'] = epoch
                            logs[f'loss_max_val'] = logs[f'loss']
                            logs[f'loss_max_epoch'] = epoch

                            logs[f'val_loss_min_val'] = logs[f'val_loss']
                            logs[f'val_loss_min_epoch'] = epoch
                            logs[f'val_loss_max_val'] = logs[f'val_loss']
                            logs[f'val_loss_max_epoch'] = epoch
                        else:
                            if old_logs[f'loss_min_val'] >= logs[f'loss']:
                                logs[f'loss_min_val'] = logs[f'loss']
                                logs[f'loss_min_epoch'] = epoch
                            else:
                                logs[f'loss_min_val'] = old_logs[f'loss_min_val']
                                logs[f'loss_min_epoch'] = old_logs[f'loss_min_epoch']

                            if old_logs[f'loss_max_val'] <= logs[f'loss']:
                                logs[f'loss_max_val'] = logs[f'loss']
                                logs[f'loss_max_epoch'] = epoch
                            else:
                                logs[f'loss_max_val'] = old_logs[f'loss_max_val']
                                logs[f'loss_max_epoch'] = old_logs[f'loss_max_epoch']

                            if old_logs[f'val_loss_min_val'] >= logs[f'val_loss']:
                                logs[f'val_loss_min_val'] = logs[f'val_loss']
                                logs[f'val_loss_min_epoch'] = epoch
                            else:
                                logs[f'val_loss_min_val'] = old_logs[f'val_loss_min_val']
                                logs[f'val_loss_min_epoch'] = old_logs[f'val_loss_min_epoch']

                            if old_logs[f'val_loss_max_val'] <= logs[f'val_loss']:
                                logs[f'val_loss_max_val'] = logs[f'val_loss']
                                logs[f'val_loss_max_epoch'] = epoch
                            else:
                                logs[f'val_loss_max_val'] = old_logs[f'val_loss_max_val']
                                logs[f'val_loss_max_epoch'] = old_logs[f'val_loss_max_val']

                    for kmetric, vmetric in self.eval_metrics.items():
                        dev_metric[f'val_{kmetric}'].append(vmetric.result())
                        vmetric.reset_state()
                        dev_epoch_metric[f'val_{kmetric}'] = dev_metric[f'val_{kmetric}'][-1]
                     #'| PSNR = {dev_psnr[-1]}')
                    pprint(dev_epoch_metric)                   
                    
                    for key, value in  dev_epoch_metric.items():
                        logs[key] = value
                        if epoch == start_epoch:
                            logs[f'{key}_min_val'] = logs[f'{key}']
                            logs[f'{key}_min_epoch'] = epoch
                            logs[f'{key}_max_val'] = logs[f'{key}']
                            logs[f'{key}_max_epoch'] = epoch
                        else:
                            if old_logs[f'{key}_min_val'] >= logs[f'{key}']:
                                logs[f'{key}_min_val'] = logs[f'{key}']
                                logs[f'{key}_min_epoch'] = epoch
                            else:
                                logs[f'{key}_min_val'] = old_logs[f'{key}_min_val']
                                logs[f'{key}_min_epoch'] = old_logs[f'{key}_min_epoch']

                            if old_logs[f'{key}_max_val'] <= logs[f'{key}']:
                                logs[f'{key}_max_val'] = logs[f'{key}']
                                logs[f'{key}_max_epoch'] = epoch
                            else:
                                logs[f'{key}_max_val'] = old_logs[f'{key}_max_val']
                                logs[f'{key}_max_epoch'] = old_logs[f'{key}_max_epoch']

                    for callback in callbacks:
                        callback.on_epoch_end(epoch, logs)
                else:
                    old_logs = {}
                    if epoch > start_epoch:
                        old_logs = logs
                    logs = {}
                    ## epoch end callbacks                    
                    if isinstance(dev_costs, tuple) or isinstance(dev_costs, list):
                        for i in range(len(costs)):
                            logs[f'loss_{i}'] = costs[i][-1]
                            if epoch == start_epoch:
                                logs[f'loss_{i}_min_val'] = logs[f'loss_{i}']
                                logs[f'loss_{i}_min_epoch'] = epoch
                                logs[f'loss_{i}_max_val'] = logs[f'loss_{i}']
                                logs[f'loss_{i}_max_epoch'] = epoch
                            else:
                                if old_logs[f'loss_{i}_min_val'] >= logs[f'loss_{i}']:
                                    logs[f'loss_{i}_min_val'] = logs[f'loss_{i}']
                                    logs[f'loss_{i}_min_epoch'] = epoch
                                else:
                                    logs[f'loss_{i}_min_val'] = old_logs[f'loss_{i}_min_val']
                                    logs[f'loss_{i}_min_epoch'] = old_logs[f'loss_{i}_min_epoch']

                                if old_logs[f'loss_{i}_max_val'] <= logs[f'loss_{i}']:
                                    logs[f'loss_{i}_max_val'] = logs[f'loss_{i}']
                                    logs[f'loss_{i}_max_epoch'] = epoch
                                else:
                                    logs[f'loss_{i}_max_val'] = old_logs[f'loss_{i}_max_val']
                                    logs[f'loss_{i}_max_epoch'] = old_logs[f'loss_{i}_max_epoch']

                        for i in range(len(dev_costs)):
                            logs[f'val_loss_{i}'] = dev_costs[i][-1]
                            if epoch == start_epoch:
                                logs[f'val_loss_{i}_min_val'] = logs[f'val_loss_{i}']
                                logs[f'val_loss_{i}_min_epoch'] = epoch
                                logs[f'val_loss_{i}_max_val'] = logs[f'val_loss_{i}']
                                logs[f'val_loss_{i}_max_epoch'] = epoch
                            else:
                                if old_logs[f'val_loss_{i}_min_val'] >= logs[f'val_loss_{i}']:
                                    logs[f'val_loss_{i}_min_val'] = logs[f'val_loss_{i}']
                                    logs[f'val_loss_{i}_min_epoch'] = epoch
                                else:
                                    logs[f'val_loss_{i}_min_val'] = old_logs[f'val_loss_{i}_min_val']
                                    logs[f'val_loss_{i}_min_epoch'] = old_logs[f'val_loss_{i}_min_epoch']
                                    
                                if old_logs[f'val_loss_{i}_max_val'] <= logs[f'val_loss_{i}']:
                                    logs[f'val_loss_{i}_max_val'] = logs[f'val_loss_{i}']
                                    logs[f'val_loss_{i}_max_epoch'] = epoch
                                else:
                                    logs[f'val_loss_{i}_max_val'] = old_logs[f'val_loss_{i}_max_val']
                                    logs[f'val_loss_{i}_max_epoch'] = old_logs[f'val_loss_{i}_max_epoch']

                    elif isinstance(dev_costs, dict):
                        # logs = {'loss' : costs[-1]}
                        logs = {}
                        old_logs = {}
                        if epoch > start_epoch:
                            old_logs = logs
                        for key, value in costs.items():
                            logs[f'{key}'] = value[-1]
                            if epoch == start_epoch:
                                logs[f'{key}_min_val'] = logs[f'{key}']
                                logs[f'{key}_min_epoch'] = epoch
                                logs[f'{key}_max_val'] = logs[f'{key}']
                                logs[f'{key}_max_epoch'] = epoch
                            else:
                                if old_logs[f'{key}_min_val'] >= logs[f'{key}']:
                                    logs[f'{key}_min_val'] = logs[f'{key}']
                                    logs[f'{key}_min_epoch'] = epoch
                                else:
                                    logs[f'{key}_min_val']  = old_logs[f'{key}_min_val'] 
                                    logs[f'{key}_min_epoch']  = old_logs[f'{key}_min_epoch']

                                if old_logs[f'{key}_max_val'] <= logs[f'{key}']:
                                    logs[f'{key}_max_val'] = logs[f'{key}']
                                    logs[f'{key}_max_epoch'] = epoch
                                else:
                                    logs[f'{key}_max_val'] = old_logs[f'{key}_max_val']
                                    logs[f'{key}_max_epoch'] = old_logs[f'{key}_max_epoch']
                                                                        
                        for key, value in dev_costs.items():
                            logs[f'val_{key}']  = value[-1]
                            if epoch == start_epoch:
                                logs[f'val_{key}_min_val'] = logs[f'val_{key}']
                                logs[f'val_{key}_min_epoch'] = epoch
                                logs[f'val_{key}_max_val'] = logs[f'val_{key}']
                                logs[f'val_{key}_max_epoch'] = epoch
                            else:
                                if old_logs[f'val_{key}_min_val'] >= logs[f'val_{key}']:
                                    logs[f'val_{key}_min_val']  = logs[f'val_{key}']
                                    logs[f'val_{key}_min_epoch'] = epoch
                                else:
                                    logs[f'val_{key}_min_val'] = old_logs[f'val_{key}_min_val'] 
                                    logs[f'val_{key}_min_epoch'] = old_logs[f'val_{key}_min_epoch']
                                    
                                if old_logs[f'val_{key}_max_val'] <= logs[f'val_{key}']:
                                    logs[f'val_{key}_max_val'] = logs[f'val_{key}']
                                    logs[f'val_{key}_max_epoch']  = epoch
                                else:
                                    logs[f'val_{key}_max_val'] = old_logs[f'val_{key}_max_val']
                                    logs[f'val_{key}_max_epoch']  = old_logs[f'val_{key}_max_epoch'] 

                    else:
                        logs = {'loss' : costs[-1],
                            'val_loss' : dev_costs[-1]}
                        if epoch == start_epoch:
                            logs[f'loss_min_val'] = logs[f'loss']
                            logs[f'loss_min_epoch'] = epoch
                            logs[f'loss_max_val'] = logs[f'loss']
                            logs[f'loss_max_epoch'] = epoch

                            logs[f'val_loss_min_val'] = logs[f'val_loss']
                            logs[f'val_loss_min_epoch'] = epoch
                            logs[f'val_loss_max_val'] = logs[f'val_loss']
                            logs[f'val_loss_max_epoch'] = epoch
                        else:
                            if old_logs[f'loss_min_val'] >= logs[f'loss']:
                                logs[f'loss_min_val'] = logs[f'loss']
                                logs[f'loss_min_epoch'] = epoch
                            else:
                                logs[f'loss_min_val'] = old_logs[f'loss_min_val']
                                logs[f'loss_min_epoch'] = old_logs[f'loss_min_epoch']

                            if old_logs[f'loss_max_val'] <= logs[f'loss']:
                                logs[f'loss_max_val'] = logs[f'loss']
                                logs[f'loss_max_epoch'] = epoch
                            else:
                                logs[f'loss_max_val'] = old_logs[f'loss_max_val']
                                logs[f'loss_max_epoch'] = old_logs[f'loss_max_epoch']

                            if old_logs[f'val_loss_min_val'] >= logs[f'val_loss']:
                                logs[f'val_loss_min_val'] = logs[f'val_loss']
                                logs[f'val_loss_min_epoch'] = epoch
                            else:
                                logs[f'val_loss_min_val'] = old_logs[f'val_loss_min_val']
                                logs[f'val_loss_min_epoch'] = old_logs[f'val_loss_min_epoch']

                            if old_logs[f'val_loss_max_val'] <= logs[f'val_loss']:
                                logs[f'val_loss_max_val'] = logs[f'val_loss']
                                logs[f'val_loss_max_epoch'] = epoch
                            else:
                                logs[f'val_loss_max_val'] = old_logs[f'val_loss_max_val']
                                logs[f'val_loss_max_epoch'] = old_logs[f'val_loss_max_val']

                    for key, value in  dev_epoch_metric.items():
                        logs[key] = value
                        if epoch == start_epoch:
                            logs[f'{key}_min_val'] = logs[f'{key}']
                            logs[f'{key}_min_epoch'] = epoch
                            logs[f'{key}_max_val'] = logs[f'{key}']
                            logs[f'{key}_max_epoch'] = epoch
                        else:
                            if old_logs[f'{key}_min_val'] >= logs[f'{key}']:
                                logs[f'{key}_min_val'] = logs[f'{key}']
                                logs[f'{key}_min_epoch'] = epoch
                            else:
                                logs[f'{key}_min_val'] = old_logs[f'{key}_min_val']
                                logs[f'{key}_min_epoch'] = old_logs[f'{key}_min_epoch']

                            if old_logs[f'{key}_max_val'] <= logs[f'{key}']:
                                logs[f'{key}_max_val'] = logs[f'{key}']
                                logs[f'{key}_max_epoch'] = epoch
                            else:
                                logs[f'{key}_max_val'] = old_logs[f'{key}_max_val']
                                logs[f'{key}_max_epoch'] = old_logs[f'{key}_max_epoch']

                    for callback in callbacks:
                        callback.on_epoch_end(epoch, logs)

            ## train end callbacks
            for callback in callbacks:
                callback.on_train_end(logs)

            print('LOGS -')
            pprint(logs)

            with open(f'{os.path.join(log_dir, ex_name)}.log', 'a') as f:
                f.write('END-LOGS\n')
                f.write('HYPER-PARAMETERS\n')
                f.write('----------------------\n')
                for key, value in logs.items(): 
                    f.write('%s:%s\n' % (key, value))
                f.write('\n')

        _fit(train_dataset, dev_dataset, batch_size, epochs,                             
                    callbacks, steps_per_epoch, validation_steps,
                    start_epoch, dataset_name, loss_name,
                    model_name, show_results)

    def eval(self, dev_dataset, validation_steps = None, dataset_name = '',
                    model_name = '', show_results=-1):
        '''
        This function gets executed on executing the script.
        
        Args ::
            dev_dataset -- Dev set            
            validation_steps - int | default None | when given, runs given number of val iterations per epoch
            dataset_name -- str | default '' | name of the dataset used in trainiing
            show_results -- int | default -1 | if set between 0 to epochs; computes
                                        metrics and displayes results from dev set
                                        in that intervals
        '''    
        if not self.is_compiled:
            print('[INFO] Please call TrainInfer.compile() before calling TrainInfer.train()')
            return
        if not self.is_implemented_train_iter:
            print('[INFO] Please override TrainInfer.train_iter before calling TrainInfer.train()')
            return
        if not self.is_implemented_val_iter:
            print('[INFO] Please override TrainInfer.val_iter before calling TrainInfer.train()')
            return    
        
        batch_size = 1

        n_minibatches_dev = dev_dataset.count_minibatches() if dev_dataset is not None else 0

        print(f'Total number of  examples = {dev_dataset.m}')         
        print(f'Number of minibatches in training set - {n_minibatches_dev}')
        print('Starting evaluation...')

        if validation_steps is None or validation_steps == 0:
            validation_steps = n_minibatches_dev  
        elif validation_steps < 0:
            print(f'[INFO] validation_steps can not be -ve, but found {validation_steps}')  
            return

        dev_metric = {f'val_{k}' : [] for k, v in self.eval_metrics.items()}
        dev_epoch_metric = {f'val_{k}' : 0 for k, v in self.eval_metrics.items()}

        for iteration in range(validation_steps):
            ## fetch one minibatch
            data_dict = dev_dataset.get_data()     
            self.val_iter(data_dict, eval_only = True)

            print(f'[DEV] minibatches - {iteration + 1}/{validation_steps} ', end='\r')
        
        for kmetric, vmetric in self.eval_metrics.items():
            dev_metric[f'val_{kmetric}'].append(vmetric.result())
            vmetric.reset_state()
            dev_epoch_metric[f'val_{kmetric}'] = dev_metric[f'val_{kmetric}'][-1]
            #'| PSNR = {dev_psnr[-1]}')
        pprint(dev_epoch_metric)
                    
        
    def infer_dir(self, image_dir, output_dir = None, save_type = 'image_RGB'):
        '''
        Use this function to infer from a dir of images
        save_type is one of ['image_RGB', 'raw']
        '''
        image_names = os.listdid(image_dir)
        for image_name in image_names:
            image_path = os.path.join(image_dir, image_name)
            input_image = self.read_RGB(input_image)
            input_image = self.pre_process(input_image)

            output = self.infer_step(image)
            output = self.post_process(output)     
            
            if output_dir is not None:
                if save_type == 'image_RGB':
                    self.save_RGB(os.path.join(output_dir, image_name), output)
                if save_type == 'raw':
                    self.save_raw(os.path.join(output_dir, image_name), output)
    