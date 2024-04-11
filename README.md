# CVML

* This codebase is a framework built to ease ML model building and training stage
* This codebase contains functions to train a model, load dataset in various ways, tflite saving etc. ultilities
* There are 5 parts to this package
    * callbacks - inherited from keras.callbacks.Callback
    * tflite -  ultilities such as exporting model as tflite
    * dataloaders - various dataloaders 
    * costs - various cost function
    * metrics - various metrics to evaluate a model similar to keras metrics
    * TrainValTest - a class, which hels in training and evaluating models

## install
* Clone this repository and put it in your project folder
* install the dependencies 
* refer readme docs to use the pacage efficiently

## dependencies
* TF = 2.11
* python 3.8 or above
* knokknok (pip install knockknock)
* 

## 1. callbacks
* Like keras callbacks, these classes can be used to do certain things at the end of the training loop or iteration
* All the callbacks are derived from "keras.callbacks.Callback"
* Refer the readme in the callback dir to see more

## 2. tflite
* This package contains supporting functionalities to port a model into tflite
* Use the functions present in the tflite_utils.py file to conver tflites
* To save model form a Keras.Model model
```
## model : Keras.Model
## exp_name : name of the experimnet | default None
## epoch : epoch from which you are exporting the model | default None
## model_name : name of the model

save_tflites(model, exp_name=None, epoch=None, model_name='')
```

* To save model from a tf2 saved_model
```
## model : Keras.Model
## tflite_path :  path where the tflite file to be stored | this inclueds the name of the model e.g. tflite_path = './model/SavedModel.tflite'

save_tflites_from_saved_models(saved_model_dir, tflite_path)
```

* To save form checkpoint
```
## model : Keras.Model
## ckpt_path :  path where the checkpoint file is stored
## tflite_path :  path where the tflite file to be stored
## tflite_name : Name of the tflite file to be used | default "SavedModel"

save_tflites_ckpt(model, ckpt_path, tflite_path, tflite_name = "SavedModel")
```

## 3. dataloaders
* This package contains a set of dataloaders to be used for various deep learning tasks

## 4. costs

## 5. metrics
* This package contains set of metrics to evaluate deep learning tasks
* All metrics classes have a similar structur like keras.metrics.* 
* All of them have update_state(label, logit) method to add a label, logit pair for evaluation
* All of them have result() method to get the result
* All of them have reset_state() method to reset label, logit pairs to 0
* Refer the readme in the metrics dir for more info

## 6. TrainValTest
* Use this class to train your model, do inference and evaluate your model
* It is recomended to use this class as a parent calss and create your own class for each experiment
* You can override and customize the below function for specific use cases
```
    def train_iter(self, data_dict,):
        '''
        This function is what's executed every train epoch
        '''

    def val_iter(self, data_dict, eval_only):
        '''
        This function is what's executed every val epoch
        '''

    def train_step(self, X, Y):
        '''
        Train one minibatch
        '''

    def test_step(self, X, Y):
        '''
        Infer one minibatch
        '''
 
    def infer_step(self, input_image):
        '''
        Infer one sample 
        '''

    def post_process(self, output):
        '''
        Post process one batch
        '''

    def normalise(self, image):
        '''
        Override this method to use custom normalization technique.
        Default - rescale by 255
        '''

```

* You can use this class to train your model 
```
    webhook_url = "https://10xar.webhook.office.com/webhookb2/3866964d-3507-4e90-bfc7-481e0cf65c5f@9de8f7b3-390e-403c-82ea-7393b60d6ee3/IncomingWebhook/1be4d2c020e141fea3238d750a9dd1ae/85ed6752-36c5-47f4-9726-73145956be76"

    ## Load datasets
    train_dataset = 
    dev_dataset = 

    ## Create model and model trainer
    model = 

    ## load pretrained model
    trainInfer.load_h5('path to the pretrained model.h5')

    # Define loss
    bce = triplet_cost  # tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Define evaluation metrics
    eval_dict = {
                'val_precision' : tf.keras.metrics.Precision(
                                    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
                                ),
                'val_recall' : tf.keras.metrics.Recall(
                                    thresholds=None, top_k=None, class_id=None, name=None, dtype=None
                                ),
                'val_bin_acc' : tf.keras.metrics.BinaryAccuracy(
                                    name='accuracy', dtype=None, threshold=0.5
                                ),
            }

    # Compile the model
    trainInfer.compile(compute_cost= bce,
                        lr = lr,
                        optimizer_name = optimizer_name, 
                        eval_metrics = eval_dict)

    # Create callbacks
    basis_dict = {
        'val_loss' : 'min'        
    }
    callbacks = [
        CkptCallback(log_dir, basis_dict, skip_from_start=4, frequency=1),
        TensorboardCallback(log_dir, basis_dict, image_shape = image_shape + [3]),
        TFLiteCallback(log_dir, basis_dict, skip_from_start=4, frequency=5)
    ]

    # Train the model
    trainInfer.train(train_dataset, dev_dataset, batch_size, epochs,                             
                    callbacks = callbacks, start_epoch = start_epoch, 
                     show_results=1, teams_webhook_url=webhook_url)
```

* You can also load checkpoint to the model using
```
## restore_model_path : path the the pretrained checkpoint
## optimizer : if False loads only the model, else loads the optimizer's state | use "optimizer = True" only after trainInfer.compile
trainInfer.load_checkpoint_train(restore_model_path, optimizer = True)
```

* You can also evaluate the model using
```
    # Define evaluation metrics
    eval_dict = {'mAP' : PascalVOCObjectDetectionMetrics('xyx2y2',  image_shape , [1], 'absolute')}

    # Compile the model
    trainInfer.compile(eval_metrics = eval_dict)

    # Create callbacks
    basis_dict = {
        'val_loss_2' : 'min'        
    }

    # Eval the model
    trainInfer.eval(dev_dataset)
```