## CVML Callbacks

* A set of classes those can be used as callbacks to keras's model.fit and also cvml training framework
* All callbacks are inherited from parent class keras.callbacks.Callback and methods are overridden as follows
* The trained model at any point in the trainining loop can be accessed as keras.callbacks.Callback.model
* At then end of each training loop training logs, which include all the losses calculated on train and dev set and metrics calculated on dev set is passed to the "on_epoch_end" and "on_train_end" etc methods

## Create custom callback
* You can create a custom callback by overridding atleaset one of
```
class CuntomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        '''
        This method is always executed at the end of an epoch

        Args --
            epoch -- int | current epoch number
            logs -- dict | default None | logs = {'loss' : float,
                                    'val_loss' : float,
                                    ...}
        '''
        ...

    def on_train_end(self, logs=None):
        '''
        This function is called at the end of the training experiment


        Args --
            logs -- dict | default None | logs = {'loss' : float,
                                    'val_loss' : float,
                                    ...}
        '''
```

## Use callbacks
* Callbacks can be used with the TrainInfer class

```
    ## Store all the callbacks in the list
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

## Available callbacks
1. Checkpoint Callback - <br>
* Use this callback to save checkpoints at the end of each epoch
* It takes log_dir (dir to store the checkpoint files) and basis_dict as input 
* Use basis_dict to indicate when to store the updated checkpoint
* e.g. in the example below 'val_loss' : 'min' tells the callback to store the updated checkpoint, whenever val loss is decreasing globally.
* frequency = 5 tells the callback to run after every 5 epochs
```
    basis_dict = {
        'val_loss' : 'min'        
    }
    CkptCallback(log_dir, basis_dict, skip_from_start=4, frequency=5),
```

2. Tensorboard Callback - <br>
* Use this callback to log data in tensorboard
* It takes log_dir (dir to store the tensorboard file) and basis_dict as input 
* Use basis_dict to indicate when to store the updated checkpoint
```
    basis_dict = {
            'val_loss' : 'min'        
    }   
    TensorboardCallback(log_dir, basis_dict, image_shape = (224, 224, 3)),
```

3. TFLite Callback - <br>
* Use this callback to convert the model into tflite after each epoch or after the training
* It takes log_dir (dir to store the tflite file) and basis_dict as input 
* Use basis_dict to indicate when to store the updated checkpoint
```
    basis_dict = {
            'val_loss' : 'min'        
    }   
    TFLiteCallback(log_dir, basis_dict, skip_from_start=4, frequency=5)
```