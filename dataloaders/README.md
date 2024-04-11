## CVML DataLoaders

* A set of classes those can be used as dataloaders in a training script
* These dataloaders are compatible with TrainValTest class defined in cvml/TrainValTest.py

## Types of loaders
### <b>Basic</b> -  
* loads data directly from SSD or HDD
* does not inherit tf2 or keras's dataloaders
* use get_data() method to get 1 minibatch
* use count_minibatches() to get number of minibatches
* dataLoader.m gives the total numbers of records present in the dataset

###

### TripletLoader.py
* This file contains dataloaders to that can load triplets (of images) along with positive class lable
* triplet contains +ve sample, -ve sample and anchor image
```
class TripletLoaderBasic:
    '''
    Loads the datset.
    Input Directory :
        IMAGE_DIR
        |_CLASS1
        |_CLASS2
        ...
        |_CLASSn
    Returns as
    [
        (
            ndarray of share BATCH_SIZExHxWxC, 
            ndarray of share BATCH_SIZExHxWxC, 
            ndarray of share BATCH_SIZExHxWxC
        )
        ndarray of shape 1xBATCH_SIZE
    ]
    def __init__(self, batch_size, dataset_path, grayscale=False, hflip=False, vflip=False, rotate=False, blur=False, size=(224, 224), visualize=False):
        '''
        Initialize the class.
        Args ::
            batch_size -- int | number of examples to be processed in one example
            dataset_path -- str | path to images dir
            dataset_info_path -- str | info.txt path
        '''
    def get_data(self):
        '''
        fetch one batch at a time
        '''
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
```

### FaceDetection.py
* This file contains dataloaders to be used in face detection tasks
* Coresponding datasets can be downloaded from sharpoint or can be used in custom datasets
```
class FaceDetectionBasic:
    '''
    Loads the datset.

    Input directory
        DATA_DIR
        |_label.txt
        |_images

    label.txt
        - contains image paths 
        - contains bbox annotation x1, y1, w, h
        - contains landmarks
        - e.g. 
            # 0--Parade/0_Parade_marchingband_1_849.jpg
            449 330 122 149 488.906 373.643 0.0 542.089 376.442 0.0 515.031 412.83 0.0 485.174 425.893 0.0 538.357 431.491 0.0
        - line containing image path starts with '#' symol and it follows annotations < x1 y1 w h lm1x lm1y lm1z ... lm5x lm5y lm5z >
        - if landmarks are not available, then put -1 in the place of the coordinate values of the landmarks

    Returns as
    {
        images : ndarray of shape BATCH_SIZE x H x W x C
        labels : ndarray of shape BATCH_SIZE x 15
    }
    def __init__(self, batch_size, dataset_path, using_bin=True, priors=None, match_thresh=0.45, ignore_thresh=0.3,
                     variances=[0.1, 0.2], distort=False, grayscale=False, flip=False, shuffle=True,
                     rotate=False, blur=False, size=(224, 224), visualize=False, has_landmarks=True):
        '''
        Initialize the class.
        Args ::
            batch_size -- int | number of examples to be processed in one example
            dataset_path -- str | dataset dir
            dataset_info_path -- str | info.txt path
        '''
    def get_data(self):
        '''
        fetch one batch at a time
        '''
```