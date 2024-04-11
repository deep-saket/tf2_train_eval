from email.mime import image
from builtins import len, print
from cProfile import label
import tensorflow as tf
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import OrderedDict
import math
import os
import random
from cvml import pista
from scipy import ndimage
import random
from pprint import pprint
from cvml.dataloaders.widerface.augmentations import *
from cvml.dataloaders.widerface.anchor import *
import logging
import tqdm

def normalize(data_batch):
    '''
    Normalizes the inopt batch
    '''
    norm_batch = []

    norm_batch.append(data_batch[0] / 255.0)
    norm_batch.append(data_batch[-1] / 255.0)

    return tuple(norm_batch)
    
class FaceDetectionTFRcLoader:
    '''
    Loads the datset.
    Returns as
    {
        images : ndarray of shape BATCH_SIZE x H x W x C
        labels : ndarray of shape BATCH_SIZE x 15
    }
    '''
    def __init__(self, batch_size, dataset_path, using_bin=True, priors=None, match_thresh=0.45, ignore_thresh=0.3,
                     variances=[0.1, 0.2], buffer_size=10240, distort=False, using_encoding=True, grayscale=False, 
                     flip=False, shuffle=True, rotate=False, blur=False, size=(224, 224), visualize=False):
        '''
        Initialize the class.
        Args ::
            batch_size -- int | number of examples to be processed in one example
            dataset_path -- str | tfrecord dir
            dataset_info_path -- str | info.txt path
        '''
        self._batch_size = batch_size
        self._dataset_path = dataset_path
        self._fetched_batch = 0
        self.grayscale = grayscale
        self.rotate = rotate
        self.blur = blur
        self.flip = flip
        self.size = size
        self.shuffle = shuffle
        self.input_size = (size[0], size[1], 3)
        self.visualize = visualize
        self.using_bin = using_bin
        self.priors = priors
        self.match_thresh = match_thresh
        self.ignore_thresh = ignore_thresh
        self.variances = variances
        self.buffer_size = buffer_size
        self.distort = distort
        self.using_encoding = using_encoding

        self.dataset = self._prepare_data(
                            tfrecord_name=self._dataset_path,
                            batch_size=self._batch_size,
                            img_dim=self.input_size ,
                            using_bin=self.using_bin,
                            using_flip=self.flip,
                            using_distort=self.distort ,
                            using_encoding=True,
                            priors=self.priors,
                            match_thresh=self.match_thresh,
                            ignore_thresh=self.ignore_thresh,
                            variances=self.variances,
                            shuffle=self.shuffle,
                            buffer_size=self.buffer_size)
        self.n_batches = tf.data.experimental.cardinality(self.dataset).numpy()
        self.m = len(list(self.dataset)) * self._batch_size      

    def _transform_data(self, img_dim, using_flip, using_distort, using_encoding, priors,
                    match_thresh, ignore_thresh, variances):
        def transform_data(img, labels):
            img = tf.cast(img, tf.float32)

            # randomly crop
            img, labels = crop(img, labels)

            # padding to square
            img = pad_to_square(img)

            # resize
            img, labels = resize(img, labels, img_dim)

            # randomly left-right flip
            if using_flip:
                img, labels = flip(img, labels)

            # distort
            if using_distort:
                img = distort(img)

            # encode labels to feature targets
            if using_encoding:
                labels = encode_tf(labels=labels, priors=priors,
                                match_thresh=match_thresh,
                                ignore_thresh=ignore_thresh,
                                variances=variances)

            return img, labels
        return transform_data

    def _parse_tfrecord(self, img_dim, using_bin, using_flip, using_distort,
                    using_encoding, priors, match_thresh, ignore_thresh,
                    variances):
        def parse_tfrecord(tfrecord):
            features = {
                'image/img_name': tf.io.FixedLenFeature([], tf.string),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark0/x': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark0/y': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark1/x': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark1/y': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark2/x': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark2/y': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark3/x': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark3/y': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark4/x': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark4/y': tf.io.VarLenFeature(tf.float32),
                'image/object/landmark/valid': tf.io.VarLenFeature(tf.float32)}
            if using_bin:
                features['image/encoded'] = tf.io.FixedLenFeature([], tf.string)
                x = tf.io.parse_single_example(tfrecord, features)
                img = tf.image.decode_jpeg(x['image/encoded'], channels=3)
            else:
                features['image/img_path'] = tf.io.FixedLenFeature([], tf.string)
                x = tf.io.parse_single_example(tfrecord, features)
                image_encoded = tf.io.read_file(x['image/img_path'])
                img = tf.image.decode_jpeg(image_encoded, channels=3)

            labels = tf.stack(
                [tf.sparse.to_dense(x['image/object/bbox/xmin']),
                tf.sparse.to_dense(x['image/object/bbox/ymin']),
                tf.sparse.to_dense(x['image/object/bbox/xmax']),
                tf.sparse.to_dense(x['image/object/bbox/ymax']),
                tf.sparse.to_dense(x['image/object/landmark0/x']),
                tf.sparse.to_dense(x['image/object/landmark0/y']),
                tf.sparse.to_dense(x['image/object/landmark1/x']),
                tf.sparse.to_dense(x['image/object/landmark1/y']),
                tf.sparse.to_dense(x['image/object/landmark2/x']),
                tf.sparse.to_dense(x['image/object/landmark2/y']),
                tf.sparse.to_dense(x['image/object/landmark3/x']),
                tf.sparse.to_dense(x['image/object/landmark3/y']),
                tf.sparse.to_dense(x['image/object/landmark4/x']),
                tf.sparse.to_dense(x['image/object/landmark4/y']),
                tf.sparse.to_dense(x['image/object/landmark/valid'])], axis=1)

            img, labels = self._transform_data(
                img_dim, using_flip, using_distort, using_encoding, priors,
                match_thresh, ignore_thresh, variances)(img, labels)

            return img, labels
        return parse_tfrecord

    def _prepare_data(self, tfrecord_name, batch_size, img_dim,
                          using_bin=True, using_flip=True, using_distort=True,
                          using_encoding=True, priors=None, match_thresh=0.45,
                          ignore_thresh=0.3, variances=[0.1, 0.2],
                          shuffle=True, buffer_size=10240): # load_tfrecord_dataset
        """load dataset from tfrecord"""
        if not using_encoding:
            assert batch_size == 1  # dynamic data len when using_encoding
        else:
            assert priors is not None

        raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
        raw_dataset = raw_dataset.repeat()
        print(4)
        if shuffle:
            raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
        dataset = raw_dataset.map(
            self._parse_tfrecord(img_dim, using_bin, using_flip, using_distort,
                            using_encoding, priors, match_thresh, ignore_thresh,
                            variances),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(5)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        print(6)
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
        print(7)

        return dataset
                    
    def _get_items(self):                 
        return self.dataset.get_single_element()        
    
    def get_data(self):
        '''
        fetch one batch at a time
        '''
        if self._fetched_batch >= self.n_batches:
            self.dataset = self._prepare_data(
                                tfrecord_name=self._dataset_path,
                                batch_size=self._batch_size,
                                img_dim=self.input_size ,
                                using_bin=self.using_bin,
                                using_flip=self.flip,
                                using_distort=self.distort ,
                                using_encoding=True,
                                priors=self.priors,
                                match_thresh=self.match_thresh,
                                ignore_thresh=self.ignore_thresh,
                                variances=self.variances,
                                shuffle=self.shuffle,
                                buffer_size=self.buffer_size)
            self._fetched_batch = 0

        images, labels = self._get_items()
        self._fetched_batch += 1

        data_batch = {'images': images, 'labels': labels}

        return data_batch
    
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
        return self.n_batches

    @classmethod
    def _bytes_feature(cls, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @classmethod
    def _float_feature(cls, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @classmethod
    def _int64_feature(cls, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))    

    @classmethod
    def make_example(cls, img_name, img_path, target, is_binary):
        # Create a dictionary with features that may be relevant.
        feature = {'image/img_name': cls._bytes_feature([img_name]),
                'image/object/bbox/xmin': cls._float_feature(target[:, 0]),
                'image/object/bbox/ymin': cls._float_feature(target[:, 1]),
                'image/object/bbox/xmax': cls._float_feature(target[:, 2]),
                'image/object/bbox/ymax': cls._float_feature(target[:, 3]),
                'image/object/landmark0/x': cls._float_feature(target[:, 4]),
                'image/object/landmark0/y': cls._float_feature(target[:, 5]),
                'image/object/landmark1/x': cls._float_feature(target[:, 6]),
                'image/object/landmark1/y': cls._float_feature(target[:, 7]),
                'image/object/landmark2/x': cls._float_feature(target[:, 8]),
                'image/object/landmark2/y': cls._float_feature(target[:, 9]),
                'image/object/landmark3/x': cls._float_feature(target[:, 10]),
                'image/object/landmark3/y': cls._float_feature(target[:, 11]),
                'image/object/landmark4/x': cls._float_feature(target[:, 12]),
                'image/object/landmark4/y': cls._float_feature(target[:, 13]),
                'image/object/landmark/valid': cls._float_feature(target[:, 14])}
        if is_binary:
            img_str = open(img_path, 'rb').read()
            feature['image/encoded'] = cls._bytes_feature([img_str])
        else:
            feature['image/img_path'] = cls._bytes_feature([img_path])

        return tf.train.Example(features=tf.train.Features(feature=feature))

    @classmethod
    def load_info(cls, txt_path):
        """load info from txt"""
        img_paths = []
        words = []

        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                img_paths.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        words.append(labels)
        return img_paths, words

    @classmethod
    def get_target(cls, labels):
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1  # w/o landmark
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        return target

    @classmethod
    def main(cls, dataset_path, output_path, is_binary):
        dataset_path = dataset_path

        if not os.path.isdir(dataset_path):
            logging.info('Please define valid dataset path.')
        else:
            logging.info('Loading {}'.format(dataset_path))

        logging.info('Reading data list...')
        img_paths, words = cls.load_info(os.path.join(dataset_path, 'label.txt'))
        samples = list(zip(img_paths, words))
        random.shuffle(samples)

        if os.path.exists(output_path):
            logging.info('{:s} already exists. Exit...'.format(
                output_path))
            exit()

        logging.info('Writing {} sample to tfrecord file...'.format(len(samples)))
        with tf.io.TFRecordWriter(output_path) as writer:
            for img_path, word in tqdm.tqdm(samples):
                target = cls.get_target(word)
                img_name = os.path.basename(img_path).replace('.jpg', '')

                tf_example = cls.make_example(img_name=str.encode(img_name),
                                        img_path=str.encode(img_path),
                                        target=target,
                                        is_binary=is_binary)

                writer.write(tf_example.SerializeToString())

class FaceDetectionBasic:
    '''
    Loads the datset.
    Returns as
    {
        images : ndarray of shape BATCH_SIZE x H x W x C
        labels : ndarray of shape BATCH_SIZE x 15
    }
    '''
    def __init__(self, batch_size, dataset_path, using_bin=True, priors=None, match_thresh=0.45, ignore_thresh=0.3,
                     variances=[0.1, 0.2], distort=False, grayscale=False, flip=False, shuffle=True,
                     rotate=False, blur=False, size=(224, 224), visualize=False, has_landmarks=True):
        '''
        Initialize the class.
        Args ::
            batch_size -- int | number of examples to be processed in one example
            dataset_path -- str | tfrecord dir
            dataset_info_path -- str | info.txt path
        '''
        self._batch_size = batch_size
        self._dataset_path = dataset_path
        self._fetched_batch = 0
        self.grayscale = grayscale
        self.rotate = rotate
        self.blur = blur
        self.flip = flip
        self.size = size
        self.shuffle = shuffle
        self.input_size = (size[0], size[1], 3)
        self.visualize = visualize
        self.using_bin = using_bin
        self.priors = priors
        self.match_thresh = match_thresh
        self.ignore_thresh = ignore_thresh
        self.variances = variances
        self.distort = distort
        self.image_paths = self.words = None
        self.using_encoding = True
        self.has_landmarks = has_landmarks        

        self.dataset = self._prepare_data()
        self.n_batches = len(list(self.dataset)) * self._batch_size      
        self.m = len(self.image_paths)

    def _transform_data(self, img_dim, using_flip, using_distort, using_encoding, priors,
                    match_thresh, ignore_thresh, variances):
        def transform_data(img, labels):
            img = tf.cast(img, tf.float32)

            # randomly crop
            img, labels = crop(img, labels)

            # padding to square
            img = pad_to_square(img)

            # resize
            img, labels = resize(img, labels, img_dim)

            # randomly left-right flip
            if using_flip:
                img, labels = flip(img, labels)

            # distort
            if using_distort:
                img = distort(img)

            # encode labels to feature targets
            if using_encoding:
                encoded_lables = encode_tf(labels=labels, priors=priors,
                                match_thresh=match_thresh,
                                ignore_thresh=ignore_thresh,
                                variances=variances)
                return img, encoded_lables, labels

            return img, labels
        return transform_data


    def _prepare_data(self): # load_tfrecord_dataset
        """load dataset from tfrecord"""
        if not self.using_encoding:
            assert self._batch_size == 1  # dynamic data len when using_encoding
        else:
            assert self.priors is not None

        if self.image_paths == None or self.words == None:
            self.image_paths, self.words = self.load_info(os.path.join(self._dataset_path, 'label.txt'))
        
        return pista.random_mini_batches_from_SSD(self.image_paths, self.words, self._batch_size)          
    
    def count_minibatches(self):
        '''
        Returns number of minibatches.
        '''
        return self.n_batches

    def load_info(self, txt_path):
        """load info from txt"""
        img_paths = []
        words = []

        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                img_paths.append(path)
            else:
                line = line.split(' ')
                labels.append(line)
                # label = [float(x) for x in line]
                # labels.append(label)

        words.append(labels)
        samples = list(zip(img_paths, words))
        return img_paths, words

    def get_target(self, labels):        
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = float(label[0])  # x1
            annotation[0, 1] = float(label[1])  # y1
            annotation[0, 2] = float(label[0]) + float(label[2])  # x2
            annotation[0, 3] = float(label[1]) + float(label[3])  # y2

            # landmarks
            if self.has_landmarks:
                annotation[0, 4] = float(label[4])    # l0_x
                annotation[0, 5] = float(label[5])    # l0_y
                annotation[0, 6] = float(label[7])    # l1_x
                annotation[0, 7] = float(label[8])    # l1_y
                annotation[0, 8] = float(label[10])   # l2_x
                annotation[0, 9] = float(label[11])   # l2_y
                annotation[0, 10] = float(label[13])  # l3_x
                annotation[0, 11] = float(label[14])  # l3_y
                annotation[0, 12] = float(label[16])  # l4_x
                annotation[0, 13] = float(label[17])  # l4_y
                if (annotation[0, 4] < 0):
                    annotation[0, 14] = -1  # w/o landmark
                else:
                    annotation[0, 14] = 1
            else:
                annotation[0, 14] = -1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        return target
    
    def _get_item(self, sample):
        image_path, label = sample
        image = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), self.size)
        label = self.get_target(label)

        image, label, decoded_lable = self._transform_data(
                self.input_size, self.flip, self.distort, True, self.priors,
                self.match_thresh, self.ignore_thresh, self.variances)(image, label)

        return image, label, decoded_lable
    
    def _get_items(self, batch):
        images = []
        labels = []
        decoded_lables = []
        for sample in zip(batch[0], batch[1]):
            image, label, decoded_lable = self._get_item(sample)
            images.append(image)
            labels.append(label)
            decoded_lables.append(decoded_lable)

        images = np.stack(images, axis=0)
        labels = np.stack(labels, 0)

        return images, labels, decoded_lables
    
    def get_data(self):
        '''
        fetch one batch at a time
        '''
        if self._fetched_batch >= self.n_batches:
            self.dataset = self._prepare_data()
            self._fetched_batch = 0

        images, labels, decoded_lables = self._get_items(self.dataset[self._fetched_batch])
        self._fetched_batch += 1

        data_batch = {'images': images, 'labels': labels, 'decoded_labels' : decoded_lables}

        return data_batch


if __name__ == '__main__':
    using_bin = True
    using_flip = True
    using_distort = True
    
    min_sizes = [[16, 32], [64, 128], [256, 512]]
    steps = [8, 16, 32]
    clip = False
    size=[640, 640]
    priors = prior_box(size, min_sizes, steps, clip)
    
    match_thresh = 0.45
    ignore_thresh = 0.3
    variances = [0.1, 0.2]
    shuffle = True
    buffer_size = 10240

    dataloader = FaceDetectionBasic(2, '/media/saket/Elements/datasets/faces/face_detection/wider_face/WIDER_val/',
                                         using_bin=using_bin, priors=priors, match_thresh=match_thresh, 
                                            ignore_thresh=ignore_thresh, variances=variances,
                                            distort=using_distort, grayscale=False, flip=using_flip,
                                            shuffle=True, rotate=False, blur=False, size=size, visualize=False, has_landmarks=False)
    print('Minibatches =', dataloader.n_batches )
    print(11)
    for i in range(10):
        data_batch = dataloader.get_data()

        print(data_batch['images'].shape)
        print(data_batch['labels'].shape)