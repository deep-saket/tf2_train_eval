import tensorflow as tf
import numpy as np
import cv2
import argparse
import os
from tflite_utils import *
from face_match_models.model_dict import model_dict

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="TFLite Convert Script")
    parser.add_argument("--model_name", type=str, default='',
                        help="Name of the model to be used")
    parser.add_arguments("--h5_path", type=str, default=None, 
                        help="Path of the pretrained H5 model")
    parser.add_arguments("--ckpt_path", type=str, default=None, 
                        help="Path of the pretrained H5 model")
    parser.add_argument("--tflite_file_name", type=str, default='',
                        help="Name of the output tflite model")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    model_name = args.model_name

    model = model_dict[model_name]()
    
    if args.h5_path is not None and args.ckpt_path == None:
      model.load_weights(args.h5_path)
    elif args.h5_path == None and args.ckpt_path is not None:
      checkpoint = tf.train.Checkpoint(net=model)
      if os.path.exists(args.ckpt_path):
        print(f'Restoring checkpoint from {args.ckpt_path}', end='\r')
        checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_path))# .expect_partial()
    elif args.h5_path is not None and args.ckpt_path is not None:
      print(f"INFO : can't give values for both args.h5_path and args.ckpt_path")

    save_tflites(model, model_name = args.tflite_file_name)