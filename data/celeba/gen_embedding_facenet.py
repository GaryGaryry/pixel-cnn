"""Validate a face recognizer on the "Labeled Faces in the Wild" dataset (http://vis-www.cs.umass.edu/lfw/).
Embeddings are calculated using the pairs from http://vis-www.cs.umass.edu/lfw/pairs.txt and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
sys.path.append('/media/disk1/gaoyuan/facenet/src')
import facenet
import os
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import cv2
from sklearn.model_selection import train_test_split
import pickle


def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:

            image_list = []
            label_list = []
            image_paths = []
            align_dataset_path = '/media/disk1/gaoyuan/pixel-cnn/data/celeba/Img_align_mtcnn/img'
            label_sub_paths = os.listdir(align_dataset_path)
#            for label_sub_path in label_sub_paths:
#                image_sub_paths = os.listdir(os.path.join(align_dataset_path, label_sub_path))
#                for image_sub_path in image_sub_paths:
#                    image_path = os.path.join(align_dataset_path, label_sub_path, image_sub_path)
#                    try:
#                        image = cv2.imread(image_path)
#                        image = cv2.resize(image, (32, 32))[..., ::-1]
#                    except:
#                        print('Unable to read image from: ', image_path)
#                        continue
#                    image_list.append(image)
#                    label_list.append(label_sub_path)
#                    image_paths.append(image_path)

            image_list = np.load('/media/disk1/gaoyuan/pixel-cnn/data/celeba/image_list.npy').tolist()
            label_list = np.load('/media/disk1/gaoyuan/pixel-cnn/data/celeba/label_list.npy').tolist()
            image_paths = np.load('/media/disk1/gaoyuan/pixel-cnn/data/celeba/image_paths.npy').tolist()
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
 
            nrof_preprocess_threads = 4
            image_size = (args.image_size, args.image_size)
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(args.model, input_map=input_map)

            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#              
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            # Run forward pass to calculate embeddings
            print('Runnning forward pass on Celeba images')
            
            # Enqueue one epoch of image paths and labels
            nrof_embeddings = len(image_paths)  # nrof_pairs * nrof_images_per_pair
            nrof_images = nrof_embeddings
            labels_array = np.expand_dims(np.arange(0,nrof_images), 1)
            image_paths_array = np.expand_dims(np.array(image_paths), 1)
            control_array = np.zeros_like(labels_array, np.int32)
            if args.use_fixed_image_standardization:
                control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
            sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
            
            embedding_size = int(embeddings.get_shape()[1])
#            assert nrof_images % args.celeba_batch_size == 0, 'The number of Celeba images must be an integer multiple of the Celeba batch size'
            nrof_batches = nrof_images // args.celeba_batch_size
            emb_array = np.zeros((nrof_images, embedding_size))
            lab_array = np.zeros((nrof_images,))
            for i in range(nrof_batches):
                feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.celeba_batch_size}
                emb, lab = sess.run([embeddings, label_batch], feed_dict=feed_dict)
                lab_array[lab] = lab
                emb_array[lab, :] = emb
                if i % 10 == 9:
                    print('.', end='')
                    sys.stdout.flush()
            print('')
            length = nrof_batches * args.celeba_batch_size

            emb_arr = emb_array[:length]
            img_arr = np.array(image_list[:length])
            label_arr = np.array(label_list[:length])

            a = train_test_split(img_arr, label_arr, test_size=0.2, random_state=2018)
            b = train_test_split(emb_arr, label_arr, test_size=0.2, random_state=2018)

            pickle.dump({'img':a[0], 'emb':b[0], 'label':a[2]}, open('celeba_train_set.p', 'wb'))
            pickle.dump({'img':a[1], 'emb':b[1], 'label':a[3]}, open('celeba_test_set.p', 'wb'))
            print(' ')
            


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('celeba_dir', type=str,
        help='Path to the data directory containing aligned Celeba face patches.')
    parser.add_argument('--celeba_batch_size', type=int,
        help='Number of images to process in a batch in the Celeba test set.', default=100)
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
