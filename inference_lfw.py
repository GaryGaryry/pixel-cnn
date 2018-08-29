import os
import numpy as np
import json
import argparse
import pickle
import tensorflow as tf
from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='/local_home/tim/pxpp/data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='/local_home/tim/pxpp/save', help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str, default='cifar', help='Can be either cifar|imagenet|lfw')
parser.add_argument('-t', '--save_interval', type=int, default=20, help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=8, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

if args.data_set != 'lfw':
    raise("unsupported dataset")
if args.class_conditional:
    raise("unsupported no-class_conditional")

import data.lfw_data as lfw_data
DataLoader = lfw_data.DataLoader

sample_data = DataLoader(args.data_dir, 'sample', args.batch_size * args.nr_gpu,
                         shuffle=False, return_labels=args.class_conditional)
obs_shape = sample_data.get_observation_size() # e.g. a tuple (32,32,3)

xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
hs = [tf.placeholder(tf.float32, shape=(args.init_batch_size,) + sample_data.h.shape[1:]) for i in range(args.nr_gpu)]

model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance }
model = tf.make_template('model', model_spec)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]
# sample
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        out = model(xs[i], hs[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix))

def sample_from_model(data, using_original_img=False):
    x_origin, y_origin, h_origin = data
    x = x_origin.copy()

    if(not using_original_img):
        x = np.array([np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)])
    else:
        x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]

    x = np.split(x, args.nr_gpu)
    feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
    if h_origin is not None:
        h = np.split(h_origin, args.nr_gpu)
        feed_dict.update({hs[i]: h[i] for i in range(args.nr_gpu)})

    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, feed_dict)
            for i in range(args.nr_gpu):
                x[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    return np.concatenate(x, axis=0), x_origin, y_origin


saver = tf.train.Saver()
with tf.Session() as sess:
    if not args.load_params:
        raise("unsupported no pretrained model")
    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    sample_list = []
    img_list = []
    label_list = []
    for data in sample_data:
        sample_x, origin_imgs, origin_labels = sample_from_model(sess)
        sample_list.append(sample_x)
        img_list.append(origin_imgs)
        label_list.append(origin_labels)

    pickle.dump({'sample': sample_list, 'img': img_list, 'label': label_list}, open('experiment_a_%s_sample_inference.p' % (args.data_set), "wb"))
