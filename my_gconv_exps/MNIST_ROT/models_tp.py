from __future__ import division, absolute_import, print_function
import os
import tensorflow as tf
from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
import numpy as np
from skimage.transform import rotate

from models_tf import z2_cnn, p4_cnn, p4m_cnn, p4_cnn_rp






class Model(ModelDesc):
    def __init__(self, name='Z2CNN'):
        super(Model, self).__init__()
        self.name = name

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 28, 28, 1), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        x, y_ = inputs
        is_train = get_current_tower_context().is_training
        if self.name == 'Z2CNN':
            logits = z2_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False)
        elif self.name == 'P4CNN':
            logits = p4_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False)
        elif self.name == 'P4MCNN':
            logits = p4m_cnn(x, y_ , is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False)
        elif self.name == 'P4CNN_RP':
            logits = p4_cnn_rp(x, y_, is_train, kernel_size=3, stride=1, drop_rate=0.3, reuse=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
        loss = tf.reduce_mean(loss, name='cross_entropy')
        self.cost = loss

        y_one_hot = tf.one_hot(y_, depth=10, on_value=1, off_value=0, dtype=tf.int32)
        # accuracy = tf.metrics.accuracy(labels=y_one_hot, predictions=logits, name='accuracy')
        # dont know what is update_op
        # correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        preds = tf.cast(tf.argmax(logits, 1), tf.int32)
        acc = tf.contrib.metrics.accuracy(predictions=preds, labels=y_, name='accuracy')
        # accuracy/Mean:0
        acc = tf.identity(acc, name='acc')
        # print(acc.name)
        
        num_params = np.sum([np.prod(v.get_shape().as_list())
                                    for v in tf.trainable_variables()])
        print('num_params: {}'.format(num_params))

        

        summary.add_moving_summary(loss)
        summary.add_moving_summary(acc)
        # monitor histogram of all weight (of conv and fc layers) in tensorboard
        summary.add_param_summary(('.*/kernel', ['histogram', 'rms']))


    def _get_optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=get_global_step_var(),
            decay_steps=468 * 10,
            decay_rate=0.3, staircase=True, name='learning_rate')
        tf.summary.scalar('lr', lr)
        opt = tf.train.AdamOptimizer(lr)
        return opt


def preprocess_mnist_data(train_data, test_data,
                          train_labels, test_labels,
                          data_format='NHWC'):
    ''' normalize the data and set the data type'''
    train_mean = np.mean(train_data)  # compute mean over all pixels make sure equivariance is preserved
    train_data -= train_mean
    test_data -= train_mean

    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std

    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    if data_format == 'NHWC':
        # print('original shape: {}'.format(train_data.shape))
        # print('convert to NHWC format...')
        if train_data.shape[1] > 3 or test_data.shape[1] >3:
            print('wrong provided data!!!!')
        train_data = train_data.transpose((0, 2, 3, 1))
        test_data = test_data.transpose((0, 2, 3, 1))
        # print('new shape: {}'.format(train_data.shape))

    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    return train_data, test_data, train_labels, test_labels


class MnistRot(RNGDataFlow):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    def __init__(self, train_or_test, train_path, test_path, shuffle=True):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        
        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        self.shuffle = shuffle
        train_set = np.load(train_path) # saved by mnist_rot.py
        val_set = np.load(test_path)
        train_data = train_set['data']
        train_labels = train_set['labels']
        val_data = val_set['data']
        val_labels = val_set['labels']

        train_data, val_data, train_labels, val_labels = preprocess_mnist_data(
                                        train_data, val_data,
                                        train_labels, val_labels,
                                        data_format='NHWC')

        if self.train_or_test == 'train':
            self.images, self.labels = train_data, train_labels
            
        else:
            self.images, self.labels = val_data, val_labels

    def size(self):
        return self.images.shape[0]

    def get_data(self):
        idxs = list(range(self.size()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k]
            label = self.labels[k]
            yield [img, label]


class RandomRotate(imgaug.ImageAugmentor):
    def __init__(self, rotation=None):
        '''rotation: the degree to rotation, in [0, 2*np.pi]
        result in rotating randomly in 0.5*[-rotation, rotation]
        '''
        self._init(locals())

    def _get_augment_params(self, _):
        # generated random params with self.rng
        # return self._rand_range(*self.range)
        return self.rng.uniform(-0.5*self.rotation, 0.5*self.rotation)

    def _augment(self, img, r):
        # hack; skimage.transform wants float images to be in [-1, 1]
        x = img
        factor = np.maximum(np.max(x), np.abs(np.min(x)))
        x = x / factor

        x_out = np.empty_like(x) # HWC format
        x_out = rotate(x, r)

        x_out *= factor
        return x_out


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    if isTrain:
      ds = MnistRot(train_or_test, shuffle=True, train_path='mnist_rotation_new/train_all.npz', test_path='mnist_rotation_new/test.npz')
    else:
      ds = MnistRot(train_or_test, shuffle=False, train_path='mnist_rotation_new/train_all.npz', test_path='mnist_rotation_new/test.npz')  
    if isTrain:
        augmentors = [
            RandomRotate(2*np.pi)
        ]
    else:
      augmentors = []
    
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder = not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds



def my_sess_config(mem_fraction=1./6, allow_growth=False):
    """
    Return a better session config to use as default.
    Tensorflow default session config consume too much resources.

    Args:
        mem_fraction(float): fraction of memory to use.
    Returns:
        tf.ConfigProto: the config to use.
    """
    conf = tf.ConfigProto()

    conf.allow_soft_placement = True
    # conf.log_device_placement = True

    # https://github.com/tensorflow/tensorflow/issues/9322#issuecomment-295758107
    # can speed up a bit
    conf.intra_op_parallelism_threads = 1
    conf.inter_op_parallelism_threads = 0

    conf.gpu_options.per_process_gpu_memory_fraction = mem_fraction
    conf.gpu_options.allocator_type = 'BFC'
    conf.gpu_options.allow_growth = allow_growth
    # force gpu compatible?

    conf.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    return conf


def get_config(model_name='Z2CNN', log_dir='./train_log'):
    # logger.auto_set_dir()
    logger.set_logger_dir(dirname=log_dir)
    dataset_train, dataset_test = get_data('train'), get_data('test')
    return TrainConfig(
        model=Model(name=model_name),
        session_creator=sesscreate.NewSessionCreator(config=my_sess_config(mem_fraction=1./6, allow_growth=False)),  #tfutils.get_default_sess_config(mem_fraction=1./6)),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(
                dataset_test,
                [ScalarStats('cross_entropy'), ScalarStats('acc')]),
        ],
        max_epoch=300,
    )

   
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model_name',   default='Z2CNN',
                        choices=['Z2CNN', 'P4CNN', 'P4MCNN', 'P4CNN_RP'],
                        help='model name')
    parser.add_argument('--extra_name', default='',  help='extra experiment name')
    args = parser.parse_args()

    # model_name = 'P4MCNN' 
    # model_name = 'P4CNN_RP'
    model_name = args.model_name # 'Z2CNN'
    if args.extra_name != '':
        extra_name = '_' + args.extra_name
    else:
        extra_name = ''
    config = get_config(model_name=model_name, 
                        log_dir=os.path.join('./train_log', model_name + extra_name))
    # if args.load:
    #     config.session_init = SaverRestore(args.load)
    SimpleTrainer(config).train()
