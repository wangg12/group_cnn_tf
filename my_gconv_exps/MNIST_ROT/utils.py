from __future__ import print_function, absolute_import, division

import logging
import os
import time
import numpy as np


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

def create_result_dir(model_name, logme=False, result_rootdir='./results/'):
    '''
    logme: bool, if True, use logging.info()
    ----
    log_fname:
    result_dir:
    '''
    # if args.restart_from is None:
    result_dir = os.path.join(result_rootdir, model_name, time.strftime('r%Y_%m_%d_%H_%M_%S'))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fname = '{}/log.txt'.format(result_dir)

    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fname, level=logging.DEBUG)
    logging.info(logme)

    # # Create init file so we can import the model module
    # f = open(os.path.join(result_dir, '__init__.py'), 'wb')
    # f.close()

    return log_fname, result_dir