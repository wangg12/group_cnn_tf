* http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007, Rotated MNIST digits

* http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
* 56M packed; 56M + 338M unpacked 12000 train, 50000 test

* mnist_all_rotation_normalized_float_train_valid.amat: 12000x785, the last col is label 
* mnist_all_rotation_normalized_float_test.amat: 50000x785 
* load by `np.loadtxt(filename)`, delimiter=' '