# Multi-digit-number-sequence-prediciton-using-deep-neural-networks

Machine-Learning-Project

Project done in partial fulfilemt of Machine Learning (BITS F464) course. Multi digit number sequence prediction on Google Street View House Numbers dataset.

Tried to implement the paper by Ian Goodfellow[1] to detect number sequences in the Google SVHN dataset[2]. Used Convolution Neural Networks to learn intricate details present in the given images to improve prediciton accuracy. The real challenge was to create a model that could accomodate varying sequence lengths.

Basic model had the following layout:

4 convolution layers followed by one fully connected layer and then 7 softmax classifiers. One for predciting the number of digits in the image and the other 6 to predict a particular digit. Implicit assumption that sequence won't be of more than 6 digits. Max pooling was done at the 2nd,4th and 5th convolution layers.

More details can be found in the document. 

[1] https://arxiv.org/pdf/1312.6082.pdf [2] http://ufldl.stanford.edu/housenumbers/
