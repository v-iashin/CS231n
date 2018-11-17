# CS231n

[CS231n: "Convolutional Neural Networks for Visual Recognition"](http://cs231n.stanford.edu/)

My solutions to the assignments to the state-of-the-art course CS231n "Convolutional Neural Networks for Visual Recognition". It was hard, but it is cool.

## Framework
During the course, there was a choice between two frameworks: **TensorFlow** and **PyTorch**. ~~I decided to follow the **TensorFlow** track. Therefore no solution is provided for PyTorch. However, it might occur someday in the future.~~ Now, the solutions are provided for both frameworks.

## Content of the Assignments (Spring 2017)
There were three assignments during the Spring 2017 version of the course. They all are completed.

1. [[Assignment #1]](http://cs231n.github.io/assignments2017/assignment1/)
- understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- develop proficiency in writing efficient **vectorized** code with numpy
- implement and apply a k-Nearest Neighbor (**kNN**) classifier
- implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- implement and apply a **Softmax** classifier
- implement and apply a **Two layer neural network** classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using **higher-level representations** than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

2. [[Assignment #2]](http://cs231n.github.io/assignments2017/assignment2/)
- understand **Neural Networks** and how they are arranged in layered architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- effectively **cross-validate** and find the best hyperparameters for Neural Network architecture
- understand the architecture of **Convolutional Neural Networks** and train gain experience with training these models on data

3. [[Assignment #3]](http://cs231n.github.io/assignments2017/assignment3/)
- understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
- understand and implement both **Vanilla RNNs** and **Long-Short Term Memory (LSTM) RNNs**
- understand how to **sample** from an RNN language model at test-time
- understand how to combine convolutional neural nets and recurrent nets to implement an **image captioning** system
- understand how a trained convolutional network can be used to compute gradients with respect to the input image
- implement and different applications of image gradients, including **saliency maps**, **fooling images**, **class visualizations**
- understand and implement **style transfer**
- understand how to train and implement a generative adversarial network (**GAN**) to produce images that look like a dataset
