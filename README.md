# ece496
ECE496 Capstone Project
Computer Vision-based Observer
Team: Luis Camero, Ki-Seok Hong, Lukasz Jagodzinski

The team is investigating the application of a control systems approach to cooking food. The primary focus is on the computer vision and neural network estimation of the food state at all arbitrary times in the cooking process. Several architectures were investigated, primarily making use of a pre-trained CNN (VGG16) and a classifier to interpret the extracted features from the CNN. Feasibility of RNNs as classifiers was tested, using an LSTM model.

Computer vision camera capture is written in C++. Neural network models, feature extraction, and training are implemented in Python using PyTorch framework.
