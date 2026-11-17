# Neural-Network-from-Scratch
A basic neural network trained on MNIST database, which is written only using numpy and python. Network achieves 92% test accuracy.

Methodology:
+ A standard Multilayer Perceptron network.
+ Takes in images from MNIST database of size 28x28 pixels
+ The largest value in the output layer is the one that gets picked as the model's prediction.
+ Model uses MSE as the cost function.
+ Uses AdamW as the optimizer (also custom written)
+ Uses RELU as the activation function in between layers
+ Uses He Uniform Initialization to create the initial state of the weights.
+ Model trains on 50% of database, and the other 50% it tests on.

Libraries used:
+ numpy: to do matrix multiplications, etc.
+ torchvision: to import the MNIST database and split it
+ Pillow (PIL): to import custom images that you draw in MS Paint.

Work:
+ written by GeniusDog3103
