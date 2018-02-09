# MNIST_Neon

MNIST is a computer vision dataset consisting of 70,000 images of handwritten digits. Each image has 28x28 pixels for a total of 784 features, and is associated with a digit between 0-9.

# MNIST_01_default
I had constructed a multi-layer perceptron (also called softmax regression) to recognize each image. I used the Neon framework from Intel. I had imported the MNIST dataset and partitioned it into train and test data.

I used Gaussian distribution to initialize weights, as it is a simple approach. 
The activation function - Softmax was used since it is common for classifications and the output is determined as a Probability Distribution. I had used Cross Entropy Loss as a cost function since it has a higher probability of getting better results than Misclassification Rate.

The optimization technique used to reduce cost was Stochastic Gradient Descent with Momentum. Further, I had trained the model for 10 epochs and evaluated the results using Misclassification Rate.

<b> The Misclassification Error was 15.6 % </b>

Further, I had tested the model using a sample image. The model had correctly classified the image.


# MNIST_02_tune_hyperparameters

Here, I had altered the learning rate from 0.1 to 0.001 and increased the batch_size to 64.

<b> I got an improvement, as the Misclassification Error was 9.0% </b>
