import torch
from softmax import *
import numpy as np

class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A torch tensor of shape (N, D) containing training data; there are N
            training samples each of dimension D.
        - y: A torch tensor of shape (N,) containing training labels; y[i] = c
            means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """

        device = X.device
        num_train, dim = X.shape
        
        num_classes = torch.max(y).item() + 1  # assume y takes values 0...K-1 where K is number of classes
        
        if self.W is None:
            # Lazily initialize W on the same device as data
            self.W = 0.001 * torch.randn(dim, int(num_classes), device=device, dtype=X.dtype)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO Task 3.5:                                                        #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            indices_train_sample = np.random.choice(num_train, size=batch_size, replace=True)
            X_batch = X[indices_train_sample, :]
            y_batch = y[indices_train_sample]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # Evaluate loss and gradient
            loss, grad = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            self.W -= learning_rate * grad

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss.item():f}')

            # check for NaN or Inf in loss to catch exploding gradients
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Exploded at iteration {it}")
                print(f"Max grad value: {torch.max(torch.abs(grad))}")
                print(f"Max W value: {torch.max(torch.abs(self.W))}")
                break

        return loss_history

    def predict(self, X):
        

        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A torch tensor of shape (N, D) containing training data; there are N
            training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a torch tensor of length N, and each element is an integer giving the predicted
            class.
        """

        ###########################################################################
        # TODO Task 3.6:                                                          #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        z_pred = X @ self.W
        # z_pred_normalized = z_pred - z_pred.max(dim=1, keepdim=True)[0]
        # p_pred = torch.exp(z_pred) / torch.sum(torch.exp(z_pred), dim=1, keepdim=True)
        # y_pred = torch.argmax(p_pred, dim=1)
        y_pred = torch.argmax(z_pred, dim=1)

        return y_pred

class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        # This calls your previously implemented vectorized torch function
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)