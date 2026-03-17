from numpy import dtype, float64
import torch
import torch.nn.functional as F

class TwoLayerNet(object):

    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """

        self.params = {}

        # Initialize as float64 to match NumPy's default precision
        self.params['W1'] = std * torch.randn((input_size, hidden_size), dtype=torch.float64)
        self.params['b1'] = torch.zeros(hidden_size, dtype=torch.float64)
        self.params['W2'] = std * torch.randn((hidden_size, output_size), dtype=torch.float64)
        self.params['b2'] = torch.zeros(output_size, dtype=torch.float64)

    def loss(self, X, y=None, reg=0.0):

        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
        an integer in the range 0 <= y[i] < C. This parameter is optional; if it
        is not passed then we only return scores, and if it is passed then we
        instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
        samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
        """

        # force everything to float64 for mathematical parity
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch.float64)
        else:
            X = X.to(torch.float64)

        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # Task 4.1                                                                  #
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        A = X @ W1 + b1
        H = torch.relu(A)
        scores = H @ W2 + b2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores
        
        one_hot_y = F.one_hot(y, num_classes=scores.shape[1]).to(scores.dtype)

        # Compute the loss
        #############################################################################
        # Task 4.2                                                                  #
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        
        P = torch.softmax(scores, dim=1)
        loss = torch.sum(-one_hot_y * torch.log(P)) / N

        loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}

        #############################################################################
        # Task 4.3                                                                  #
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        dL_dscores = (P - one_hot_y) / N # shape : (N, C)

        dscores_dW2 = H.T # shape : (H, N)
        dP_db = 1
        
        dL_dW2 = dscores_dW2 @ dL_dscores 
        dL_db2 = torch.sum(dL_dscores, dim = 0)

        grads['W2'] = dL_dW2 + 2 * reg * W2
        grads['b2'] = dL_db2

        dscores_dH = W2.T # shape: (C, H)
        dL_dH = dL_dscores @ dscores_dH # shape: (N, H)
        
        dL_dA = dL_dH * torch.where(A > 0, 1, 0).to(A.dtype) # shape: (N, H)

        dA_dW1 = X.T # (shape, D, N)
        dA_db1 = 1

        dL_dW1 = dA_dW1 @ dL_dA # shape: (D, H)
        dL_db1 = torch.sum(dL_dA, dim = 0)  

        grads['W1'] = dL_dW1 + 2 * reg * W1
        grads['b1'] = dL_db1


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A torch tensor of shape (N, D) giving training data.
        - y: A torch tensor f shape (N,) giving training labels; y[i] = c means that
        X[i] has label c, where 0 <= c < C.
        - X_val: A torch tensor of shape (N_val, D) giving validation data.
        - y_val: A torch tensor of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
        after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        
        # Ensure data is correctly typed
        if not isinstance(X, torch.Tensor): X = torch.from_numpy(X).to(torch.float64)
        if not isinstance(y, torch.Tensor): y = torch.from_numpy(y).long()
        if not isinstance(X_val, torch.Tensor): X_val = torch.from_numpy(X_val).to(torch.float64)
        if not isinstance(y_val, torch.Tensor): y_val = torch.from_numpy(y_val).long()

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        val_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):

            #########################################################################
            # Task 4.4                                                              #
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            perm = torch.randperm(num_train)
            batch_indices = perm[:batch_size]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss.item())

            #########################################################################
            # Task 4.4                                                              #
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']

            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and it % 100 == 0:
                print(f'iteration {it} / {num_iters}: loss {loss.item():.6f}')

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).float().mean().item()
                val_acc = (self.predict(X_val) == y_val).float().mean().item()
                
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                
                val_loss, _ = self.loss(X_val, y=y_val, reg=reg)
                val_loss_history.append(val_loss.item())

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'val_loss_history': val_loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):

        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
        to have class c, where 0 <= c < C.
        """

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        
        # Ensure input is float64
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).to(torch.float64)

        scores = self.loss(X)
        y_pred = scores.argmax(dim = 1)

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred