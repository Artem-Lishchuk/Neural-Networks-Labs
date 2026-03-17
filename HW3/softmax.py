import torch
import torch.nn.functional as F

def softmax_loss_naive(W, X, y, reg):

    """

    Softmax (cross-entropy) loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A torch tensor of shape (D, C) containing weights.
    - X: A torch tensor of shape (N, D) containing a minibatch of data.
    - y: A torch tensor of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
        You might or might not want to transform it into one-hot form (not obligatory)
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """

    y = y.to(torch.long) 
    y_onehot = F.one_hot(y, num_classes=W.shape[1])
    y_onehot = y_onehot.to(torch.float64)

    W = W.to(torch.float64)
    X = X.to(torch.float64)

    loss = torch.tensor(0.0, dtype=torch.float64, device=W.device)
    dW = torch.zeros_like(W, dtype=torch.float64)
    num_train = X.shape[0]

    # In this naive implementation we have a for loop over the N samples
    for i in range(num_train):

        #############################################################################
        # TODO Task 3.1: 
        # Compute the cross-entropy loss using explicit loops and store the   #
        # sum of losses in "loss". If you already understand the process well       #
        # and are familiar with vectorized operations, you can solve this task      #
        # without inner loops and use vectorized operations instead.                #
        # PS! But in this case still keep the outer loop that enumerates over X!    #
        # If you are not careful in implementing softmax, it is easy to run into    #
        # numeric instability, because exp(a) is huge if a is large.                #
        # Read the Practical issues: numeric stability section from here:           #
        # https://cs231n.github.io/linear-classify/#softmax-classifier              #
        #############################################################################
        
        # softmax 
        z = X[i].unsqueeze(0) @ W
        z_normalized = z - torch.max(z)
        p = torch.exp(z_normalized) / torch.sum(torch.exp(z_normalized))
        # loss: -log(p) of the correct class
        loss += torch.sum(-y_onehot[i] * torch.log(p + 1e-10))
        #############################################################################
        # TODO Task 3.3:                                                            #
        # Compute the gradient using explicit loops and store the sum over          #
        # samples in dW. Again, you are allowed to use vectorized operations        #
        # if you know how to.                                                       #
        #############################################################################
        dL_dz = p - y_onehot[i]
        dz_dW = X[i].unsqueeze(1)
        dW += dz_dW @ dL_dz

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    # Average over training samples
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * torch.sum(W * W)
    dW += 2 * reg * W 

    return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
    

    """
    Softmax (cross-entropy) loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    W = W.to(torch.float64)
    X = X.to(torch.float64)
    y = y.to(torch.long)

    y_onehot = F.one_hot(y, num_classes=W.shape[1])
    y_onehot = y_onehot.to(torch.float64)
    
    num_train = X.shape[0]

    #############################################################################
    # TODO Task 3.4:                                                            # 
    # Compute the cross-entropy loss and its gradient using no loops.           #
    # Store the loss in loss and the gradient in dW.                            #
    # Make sure you take the average.                                           #
    # If you are not careful with softmax, you migh run into numeric instability#
    #############################################################################
    
    # Compute probabilities
    z = X @ W
    z_normalized = z - z.max(dim=1, keepdim=True)[0]
    p = torch.exp(z_normalized) / torch.sum(torch.exp(z_normalized), dim=1, keepdim=True)

    loss = torch.sum(-y_onehot * torch.log(p + 1e-10)) / num_train

    dL_dz = p - y_onehot
    dz_dW = X

    dW = dz_dW.T @ dL_dz
    dW = dW / num_train

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    # Add regularization to the loss and gradients.
    loss += reg * torch.sum(W * W)
    dW += 2 * reg * W

    return loss, dW