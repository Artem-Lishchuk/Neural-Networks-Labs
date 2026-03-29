import torch 

def affine_forward(x, w, b):
    
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples,  whereeach example x[i] has shape (d_1, ..., d_k). For example,
    batch of 500 RGB CIFAR-10 images would have shape (500, 32, 32, 3). We 
    will reshape each input into a vector of dimension D = d_1 * ... * d_k,
    and then transform it to an output vector of dimension M.

    Inputs:
    - x: A torch tensor containing input data, of shape (N, d_1, ..., d_k)
    - w: A torch tensor of weights, of shape (D, M)
    - b: A torch tensor of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """

    out = None
    ###########################################################################
    # Task 5.1.1                                                              #
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    input = torch.reshape(x, (x.shape[0], -1)) # shape: (N, d1 * d2 * d3)
    # w shape: (d1 * d2 * d3, M)
    out = input @ w + b# shape: (N, M)

    #                             END OF YOUR CODE                            #
    ##    ###########################################################################
#########################################################################

    cache = (x, w, b)

    return out, cache


def affine_backward(dout, cache):
    
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """

    x, w, b = cache
    dx, dw, db = None, None, None

    ###########################################################################
    # Task 5.1.2                                                              #
    # TODO: Implement the affine backward pass. Do not forget to reshape your #
    # dx to match the dimensions of x.                                        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    assert dx.shape == x.shape, "dx.shape != x.shape: " + str(dx.shape) + " != " + str(x.shape)
    assert dw.shape == w.shape, "dw.shape != w.shape: " + str(dw.shape) + " != " + str(w.shape)
    assert db.shape == b.shape, "db.shape != b.shape: " + str(db.shape) + " != " + str(b.shape)

    return dx, dw, db

def relu_forward(x):

    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """

    out = None

    ###########################################################################
    # Taks 5.1.3                                                              #
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = x

    return out, cache

def relu_backward(dout, cache):

    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """

    dx, x = None, cache

    ###########################################################################
    # Task 5.1.4                                                              #
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx

def dropout_forward(x, dropout_param):

    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """

    p, mode = dropout_param['p'], dropout_param['mode']

    if 'seed' in dropout_param:
        torch.manual_seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':

        #######################################################################
        # Task 5.5.1                                                          #
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        # HINT: http://cs231n.github.io/neural-networks-2/#reg                #
        #######################################################################

        pass

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    elif mode == 'test':

        #######################################################################
        # Task 5.5.1                                                          #
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################

        pass

        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.to(x.dtype)

    return out, cache

def dropout_backward(dout, cache):

    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """

    dropout_param, mask = cache
    p, mode = dropout_param['p'], dropout_param['mode']

    dx = None

    if mode == 'train':

        #######################################################################
        # Task 5.5.2                                                          #
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################

        pass

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    elif mode == 'test':

        dx = dout

    return dx

def softmax_loss(x, y):

    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

    y = y.long() 

    log_probs = x - torch.logsumexp(x, dim=1, keepdim=True)
    probs = torch.exp(log_probs)
    N = x.shape[0]
    loss = -torch.sum(log_probs[torch.arange(N), y]) / N
    dx = probs.clone()
    dx[torch.arange(N), y] -= 1
    dx /= N
    
    return loss, dx