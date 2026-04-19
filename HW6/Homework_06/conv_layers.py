import torch

def conv_forward_naive(x, w, b, conv_param):

    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    ###########################################################################
    # Task 5.1                                                                #
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, w, b, conv_param) # store the input data and parameters for backpropagation

    return out, cache

def conv_backward_naive(dout, cache):

    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    ###########################################################################
    # Task 6.3                                                                # 
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dw, db

def max_pool_forward_naive(x, pool_param):

    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """

    ###########################################################################
    # Task 6.4                                                                #
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, pool_param)

    return out, cache

def max_pool_backward_naive(dout, cache):

    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    ###########################################################################
    # Task 6.5                                                                #
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx