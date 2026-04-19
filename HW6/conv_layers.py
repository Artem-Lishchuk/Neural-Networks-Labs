from sympy.parsing.sympy_parser import null
import torch
import numpy as np

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
    images = []
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']
    
    for image in x:
      image = np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant')
      filters = []
      for (filter, bias) in zip(w,b):

        H_conv = (H + 2 * pad - HH) // stride + 1
        W_conv = (W + 2 * pad - WW) // stride + 1

        image_convoluted = torch.empty((H_conv, W_conv), dtype = torch.float64)
        r_conv, c_conv = 0, 0
        for r in range(0, H + 2 * pad - HH + 1, stride):
          for c in range(0, W + 2 * pad - WW + 1, stride):
            x_batch = image[:, r: r + HH, c : c + WW] 
            # scalar = filter.T @ x_batch + bias
            scalar = torch.sum(filter * x_batch, dtype = torch.float64) + bias
            image_convoluted[r_conv][c_conv] = scalar
            c_conv += 1

          c_conv = 0
          r_conv += 1

        filters.append(image_convoluted)
        filters_tensor = torch.stack(filters)

      images.append(filters_tensor)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    cache = (x, w, b, conv_param) # store the input data and parameters for backpropagation
    out = torch.stack(images) # shape: (N, F, H_conv, W_conv)
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

    x, w, b, conv_param = cache

    ###########################################################################
    # Task 6.3                                                                # 
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    images = []
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    pad = conv_param['pad']
    stride = conv_param['stride']

    db = dout.sum(dim=(0, 2, 3))
    dw = torch.zeros_like(w)    
    dx = torch.zeros_like(x)
    
    for n, (image_input, image_grad) in enumerate(zip(x, dout)):
      image = np.pad(image_input, ((0, 0), (pad, pad), (pad, pad)), 'constant')
      image = torch.from_numpy(image)
      dx_pad = torch.zeros(C, H + 2*pad, W + 2*pad, dtype=x.dtype)
      
      for f, (filter, bias, filter_grad) in enumerate(zip(w,b, image_grad)):
        H_conv = (H + 2 * pad - HH) // stride + 1
        W_conv = (W + 2 * pad - WW) // stride + 1

        for r in range(H_conv):
          for c in range(W_conv):
            r0 = r * stride
            c0 = c * stride
            x_batch = image[:, r0:r0+HH, c0:c0+WW]
            dw[f] += filter_grad[r][c] * x_batch
            dx_pad[:, r0:r0+HH, c0:c0+WW] += filter_grad[r][c] * filter
      
      dx[n] = dx_pad[:, pad:pad+H, pad:pad+W]
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
    images = []
    N, C, H, W = x.shape
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']

    for n, image in enumerate(x):
      H_out = (H - pool_height) // stride + 1
      W_out = (W - pool_width) // stride + 1
      image_pooled = torch.zeros((C, H_out, W_out), dtype = x.dtype)

      for r in range(H_out):
        for c in range(W_out):
          r0 = r * stride
          c0 = c * stride
          batch = image[:, r0:r0+ pool_height, c0:c0 + pool_width]
          image_pooled[:, r , c] = torch.Tensor.amax(batch, dim = (1,2))
      
      images.append(image_pooled)

    out = torch.stack(images)
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