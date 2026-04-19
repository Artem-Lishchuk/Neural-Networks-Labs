import torch

def col2im_6d(cols, N, C, H, W, HH, WW, pad, stride):

    """
    Parameters:
    - cols: (C, HH, WW, N, out_h, out_w)
    - N, C, H, W: shape of original input
    - HH, WW: filter height/width
    - pad: int
    - stride: int

    Output:
    - img: (N, C, H, W)
    """

    H_padded, W_padded = H + 2*pad, W + 2*pad
    img = torch.zeros((N, C, H_padded, W_padded), dtype=cols.dtype, device=cols.device)
    C, HH, WW, N, out_h, out_w = cols.shape

    # Move N to front for easier indexing
    cols = cols.permute(3, 0, 1, 2, 4, 5)  # (N, C, HH, WW, out_h, out_w)

    for y in range(out_h):
        for x in range(out_w):
            h_start = y * stride
            w_start = x * stride
            # This adds into the correct locations (sums-overlapping)
            img[:, :, h_start:h_start+HH, w_start:w_start+WW] += cols[:, :, :, :, y, x]

    # Remove padding if needed
    if pad > 0:
        img = img[:, :, pad:-pad, pad:-pad]
    return img

def conv_forward_strides(x, w, b, conv_param):

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    stride, pad = conv_param['stride'], conv_param['pad']

    # Pad the input
    if pad > 0:
        x_padded = torch.zeros((N, C, H + 2*pad, W + 2*pad), dtype=x.dtype, device=x.device)
        x_padded[:, :, pad:pad+H, pad:pad+W] = x
    else:
        x_padded = x

    # Figure out output dimensions
    H_padded = H + 2 * pad
    W_padded = W + 2 * pad
    out_h = (H_padded - HH) // stride + 1
    out_w = (W_padded - WW) // stride + 1

    # Perform an im2col operation by picking clever strides
    x_cols = []

    for i in range(0, H_padded - HH + 1, stride):
        for j in range(0, W_padded - WW + 1, stride):
            
            patch = x_padded[:, :, i:i+HH, j:j+WW] 
            x_cols.append(patch.reshape(N, -1))   

    x_cols = torch.stack(x_cols, dim=-1)      
    x_cols = x_cols.permute(1, 0, 2).reshape(C*HH*WW, -1) 

    # Reshape filters
    w_row = w.reshape(F, -1)             

    # Matrix multiply
    res = w_row @ x_cols + b.reshape(-1, 1)

    # Reshape to output
    res = res.reshape(F, N, out_h, out_w)
    out = res.permute(1, 0, 2, 3).contiguous()     

    cache = (x, w, b, conv_param, x_cols)

    return out, cache

def conv_backward_strides(dout, cache):

    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = dout.sum(dim=(0, 2, 3))

    dout_reshaped = dout.permute(1, 0, 2, 3).reshape(F, -1)

    dw = dout_reshaped @ x_cols.t()
    dw = dw.reshape(w.shape)

    dx_cols = w.reshape(F, -1).t() @ dout_reshaped
    dx_cols = dx_cols.reshape(C, HH, WW, N, out_h, out_w)

    dx = col2im_6d(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db

def im2col(x, HH, WW, padding=0, stride=1):

    """
    x: (N*C, 1, H, W)
    Returns: (HH*WW, N*out_h*out_w)
    """

    N, C, H, W = x.shape

    x_unf = x.unfold(2, HH, stride).unfold(3, WW, stride)
    out_h = x_unf.size(2)
    out_w = x_unf.size(3)

    cols = x_unf.contiguous().permute(0,1,4,5,2,3).reshape(N*C, HH*WW, out_h*out_w)

    cols = cols.reshape(-1, out_h*out_w)

    return cols

def max_pool_forward_reshape(x, pool_param):

    """
    A fast implementation of the forward pass for the max pooling layer that uses
    some clever reshaping.

    This can only be used for square pooling regions that tile the input.
    """

    N, C, H, W = x.shape
    
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_width == 0

    x_reshaped = x.reshape(
        N, 
        C,
        H // pool_height, 
        pool_height,
        W // pool_width, 
        pool_width
    )

    out = x_reshaped.max(dim=3).values.max(dim=4).values

    cache = (x, x_reshaped, out)

    return out, cache

def max_pool_forward_im2col(x, pool_param):

    """
    An implementation of the forward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """

    N, C, H, W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']

    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_reshaped = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_reshaped, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = x_cols.argmax(dim=0)
    x_cols_max = x_cols[x_cols_argmax, torch.arange(x_cols.shape[1], device=x.device)]

    out = x_cols_max.reshape(out_height, out_width, N, C).permute(2, 3, 0, 1).contiguous()

    cache = (x, x_cols, x_cols_argmax, pool_param)

    return out, cache

def max_pool_forward_fast(x, pool_param):

    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the im2col method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the im2col method, which
    is not much faster than the naive method.
    """

    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0

    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im2col_cache)

    return out, cache

def max_pool_backward_reshape(dout, cache):

    """
    A fast implementation of the backward pass for the max pooling layer that
    uses some clever broadcasting and reshaping.

    This can only be used if the forward pass was computed using
    max_pool_forward_reshape.

    NOTE: If there are multiple argmaxes, this method will assign gradient to
    ALL argmax elements of the input rather than picking one. In this case the
    gradient will actually be incorrect. However this is unlikely to occur in
    practice, so it shouldn't matter much. One possible solution is to split the
    upstream gradient equally among all argmax elements; this should result in a
    valid subgradient. You can make this happen by uncommenting the line below;
    however this results in a significant performance penalty (about 40% slower)
    and is unlikely to matter in practice so we don't do it.
    """

    x, x_reshaped, out = cache

    dx_reshaped = torch.zeros_like(x_reshaped)
    out_newaxis = out.unsqueeze(3).unsqueeze(5)
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout.unsqueeze(3).unsqueeze(5)
    dx_reshaped[mask] = dout_newaxis.expand_as(dx_reshaped)[mask]
    dx_reshaped /= mask.sum(dim=(3, 5), keepdim=True) # <-----
    dx = dx_reshaped.reshape(x.shape)

    return dx

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1, device=None):
    
    # First figure out what the size of the output should be

    N, C, H, W = x_shape

    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0

    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = torch.repeat_interleave(torch.arange(field_height, device=device), field_width)
    i0 = i0.repeat(C)
    i1 = stride * torch.repeat_interleave(torch.arange(out_height, device=device), out_width)
    j0 = torch.tile(torch.arange(field_width, device=device), (field_height * C,))
    j1 = stride * torch.tile(torch.arange(out_width, device=device), (out_height,))
    i = i0.view(-1, 1) + i1.view(1, -1)
    j = j0.view(-1, 1) + j1.view(1, -1)
    
    k = torch.repeat_interleave(torch.arange(C, device=device), field_height * field_width).view(-1, 1)

    return k.long(), i.long(), j.long()

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):

    """ An implementation of col2im based on fancy indexing and np.add.at """

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = torch.zeros((N, C, H_padded, W_padded), dtype=cols.dtype, device=cols.device)

    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride, device=cols.device)

    cols_reshaped = cols.view(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.permute(2, 0, 1)

    batch_indices = torch.arange(N, device=cols.device).view(-1, 1, 1).expand_as(cols_reshaped)

    x_padded.index_put_((batch_indices, k, i, j), cols_reshaped, accumulate=True)

    if padding == 0:
        return x_padded
    
    return x_padded[:, :, padding:-padding, padding:-padding]

def max_pool_backward_im2col(dout, cache):

    """
    An implementation of the backward pass for max pooling based on im2col.

    This isn't much faster than the naive version, so it should be avoided if
    possible.
    """

    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape

    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    out_h = (H - pool_height) // stride + 1
    out_w = (W - pool_width) // stride + 1

    dout_reshaped = dout.permute(2, 3, 0, 1).reshape(-1)

    dx_cols = torch.zeros_like(x_cols)
    dx_cols[x_cols_argmax, torch.arange(dx_cols.shape[1], device=x.device)] = dout_reshaped

    dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride)
    dx = dx.reshape(x.shape)

    return dx

def max_pool_backward_fast(dout, cache):

    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """

    method, real_cache = cache

    if method == 'reshape':
        return max_pool_backward_reshape(dout, real_cache)  # your PyTorch implementation
    elif method == 'im2col':
        return max_pool_backward_im2col(dout, real_cache)   # your PyTorch implementation
    else:
        raise ValueError(f'Unrecognized method "{method}"')