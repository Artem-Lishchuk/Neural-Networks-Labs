import torch

def eval_numerical_gradient(f, x, verbose=True, h=1e-5):
    
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = torch.zeros_like(x)

    # iterate over all indexes in x
    with torch.no_grad():
        for i in range(x.numel()):

            ix = torch.unravel_index(torch.tensor(i), x.shape)

            oldval = x[ix].item()

            x[ix] = oldval + h       # increment by h (in-place on the ORIGINAL tensor)
            fxph = f(x)              # evaluate f(x + h)

            x[ix] = oldval - h
            fxmh = f(x)              # evaluate f(x - h)

            x[ix] = oldval           # restore original value

            # compute the partial derivative with centered formula
            grad[ix] = (fxph - fxmh) / (2 * h)

            if verbose:
                print(ix, grad[ix].item())

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):

    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """

    grad = torch.zeros_like(x)

    x_flat = x.detach().flatten()
    grad_flat = grad.flatten()

    with torch.no_grad():
        for i in range(x_flat.numel()):

            oldval = x_flat[i].item()

            x_flat[i] = oldval + h
            pos = f(x_flat.reshape(x.shape)).clone()

            x_flat[i] = oldval - h
            neg = f(x_flat.reshape(x.shape)).clone()

            x_flat[i] = oldval

            grad_flat[i] = torch.sum((pos - neg) * df) / (2 * h)

    grad = grad_flat.reshape(x.shape)

    return grad