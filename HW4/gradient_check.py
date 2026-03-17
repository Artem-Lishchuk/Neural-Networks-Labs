import torch

def eval_numerical_gradient(f, x, verbose=True, h=2e-5):
    
    """ 
    a naive implementation of numerical gradient of f at x 
    - f should be a function that takes a single argument
    - x is the point to evaluate the gradient at
    """ 
    
    x = x.detach().clone()

    fx = f(x) # evaluate function value at original point
    grad = torch.zeros_like(x)

    # iterate over all indexes in x
    with torch.no_grad():
        for i in range(x.numel()):

            ix = torch.unravel_index(torch.tensor(i), x.shape)

            oldval = x[ix].item()

            x[ix] = oldval + h # increment by h
            fxph = f(x) # evalute f(x + h)

            x[ix] = oldval - h
            fxmh = f(x) # evaluate f(x - h)

            x[ix] = oldval  # restore

            # compute the partial derivative with centered formula
            grad[ix] = (fxph - fxmh) / (2 * h) # the slope
            
            if verbose:
                print(ix, grad[ix].item())

    return grad