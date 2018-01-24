import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

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
    - cache: (x, w, b, conv_param) for the backward pass
    """
    #out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    (N, C, H, W) = x.shape
    (F, C, HH, WW) = w.shape
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride

    out = np.zeros((N, F, Hp, Wp))

    # Add padding around each 2D image
    padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

    for i in range(N): # ith example
        for j in range(F): # jth filter

              # Convolve this filter over each input
            for k in range(Hp):
                hs = k * stride
                for l in range(Wp):
                    ws = l * stride

                    # Window we want to apply the respective jth filter over (C, HH, WW)
                    window = padded[i, :, hs:hs+HH, ws:ws+WW]

          
                    out[i, j, k, l] = np.sum(window*w[j]) + b[j]
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
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
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    Hp = 1 + (H + 2 * pad - HH) // stride
    Wp = 1 + (W + 2 * pad - WW) // stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

  # Add padding around each 2D image (and respective gradient)
  # There may be a prettier way to do this but I can't think of any nice way at
  # least. You want to contribute to the boundary sums and in some cases the
  # only way to do that is by writing into the padding. I'm sure some very nasty
  # indexing trick will do; with lots of floors and ceils.
    padded = np.pad(x, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')
    padded_dx = np.pad(dx, [(0,0), (0,0), (pad,pad), (pad,pad)], 'constant')

    for i in range(N): # ith example
        for j in range(F): # jth filter
      # Convolve this filter over windows
            for k in range(Hp):
                hs = k * stride
                for l in range(Wp):
                    ws = l * stride

          # Window we applies the respective jth filter over (C, HH, WW)
                    window = padded[i, :, hs:hs+HH, ws:ws+WW]

          # Compute gradient of out[i, j, k, l] = np.sum(window*w[j]) + b[j]
                    db[j] += dout[i, j, k, l]
                    dw[j] += window*dout[i, j, k, l]
                    padded_dx[i, :, hs:hs+HH, ws:ws+WW] += w[j] * dout[i, j, k, l]

  # "Unpad"
    dx = padded_dx[:, :, pad:pad+H, pad:pad+W]
    return dx, dw, db
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
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
    - cache: (x, maxIdx, pool_param) for the backward pass with maxIdx, of shape (N, C, H, W, 2)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) // stride
    Wp = 1 + (W - WW) // stride

    out = np.zeros((N, C, Hp, Wp))

    for i in range(N):
    # Need this; apparently we are required to max separately over each channel
      for j in range(C):
        for k in range(Hp):
            hs = k * stride
            for l in range(Wp):
                ws = l * stride

          # Window (C, HH, WW)
                window = x[i, j, hs:hs+HH, ws:ws+WW]
                out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)
    return out, cache
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, maxIdx, pool_param)
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
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    x, pool_param = cache
    HH = pool_param['pool_height']
    WW = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    Hp = 1 + (H - HH) // stride
    Wp = 1 + (W - WW) // stride

    dx = np.zeros_like(x)

    for i in range(N):
        for j in range(C):
            for k in range(Hp):
                hs = k * stride
                for l in range(Wp):
                    ws = l * stride

          # Window (C, HH, WW)
                    window = x[i, j, hs:hs+HH, ws:ws+WW]
                    m = np.max(window)

          # Gradient of max is indicator
                    dx[i, j, hs:hs+HH, ws:ws+WW] += (window == m) * dout[i, j, k, l]

    return dx

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        check = 1
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    elif mode == 'test':
        check = 2
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

   # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return dx, dgamma, dbeta


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
    """
    out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return dx, dgamma, dbeta
