import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

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
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  out_height = 1 + (H + 2 * pad - HH) // stride
  out_width = 1 + (W + 2 * pad - WW) // stride
  out = np.zeros((N, F, out_height, out_width))

  # Pad the input
  x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
  
  for n in range(N):
        for f in range(F):
            for i in range(out_height):
                for j in range(out_width):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + HH
                    w_end = w_start + WW
                    x_slice = x_pad[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(x_slice * w[f, :, :, :]) + b[f]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
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

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

    # Pad x and dx for the gradient computation
  x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
  dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

  for n in range(N):  # For each image in the input batch
      for f in range(F):  # For each filter
          for i in range(out_height):
              for j in range(out_width):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + f_height
                    w_end = w_start + f_width

                    # Calculate gradients
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += w[f, :, :, :] * dout[n, f, i, j]
                    dw[f, :, :, :] += x_padded[n, :, h_start:h_end, w_start:w_end] * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]

    # Unpad dx to get the correct shape
  dx = dx_padded[:, :, pad:-pad, pad:-pad]
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

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
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    
    # Calculate output dimensions
  out_height = (H - pool_height) // stride + 1
  out_width = (W - pool_width) // stride + 1
  out = np.zeros((N, C, out_height, out_width))
  for n in range(N):
      for c in range(C):
          for i in range(out_height):
              for j in range(out_width):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + pool_height
                    w_end = w_start + pool_width
                    x_pooling_region = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, i, j] = np.max(x_pooling_region)

  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
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
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N, C, H, W = x.shape
  
  out_height = (H - pool_height) // stride + 1
  out_width = (W - pool_width) // stride + 1
    
  dx = np.zeros_like(x)
    
  for n in range(N):
      for c in range(C):
          for i in range(out_height):
              for j in range(out_width):
                    h_start = i * stride
                    w_start = j * stride
                    h_end = h_start + pool_height
                    w_end = w_start + pool_width
                    
                    # Find the max value in the input region
                    x_pooling_region = x[n, c, h_start:h_end, w_start:w_end]
                    max_value = np.max(x_pooling_region)

                    for i in range(pool_height):
                        for j in range(pool_width):
                            if x_pooling_region[i, j] == max_value:
                                dx[n, c, h_start + i, w_start + j] += dout[n, c, i, j]
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

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

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N, C, H, W = x.shape
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))

  if mode == 'train':
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        
        # Update the running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * mean.squeeze()
        running_var = momentum * running_var + (1 - momentum) * var.squeeze()
        
        # Normalize the batch
        x_normalized = (x - mean) / np.sqrt(var + eps)
        
        # Scale and shift
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        
        cache = (x, x_normalized, mean, var, gamma, beta, eps)
        
        # Update bn_param
        bn_param['running_mean'] = running_mean
        bn_param['running_var'] = running_var
  elif mode == 'test':
        # Use the running mean and variance to normalize during inference
        x_normalized = (x - running_mean.reshape(1, C, 1, 1)) / np.sqrt(running_var.reshape(1, C, 1, 1) + eps)
        out = gamma.reshape(1, C, 1, 1) * x_normalized + beta.reshape(1, C, 1, 1)
        cache = None
  else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

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

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  x, x_normalized, mean, var, gamma, beta, eps = cache
  N, C, H, W = dout.shape
    
    # Reshape gamma and dout for easier computation
  gamma_reshaped = gamma.reshape(1, C, 1, 1)
  dout_reshaped = dout * gamma_reshaped
    
    # Compute gradients with respect to the normalized input
  dx_normalized = dout_reshaped
  dvar = np.sum(dx_normalized * (x - mean) * -0.5 * np.power(var + eps, -1.5), axis=(0, 2, 3), keepdims=True)
  dmean = np.sum(dx_normalized * -1 / np.sqrt(var + eps), axis=(0, 2, 3), keepdims=True) + \
            dvar * np.mean(-2 * (x - mean), axis=(0, 2, 3), keepdims=True)
    
  dx = dx_normalized / np.sqrt(var + eps) + \
         dvar * 2 * (x - mean) / (N * H * W) + \
         dmean / (N * H * W)
    
  dgamma = np.sum(dout * x_normalized, axis=(0, 2, 3))
  dbeta = np.sum(dout, axis=(0, 2, 3))
    
    # Reshape gradients back to their original shapes
  dx = dx.reshape(N, C, H, W)
  dgamma = dgamma.reshape(C,)
  dbeta = dbeta.reshape(C,)
  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta