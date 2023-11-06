import numpy as np
import torch

def group(xs, aggr):
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out

def gather_nd(params, indices, batch_dims=0):
    """ The same as tf.gather_nd.
    Reference:
    https://colab.research.google.com/drive/19iN4ybH1uvTbOMfSVp5hN5NHucAvM94f?usp=sharing#scrollTo=-08v0MPusUBV

    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:] 

    """
    if isinstance(indices, torch.Tensor):
      indices = indices.cpu().numpy()
    else:
      if not isinstance(indices, np.array):
        raise ValueError(f'indices must be `torch.Tensor` or `numpy.array`. Got {type(indices)}')
    if batch_dims == 0:
        orig_shape = list(indices.shape)
        num_samples = int(np.prod(orig_shape[:-1]))
        m = orig_shape[-1]
        n = len(params.shape)

        if m <= n:
            out_shape = orig_shape[:-1] + list(params.shape[m:])
        else:
            raise ValueError(
                f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
            )
        indices = indices.reshape((num_samples, m)).transpose().tolist()
        output = params[indices]    # (num_samples, ...)
        return output.reshape(out_shape).contiguous()
    else:
        batch_shape = params.shape[:batch_dims]
        orig_indices_shape = list(indices.shape)
        orig_params_shape = list(params.shape)
        assert (
            batch_shape == indices.shape[:batch_dims]
        ), f'if batch_dims is not 0, then both "params" and "indices" have batch_dims leading batch dimensions that exactly match.'
        mbs = np.prod(batch_shape)
        if batch_dims != 1:
            params = params.reshape(mbs, *(params.shape[batch_dims:]))
            indices = indices.reshape(mbs, *(indices.shape[batch_dims:]))
        output = []
        for i in range(mbs):
            output.append(gather_nd(params[i], indices[i], batch_dims=0))
        output = torch.stack(output, dim=0)
        output_shape = orig_indices_shape[:-1] + list(orig_params_shape[orig_indices_shape[-1]+batch_dims:])
        return output.reshape(*output_shape).contiguous()