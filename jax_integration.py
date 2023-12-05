import numpy as np
import torch
from torch._decomp import get_decompositions, core_aten_decompositions
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_map_only
from jax import numpy as jnp
import jax
from torch._functorch.make_functional import make_functional_with_buffers
import time

_lowerings = {}
def register(name):
    def inner(func):
        global _lowerings
        _lowerings[name] = func
        return func
    return inner


def _get_numpy_dtype(dtype):
  return {
      torch.double: jnp.double,
      torch.float32: jnp.float32,
      #torch.half: jnp.half,
      torch.long: jnp.int64,
      torch.int32: jnp.int32,
      torch.int16: jnp.int16,
      torch.bool: jnp.bool_,
  }.get(dtype)

@register("aten::view")
@register("aten::_unsafe_view")
def _aten_unsafe_view(x, shape):
    return jnp.reshape(x, shape)

@register("aten::add")
def _aten_add(x, y):
    '''
    if isinstance(x, jnp.ndarray) and isinstance(y, jnp.ndarray):
        assert x.dtype == y.dtype, (x.dtype, y.dtype)
    '''
    return x + y

@register("aten::copy_")
def _aten_copy(x, y, memory_format=None):
    return jnp.copy(y)

@register("aten::clone")
def _aten_clone(x, memory_format=None):
    return jnp.copy(x)

@register("aten::full")
def _aten_full(size, value, **kwargs):
    return jnp.full(size, value)

@register("aten::index_copy")
def _aten_index_copy(x, dim, indexes, source):
    # return jax.lax.scatter(x, index, dim)
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(indexes)
        else:
            dims.append(slice(None, None, None))
    return x.at[dim].set(source)

@register("aten::select")
def _aten_select(x, dim, index):
    """
    slice_sizes = list(x.shape)
    slice_sizes[dim] = 1
    indexes = jnp.append(indexes, 1)
    offset_dims = [i for i in range(len(x.shape)) if i != dim]
    gather_dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(dim, ),
        collapsed_slice_dims=(dim, ),
        start_index_map=(dim, ),
    )
    return jax.lax.gather(x, indexes, gather_dnums, tuple(slice_sizes))
    """
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(index)
        else:
            dims.append(slice(None, None, None))
    return x[tuple(dims)]

@register("aten::index_select")
def _aten_index_select(x, dim, indexes):
    """
    slice_sizes = list(x.shape)
    slice_sizes[dim] = 1
    indexes = jnp.append(indexes, 1)
    offset_dims = [i for i in range(len(x.shape)) if i != dim]
    gather_dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(dim, ),
        collapsed_slice_dims=(dim, ),
        start_index_map=(dim, ),
    )
    return jax.lax.gather(x, indexes, gather_dnums, tuple(slice_sizes))
    """
    dims = []
    for i in range(len(x.shape)):
        if i == dim:
            dims.append(indexes)
        else:
            dims.append(slice(None, None, None))
    return x[tuple(dims)]

@register("aten::mean")
def _aten_mean(x, dim, keepdim):
    return jnp.mean(x, dim, keepdims=keepdim)

@register("aten::sub")
def _aten_sub(x, y):
    return x - y

@register("aten::mm")
def _aten_mm(x, y):
  res = x @ y
  assert res.dtype == jnp.bfloat16
  return res

@register("aten::mul")
def _aten_mul(x, y):
    return x * y

@register("aten::silu")
def _aten_silu(x):
    return jax.nn.silu(x)

@register("aten::t")
def _aten_t(x):
    return jnp.transpose(x)
    
@register("aten::transpose")
def _aten_transpose(x, dim0, dim1):
    shape = list(range(len(x.shape)))
    shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
    return jnp.transpose(x, shape)

@register("aten::triu")
def _aten_triu(m, k):
    return jnp.triu(m, k)

@register("aten::slice")
def _aten_slice(self, dim=0, start=None, end=None, step=1):
    sl = slice(start, end, step)
    dims = []
    for i in range(len(self.shape)):
        if i == dim:
            dims.append(sl)
        else:
            dims.append(slice(None, None, None))
    return self[tuple(dims)]

@register("aten::detach")
def _aten_detach(self):
    return self



@register("aten::view_as_real")
def _aten_view_as_real(x):
    real = jnp.real(x)
    im = jnp.imag(x)
    res = jnp.stack([real, im], -1)
    return res

@register("aten::stack")
def _aten_stack(tensors, dim=0):
    return jnp.stack(tensors, dim)

@register('aten::_softmax')
def _aten_softmax(x, dim, halftofloat):
    return jax.nn.softmax(x, dim)


@register('aten::pow')
def _aten_pow(x, y):
  if isinstance(y, int):
    y = float(y)
  if isinstance(y, jnp.ndarray):
    y = y.astype(jnp.astype(jnp.bfloat16))
  return jnp.power(x, y)

@register('aten::view_as_complex')
def _aten_view_as_complex(input):
  if input.dtype == jnp.bfloat16:
      input = input.astype(jnp.float32)
  x, y = input[..., 0], input[..., 1]
  return jax.lax.complex(x, y)

@register('aten::div')
def _aten_div(x, y):
  return x / y

@register('aten::bmm')
def _aten_bmm(x, y):
  res = x @ y
  assert res.dtype == jnp.bfloat16
  return res
  # return jnp.einsum('bnm,bmk->bnk', x, y)

@register('aten::embedding')
def _aten_embedding(a, w):
  return jnp.take(a, w, axis=0)

@register('aten::rsqrt')
def _aten_rsqrt(x):
  return jax.lax.rsqrt(x)

@register('aten::expand')
def _aten_expand(x, dims):
    return jnp.broadcast_to(x, dims)

@register('aten::dot')
def _aten_dot(x, y):
    return jnp.dot(x, y)

@register('aten::_to_copy')
def _aten__to_copy(tensor, **kwargs):
    dtype = _get_numpy_dtype(kwargs['dtype'])
    if dtype != tensor.dtype:
        return tensor.astype(dtype)
    return jnp.copy(tensor)


@register('aten::empty')
def _aten_empty(sizes, **kwargs):
    return jnp.zeros(sizes)

@register('aten::index_put_')
def _aten_index_put(self, indexes, values):
    indexes = [
        slice(None, None, None) if i is None
        else i
        for i in indexes
    ]
    indexes = tuple(indexes)
    return self.at[indexes].set(values)

@register('aten::index')
def _aten_index(self, indexes):
    indexes = [
        slice(None, None, None) if i is None
        else i
        for i in indexes
    ]
    indexes = tuple(indexes)
    return self[indexes]

import jax.numpy as jnp
import jax.lax as lax

@register('aten::split_with_sizes')
def split_with_sizes(x, sizes, dim):
  """Splits an array `x` into sub-arrays based on static sizes `sizes`.

  Args:
    x: The input array to split.
    sizes: A 1D array of integer sizes for each sub-array.

  Returns:
    A list of sub-arrays.
  """
  rank = x.ndim
  splits = np.cumsum(sizes)  # Cumulative sum for split points
  def make_range(rank, dim, start, end):
    res = [slice(None, None, None)] * rank
    res[dim] = slice(start, end)
    return tuple(res)
  return [x[make_range(rank, dim, start, end)] 
          for start, end in zip([0] + list(splits[:-1]), splits)]

@register('aten::permute')
def permute(t, dims):
    return jnp.transpose(t, dims)

@register('aten::unsqueeze')
def _aten_unsqueeze(self, dim):
    if dim < 0:
        dim += input.ndim
    return jnp.expand_dims(self, dim)



def get_function(name):
    name = name.split('.')[0]
    return _lowerings[name]


from torch.utils._python_dispatch import TorchDispatchMode

def make_jax_array(tensor):
    if isinstance(tensor, JaxTensor):
        return tensor._elem
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach()
        if hasattr(tensor, '_elem'):
            return tensor._elem
        return t2j(tensor)
    return tensor
    # TODO use dlpack


import torch
import torch.utils.dlpack as torchdl
from jax import dlpack as jaxdl

device = jax.devices()[0]

def t2j(t):
    if isinstance(t, JaxTensor):
        return t._elem
    if t.dtype == torch.bool:
        t = t.to(torch.int32)
    dl = torchdl.to_dlpack(t)
    res = jaxdl.from_dlpack(dl)
    return jax.device_put(res, device=device)

def move_to_device(t):
    return JaxTensor(t2j(t))


CORE_ATEN = core_aten_decompositions()

# All of the tensor examples in this zoo inherit from BaseTensor.  Ideally,
# however, they would inherit directly from Tensor.  This is just our staging
# ground for applying behavior that hasn't yet made it into core but that
# we would like to apply by default.
class JaxTensor(torch.Tensor):
    # See https://github.com/pytorch/pytorch/pull/73727 ; this is necessary
    # to ensure that super().__new__ can cooperate with each other
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(elem.shape, dtype=torch.float32, device="meta"),
            require_grad=False,
        )

    def __init__(self, elem):
        super().__init__()
        self._elem = elem

    def __str__(self):
        return 'Tensor backed by Jax ' + str(self._elem)


    @property
    def shape(self):
        return self._elem.shape

    @property
    def ndim(self):
        return len(self._elem.shape)

    def flatten(self, start_dim=0, end_dim=-1):
        if end_dim == -1:
            end_dim = self.ndim
        new_shape = self.shape[:start_dim] + (-1,) + self.shape[end_dim:]
        return torch.reshape(self, new_shape)

    def __setitem__(self, key, val):
        if isinstance(key, tuple):
            key = tuple(make_jax_array(k) if isinstance(k, torch.Tensor) else k for k in key)
        else:
            key = key._elem
        self._elem = self._elem.at[key].set(val._elem)

    def type_as(self, other):
        self._elem = self._elem.astype(other._elem.dtype)
        return self

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # print('running...', func.name())
        if func.name() not in _lowerings:
            if func in CORE_ATEN:
                return CORE_ATEN[func](*args, **kwargs)
        args = tree_map(make_jax_array, args)
        kwargs = tree_map(make_jax_array, kwargs)
        res = get_function(func.name())(*args, **kwargs)
        # run_torch_and_diff(func, args, kwargs, res)
        if func.name() == 'aten::copy_':
            args[0]._elem = res
            return args[0]
        if isinstance(res, (tuple, list)):
            return tree_map_only(jnp.ndarray, JaxTensor, res)
        return JaxTensor(res)


class PrintingMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    print(func.name())
    return func(*args, **kwargs)

class VerifyMode(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        jax_args = tree_map(make_jax_array, args)
        jax_kwargs = tree_map(make_jax_array, kwargs)
        jax_res = get_function(func.name())(*jax_args, **jax_kwargs)
        torch_res = func(*args, **kwargs)
        print(func.name())
        jax_res_t = torch.from_numpy(np.array(jax_res))
        if not torch.allclose(torch_res, torch.from_numpy(np.array(jax_res)), atol=1e-4):
            print('FAIL', func.name(), (torch_res - jax_res_t).norm())
        else:
            print('... passes')
        return torch_res


def run_torch_and_diff(func, args, kwargs, res):
    def to_torch(x):
        if isinstance(x, jnp.ndarray):
            return torch.from_numpy(np.array(x))
        return x
    args = tree_map(to_torch, args)
    kwargs = tree_map(to_torch, kwargs)
    if func.name() == 'aten::view':
        torch_res = args[0].reshape(args[1])
    else:
        torch_res = func(*args, **kwargs)
    res = to_torch(res)
    print(func.name(), end=' ')
    if torch.allclose(torch_res, res, atol=1e-3):
        print('pass')
    else:
        print('fail')
    


class JaxMode(TorchDispatchMode):
  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    # print('running...', func.name())
    args = tree_map(make_jax_array, args)
    kwargs = tree_map(make_jax_array, kwargs)
    res = get_function(func.name())(*args, **kwargs)
    # run_torch_and_diff(func, args, kwargs, res)
    if func.name() == 'aten::copy_':
        args[0]._elem = res
        return args[0]
    return JaxTensor(res)


def make_jax_loop(model_arg, model_prefill, model_decode):

    # weight include caches?
    def run_loop(weights, orig_inputs):
        # make cache here?

        prefill_input = (
            orig_inputs,
            jnp.arange(0, context_length),
            jnp.arange(0, context_length),
            mask,
            freqs)

        logits, caches = model_prefill(weights, prefill_input)

        next_token = jnp.argmax(logits[0][-1]).reshape((1,))

        def body(state):
            decode_input = (
                state.input_tokens.reshape( (1, 1)),
                state.index,
                state.cache_index,
                None,
                freqs,
            )
            logits, caches = model_decode(weights, decode_input)
            next_token = jnp.argmax(logits[0][-1]).reshape((1,))
            res = state.res.at[state.index - context_length].set(next_token[0])
            return _State(res, caches, next_token, state.index + 1, state.cache_index + 1)

        def condition(states):
            return states.index[-1] < infer_length  + context_length

        results = jnp.zeros((infer_length, )).astype(jnp.int64)
        results = results.at[0].set(next_token[0])
        start = _State(
            results,
            caches,
            next_token,
            jnp.arange(1 + context_length, context_length + 2),
            jnp.arange(2, 2 + context_length),
        )

        return jax.lax.while_loop(condition, body, start).res
    return run_loop


class JaxJitted:

    def __init__(self, m, set_weights=True):
        self._m = m
        m_func, _weight, _buffer = make_functional_with_buffers(m)
        if set_weights:
            self._weight = [x._elem for x in _weight]
            self._buffer = [x._elem for x in _buffer]


        def m_func_jax(weights, buffers, args):
            weights, buffers, args = tree_map_only(
                jnp.ndarray, 
                lambda x: JaxTensor(x),
                (weights, buffers, args)
            )
            res = m_func(weights, buffers, *args)
            return res._elem
        
        self._jitted = jax.jit(m_func_jax)

    def __call__(self, *args):
        args = [x._elem for x in args]
        res = self._jitted(self._weight, self._buffer, args)
        return JaxTensor(res)

    def set_weights(self, weights, buffer):
        self._weight = [x._elem for x in weights]
        self._buffer = [x._elem for x in buffer]
