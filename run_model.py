import sys
import torch
from model import Transformer, ModelArgs, transformer_configs


def benchmark_torch_function(iters, f):
    f()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(iters):
        f()
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def main():
    with torch.device('cuda'):
        model_args = ModelArgs(**transformer_configs['7B'])
        m = Transformer(model_args)
        m.to(torch.bfloat16)
        m.setup_caches(1, 2048)

        sample_args = (
            torch.randint(0, 32000, (1, 2048)),
            torch.arange(0, 2048),
        )
        def f():
            m(*sample_args)

        print(benchmark_torch_function(1, f))


from torch.utils._pytree import tree_map
from torch._functorch.make_functional import make_functional_with_buffers
from jax_integration import *

def wrap(jaxarray):
    return tree_map_only(jnp.ndarray, JaxTensor, jaxarray)

def unwrap(torchtensors):
    return tree_map_only(JaxTensor, lambda x: x._elem, torchtensors)


def main2(jitted_block=False, fori_loop=False):
    sample_args = (
        torch.randint(0, 32000, (1, 2048)),
        torch.arange(0, 2048),
    )
    sample_args = tree_map(move_to_device, sample_args)
    model_args = ModelArgs(**transformer_configs['7B'])
    # model_args.n_layer = 2
    m = Transformer(model_args, jitted_block, fori_loop)
    m.to(torch.bfloat16)
    m.setup_caches(1, 2048)
    m_func, weights, buffer = make_functional_with_buffers(m)

    causal_mask = move_to_device(m.causal_mask)
    freqs_cis = move_to_device(m.freqs_cis)

    if fori_loop:
        m_func.stateless_model._stacked_buffers = tree_map_only(torch.Tensor, move_to_device, m._stacked_buffers)
        m_func.stateless_model._stacked_weights = tree_map_only(torch.Tensor, move_to_device, m._stacked_weights)
    weights = tree_map(move_to_device, weights)
    buffer = tree_map(move_to_device, buffer)




    @jax.jit
    def m_func_jit(
        weights, buffer, args, causal_mask, freqs_cis
    ):
        weights, buffer, args, causal_mask, freqs_cis = wrap(
            (weights, buffer, args, causal_mask, freqs_cis)
        )
        m_func.stateless_model.freqs_cis = freqs_cis
        m_func.stateless_model.causal_mask = causal_mask
        res = m_func(weights, buffer, *args)
        res = unwrap(res)
        return res

    args = weights, buffer, sample_args, causal_mask, freqs_cis 
    args = unwrap(args)
    # print(m_func_jit.lower(*args).as_text())
    for _ in range(3):
        start = time.time()
        res = m_func_jit(*args)
        res = jax.block_until_ready(res)
        end = time.time()
        print(_, end - start)

if __name__ == '__main__':
    if sys.argv[1] == 'torch':
        main()
    elif sys.argv[1] == 'jax':
        main2()
    elif sys.argv[1] == 'jax_block':
        main2(True, False)
    elif sys.argv[1] == 'jax_fori':
        main2(False, True)


    