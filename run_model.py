import sys
import torch
from model import Transformer, ModelArgs, transformer_configs


def benchmark_torch_function(iters, f):
    for i in range(iters):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        f()
        end_event.record()
        torch.cuda.synchronize()
        print(i, start_event.elapsed_time(end_event) * 1.0e-3)


def main(inductor=False):
    with torch.device('cuda'):
        model_args = ModelArgs(**transformer_configs['7B'])
        m = Transformer(model_args)
        m.to(torch.bfloat16)
        m.setup_caches(1, 2048)

        sample_args = (
            torch.randint(0, 32000, (1, 2048)),
            torch.arange(0, 2048),
        )
        if inductor:
            m = torch.compile(m, mode="reduce-overhead", fullgraph=True)
        def f():
            m(*sample_args)

        benchmark_torch_function(2, f)


from torch.utils._pytree import tree_map
from torch._functorch.make_functional import make_functional_with_buffers
from jax_integration import *

def wrap(jaxarray):
    return tree_map_only(jnp.ndarray, JaxTensor, jaxarray)

def unwrap(torchtensors):
    return tree_map_only(JaxTensor, lambda x: x._elem, torchtensors)


def main2(jit=True):
    sample_args = (
        torch.randint(0, 32000, (1, 2048)),
        torch.arange(0, 2048),
    )
    sample_args = tree_map(move_to_device, sample_args)
    model_args = ModelArgs(**transformer_configs['7B'])
    # model_args.n_layer = 2
    m = Transformer(model_args)
    m.to(torch.bfloat16)
    m.setup_caches(1, 2048)
    m_func, weights, buffer = make_functional_with_buffers(m)

    causal_mask = move_to_device(m.causal_mask)
    freqs_cis = move_to_device(m.freqs_cis)

    weights = tree_map(move_to_device, weights)
    buffer = tree_map(move_to_device, buffer)




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

    if jit:
        m_func_jit = jax.jit(m_func_jit)
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
        main(False)
    if sys.argv[1] == 'torch_inductor':
        main(True)
    elif sys.argv[1] == 'jax':
        main2(False)
    elif sys.argv[1] == 'jax_jit':
        main2(True)


    