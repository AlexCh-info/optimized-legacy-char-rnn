import asyncio
import time
import torch
import pandas as pd
from helpers import char_tensor, all_characters
from new_generate import Generation
from generate import generate as legacy_generate



def load_model(path: str, cuda: bool = False):
    decoder = torch.load(path, weights_only=False)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    if cuda:
        decoder = decoder.cuda()
    return decoder

def benchmark_legacy_repeated_calls(decoder, base_text: str, growth_step: int, n_calls: int, predict_len: int = 1, cuda: bool = False):
    """
    Симулирует ситуацию 'стриминга' поверх старой функции: каждый следующий
    вызов должен учитывать всё, что было сгенерировано раньше, поэтому
    prime_str каждый раз становится длиннее.
    """
    results = []
    text = base_text
    for i in range(n_calls):
        prime_len = len(text)
        start = time.perf_counter()
        _ = legacy_generate(decoder, prime_str=text, predict_len=predict_len, cuda=cuda)
        elapsed = time.perf_counter() - start
        results.append({"call_idx": i, "prime_len": prime_len, "elapsed_sec": elapsed})
        text += "a" * growth_step
    return pd.DataFrame(results)

async def benchmark_new_class(decoder, base_text: str, n_steps: int, cuda: bool = False):
    gen = Generation(decoder, cuda=cuda)

    start_warmup = time.perf_counter()
    await gen.warm_up(base_text)
    warmup_time = time.perf_counter() - start_warmup

    results = []
    for i in range(n_steps):
        start = time.perf_counter()
        _ = await gen.step()
        elapsed = time.perf_counter() - start
        results.append({"step_idx": i, "elapsed_sec": elapsed})

    return warmup_time, pd.DataFrame(results)

async def run_one_client(decoder, prime_str: str, n_steps: int, cuda: bool = False):
    gen = Generation(decoder, cuda=cuda)
    await gen.warm_up(prime_str)
    chars = []
    for _ in range(n_steps):
        chars.append(await gen.step())
    return "".join(chars)


async def benchmark_parallel_clients(decoder, n_clients: int, prime_str: str, n_steps: int, cuda: bool = False):
    start = time.perf_counter()
    tasks = [run_one_client(decoder, prime_str, n_steps, cuda) for _ in range(n_clients)]
    outputs = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start

    unique_outputs = len(set(outputs))
    print(f"[parallel] clients={n_clients} total_time={elapsed:.3f}s "
          f"unique_outputs={unique_outputs}/{n_clients}")
    for i, out in enumerate(outputs):
        print(f"  client {i}: {out!r}")
    return elapsed, outputs

async def main():
    MODEL_PATH = "weights.pt"
    CUDA = False

    decoder = load_model(MODEL_PATH, cuda=CUDA)

    print("1. Legacy generate(): растущий prime_str")
    df_legacy = benchmark_legacy_repeated_calls(
        decoder, base_text="The ", growth_step=50, n_calls=10, predict_len=1, cuda=CUDA
    )
    print(df_legacy)
    print(f"Итого time legacy: {df_legacy['elapsed_sec'].sum():.4f} sec\n")

    print("2. Новый класс Generation: warm_up + N step()")
    warmup_time, df_new = await benchmark_new_class(
        decoder, base_text="The ", n_steps=10, cuda=CUDA
    )
    print(f"warm_up time: {warmup_time:.4f} sec")
    print(df_new)
    print(f"Итого time step()'ов: {df_new['elapsed_sec'].sum():.4f} sec\n")

    print("3. Параллельные клиенты")
    await benchmark_parallel_clients(
        decoder, n_clients=3, prime_str="Hello world", n_steps=20, cuda=CUDA
    )


if __name__ == "__main__":
    asyncio.run(main())
