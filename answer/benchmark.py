import argparse, torch, timeit, statistics
import numpy as np
from tqdm import tqdm
from cs336_basics import model, data, nn_utils, optimizer
from torch.optim import AdamW
import torch.cuda.nvtx as nvtx

model_size = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

def benchmark(args):
    print("Start benchmarking...")

    vocab_size = args.vocab_size
    context_length = args.context_length
    batch_size = args.batch_size
    transformer = model.BasicsTransformerLM(
        **model_size[args.model],
        vocab_size=vocab_size,
        context_length=context_length
    ).cuda()
    compile = args.compile

    if compile:
        transformer = torch.compile(transformer)

    if args.mode == "forward":
        with torch.no_grad():
            with nvtx.range("warmup"):
                for _ in range(args.warmup):
                    input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                    transformer.forward(input_ids)
                    torch.cuda.synchronize()

            times = []

            with nvtx.range("measure"):
                for _ in tqdm(range(args.steps)):
                    input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()

                    t0 = timeit.default_timer()
                    transformer.forward(input_ids)
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()
                    times.append(t1 - t0)

            print(f"average forward time = {statistics.mean(times)}")
            print(f"stdev forward time = {statistics.stdev(times)}")

    if args.mode == "backward":
        optim = optimizer.AdamW(transformer.parameters())

        with nvtx.range("warmup"):
            for _ in range(args.warmup):
                input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                labels = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                logits = transformer.forward(input_ids)
                optim.zero_grad()
                loss = nn_utils.cross_entropy(logits, labels)
                loss.backward()
                torch.cuda.synchronize()

        times = []

        with nvtx.range("measure"):
            for _ in tqdm(range(args.steps)):
                input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                labels = torch.randint(vocab_size, (batch_size, context_length)).cuda()

                t0 = timeit.default_timer()
                logits = transformer.forward(input_ids)
                torch.cuda.reset_peak_memory_stats()
                optim.zero_grad()
                loss = nn_utils.cross_entropy(logits, labels)
                loss.backward()
                print(f"after backward: {torch.cuda.max_memory_allocated()}")
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                times.append(t1 - t0)

        print(f"average forward & backward time = {statistics.mean(times)}")
        print(f"stdev forward & backward time = {statistics.stdev(times)}")

    if args.mode == "optimizer":
        optim = optimizer.AdamW(transformer.parameters())

        with nvtx.range("warmup"):
            for _ in range(args.warmup):
                input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                labels = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                logits = transformer.forward(input_ids)
                optim.zero_grad()
                loss = nn_utils.cross_entropy(logits, labels)
                loss.backward()
                optim.step()
                torch.cuda.synchronize()

        times = []
        
        with nvtx.range("measure"):
            for _ in tqdm(range(args.steps)):
                input_ids = torch.randint(vocab_size, (batch_size, context_length)).cuda()
                labels = torch.randint(vocab_size, (batch_size, context_length)).cuda()

                t0 = timeit.default_timer()
                logits = transformer.forward(input_ids)
                optim.zero_grad()
                loss = nn_utils.cross_entropy(logits, labels)
                loss.backward()
                optim.step()
                torch.cuda.synchronize()
                t1 = timeit.default_timer()
                times.append(t1 - t0)

        print(f"average forward & backward w/ optimizer time = {statistics.mean(times)}")
        print(f"stdev forward & backward w/ optimizer time = {statistics.stdev(times)}")

pytorch_attention_benchmark_config = {
    "d_model": [16, 32, 64, 128],
    "seq_len": [256, 1024, 4096, 8192, 16384]
}

def bench_pytorch_attention(batch_size, compiled=True):
    warmup = 15
    steps = 100

    sdpa = model.scaled_dot_product_attention

    if compiled:
        sdpa = torch.compile(model.scaled_dot_product_attention)

    for d_model in pytorch_attention_benchmark_config["d_model"]:
        for seq_len in pytorch_attention_benchmark_config["seq_len"]:
            try:
                forward_times = []
                backward_times = []
                gmems = []
                for step in tqdm(range(warmup + steps)):

                    Q = torch.rand(batch_size, seq_len, d_model, requires_grad=True).cuda()
                    K = torch.rand(batch_size, seq_len, d_model, requires_grad=True).cuda()
                    V = torch.rand(batch_size, seq_len, d_model, requires_grad=True).cuda()

                    t0 = timeit.default_timer()
                    out = sdpa(Q, K, V)
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()

                    if step >= warmup:
                        forward_times.append(t1 - t0)
                        gmems.append(torch.cuda.memory_allocated())

                    t0 = timeit.default_timer()
                    out.sum().backward()
                    torch.cuda.synchronize()
                    t1 = timeit.default_timer()

                    if step >= warmup:
                        backward_times.append(t1 - t0)

                print(f"d_model={d_model}, seq_len={seq_len}: forward mean={statistics.mean(forward_times)}, stdev={statistics.stdev(forward_times)}; \
backward mean={statistics.mean(backward_times)}, stdev={statistics.stdev(backward_times)}; \
gmem mean={statistics.mean(gmems)}, stdev={statistics.stdev(gmems)}")

            except (torch.AcceleratorError, torch.OutOfMemoryError):
                print(f"d_model={d_model}, seq_len={seq_len}: cudaOOMError")
                torch.cuda.empty_cache()
                continue

def bench_transformer(args):
    warmup = 15
    steps = 100
    vocab_size = 10000
    context_length = 512
    batch_size = args.batch_size

    transformer = model.BasicsTransformerLM(**model_size["small"],
                                            vocab_size=vocab_size,
                                            context_length=context_length).cuda()

    if args.compiled:
        try:
            transformer = torch.compile(transformer)
        except AttributeError:
            print(f"Compile transformer LM error.")
            return

    try:
        forward_times = []
        backward_times = []
        gmems = []
        for step in tqdm(range(warmup + steps)):

            input, _ = data.get_batch(np.arange(vocab_size), batch_size=batch_size, context_length=context_length, device="cuda")
            t0 = timeit.default_timer()
            out = transformer.forward(input)
            torch.cuda.synchronize()
            t1 = timeit.default_timer()

            if step >= warmup:
                forward_times.append(t1 - t0)
                gmems.append(torch.cuda.memory_allocated())

            t0 = timeit.default_timer()
            out.sum().backward()
            torch.cuda.synchronize()
            t1 = timeit.default_timer()

            if step >= warmup:
                backward_times.append(t1 - t0)

        print(f"forward mean={statistics.mean(forward_times)}, stdev={statistics.stdev(forward_times)}; \
backward mean={statistics.mean(backward_times)}, stdev={statistics.stdev(backward_times)}; \
gmem mean={statistics.mean(gmems)}, stdev={statistics.stdev(gmems)}")

    except (torch.AcceleratorError, torch.OutOfMemoryError):
        print(f"cudaOOMError")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--model", type=str, choices=["small", "medium"], default="small")
    parser.add_argument("--mode", type=str, choices=["forward", "backward", "optimizer"], default="forward")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--mixed-precision", action="store_true")  # 开关，不用传值
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # benchmark(args)
    benchmark(args)