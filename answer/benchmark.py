import argparse, torch, timeit, statistics
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
                optim.zero_grad()
                loss = nn_utils.cross_entropy(logits, labels)
                loss.backward()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--model", type=str, choices=["small", "medium"], default="small")
    parser.add_argument("--mode", type=str, choices=["forward", "backward", "optimizer"], default="forward")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--mixed-precision", action="store_true")  # 开关，不用传值
    args = parser.parse_args()

    benchmark(args)