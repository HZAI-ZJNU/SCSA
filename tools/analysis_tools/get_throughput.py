import argparse
import time

import torch
import torch.nn as nn
from mmpretrain import get_model


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Get model throughput')
        self.parser.add_argument('config', help='config file path')
        self.parser.add_argument('--batch', type=int, default=64, help='data and model type')
        self.parser.add_argument('--warmup', type=int, default=50,
                                 help='Warming up rounds')
        self.parser.add_argument('--run', type=int, default=30, help='Running rounds')
        self.opts = self.parser.parse_args()


@torch.no_grad()
def throughput(
        model: nn.Module,
        batch_size: int = 64,
        warm_up_epochs: int = 50,
        run_epochs: int = 30,
):
    model.eval()
    model.cuda()

    images = torch.randn((batch_size, 3, 224, 224)).cuda()

    # warm up
    for i in range(warm_up_epochs):
        model(images)
    torch.cuda.synchronize()
    print('warm up over!')
    print(f"throughput averaged with 30 times")
    t2 = time.time()
    for i in range(run_epochs):
        model(images)
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"batch_size {batch_size} throughput {30 * batch_size / (t1 - t2)}")


if __name__ == '__main__':
    # python .\tools\analysis_tools\get_throughput.py .\work_dirs\resnet\fca\resnet101_8xb32_in1k_fca.py
    args = Args()
    model = get_model(args.opts.config)
    throughput(
        model=model,
        batch_size=args.opts.batch,
        warm_up_epochs=args.opts.warmup,
        run_epochs=args.opts.run,
    )
