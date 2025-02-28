import os
import os.path as osp
import time
import argparse
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from utils import get_logger, all_reduce_tensor, extant_file

class Engine(object):
    def __init__(self, custom_parser=None):
        self.devices = None
        self.distributed = False

        if custom_parser is None:
            self.parser = argparse.ArgumentParser()
        else:
            assert isinstance(custom_parser, argparse.ArgumentParser)
            self.parser = custom_parser

        self.inject_default_parser()
        self.args = self.parser.parse_args()
        self.continue_state_object = self.args.continue_fpath
        
        # Remove CUDA-specific code
        self.devices = [0]  # Just use CPU
        self.local_rank = 0
        self.world_size = 1

    def data_parallel(self, model):
        # Simple wrapper for consistency
        return model

    def get_train_loader(self, train_dataset, collate_fn=None):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=True,
            pin_memory=False,  # Changed from True since we're on CPU
            collate_fn=collate_fn
        )
        return train_loader, None

    def get_test_loader(self, test_dataset):
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=False  # Changed from True since we're on CPU
        )
        return test_loader, None

    def all_reduce_tensor(self, tensor, norm=True):
        return torch.mean(tensor)

    def inject_default_parser(self):
        p = self.parser
        p.add_argument('-d', '--devices', default='',
                       help='set data parallel training')
        p.add_argument('-c', '--continue', type=extant_file,
                       metavar="FILE",
                       dest="continue_fpath",
                       help='continue from one certain checkpoint')

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        if type is not None:
            print("A exception occurred during Engine initialization, "
                  "give up running process")
            return False

