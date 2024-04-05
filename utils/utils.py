import numpy as np
import torch
import random
import json
import os
import platform

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def read_data_ntp(fname):
    chat = [] #will be 4 times
    response = []
    label = []
    data=json.load(open(fname,'r'))
    for datum in data:
        for key, option in datum['next_tweet']:
            chat.append(' '.join(datum['tweets']))
            response.append(option)
            label.append(key)
    return chat, response, label

def read_data_tweet_ordering(fname):
    tweets = []
    labels = []
    data=json.load(open(fname,'r'))
    for datum in data:
        tweets.append(datum['tweets'])
        labels.append(datum['order'])
    return tweets, labels

def setup_environment(args):
    # Check if CUDA is available and not disabled by --no_cuda
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    print(f"{'CUDA' if use_cuda else 'CPU'} available")

    # Set device based on whether CUDA is used
    device = torch.device("cuda" if use_cuda else "cpu")

    # Setup for distributed training
    if args.local_rank != -1:
        print("Running in distributed mode")

        # Initialize the correct device based on local_rank
        if use_cuda:
            torch.cuda.set_device(args.local_rank)

        # The backend is set to NCCL for CUDA, Gloo for CPUs
        backend = "nccl" if use_cuda else "gloo"

        # Set environment variables for the master address and port
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'  # Use an open port

        # Initialize the distributed environment
        torch.distributed.init_process_group(backend=backend, rank=args.local_rank, world_size=torch.cuda.device_count() if use_cuda else args.world_size)
        
    else:
        print("Running in non-distributed mode")

    # Update args with the selected device and number of GPUs
    args.device = device
    args.n_gpu = torch.cuda.device_count() if use_cuda else 1

    return args