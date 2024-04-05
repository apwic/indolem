import logging
import argparse
import torch
import pandas as pd
from utils.utils import set_seed, read_data_ntp, setup_environment
from utils.preprocessing import NTPData
from utils.models import NTPModel
from utils.argparser import NTPArgParser

logger = logging.getLogger(__name__)
lang2model = { 'id': 'indolem/indobert-base-uncased',
               'multi': 'bert-base-multilingual-uncased',
               'my': 'huseinzol05/bert-base-bahasa-cased' }
lang2pad = {'id': 0, 'multi': 0, 'my': 5}

if __name__ == "__main__":
    args_parser = NTPArgParser()
    args = args_parser.parse_args()
    args = setup_environment(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    if args.local_rank == 0:
        torch.distributed.barrier()

    bertdata = NTPData(args, lang2model)

    trainset = read_data_ntp(args.data_path+'train.json')
    devset = read_data_ntp(args.data_path+'dev.json')
    testset = read_data_ntp(args.data_path+'test.json')
    model = NTPModel(args, logger, lang2model, lang2pad)
    model.to(args.device)
    train_dataset = bertdata.preprocess(trainset[0], trainset[1], trainset[2])
    dev_dataset = bertdata.preprocess(devset[0], devset[1], devset[2])
    test_dataset = bertdata.preprocess(testset[0], testset[1], testset[2])

    global_step, tr_loss, best_acc_dev, best_acc_test = model.train_model(train_dataset, dev_dataset, test_dataset)
    print('Dev set accuracy', best_acc_dev)
    print('Test set accuracy', best_acc_test)