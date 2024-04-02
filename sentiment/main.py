import logging
import argparse
import torch
import pandas as pd
from utils.utils import set_seed
from utils.preprocessing import BertData
from utils.models import Model

logger = logging.getLogger(__name__)
lang2model = { 'id': 'indolem/indobert-base-uncased',
               'multi': 'bert-base-multilingual-uncased',
               'my': 'huseinzol05/bert-base-bahasa-cased' }
lang2pad = {'id': 0, 'multi': 0, 'my': 5}

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
args_parser.add_argument('--bert_lang', default='id', choices=['id', 'multi', 'my'], help='select one of language')
args_parser.add_argument('--max_token', type=int, default=200, help='maximum token allowed for 1 instance')
args_parser.add_argument('--batch_size', type=int, default=30, help='batch size')
args_parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
args_parser.add_argument('--max_grad_norm', type=float, default=1.0)
args_parser.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
args_parser.add_argument('--warmup_steps', type=int, default=242, help='warmup_steps, the default value is 10% of total steps')
args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
args_parser.add_argument('--seed', type=int, default=2020)
args_parser.add_argument('--local_rank', type=int, default=-1)
args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
args_parser.add_argument('--no_cuda', default=False)
args = args_parser.parse_args()

# Setup CUDA, GPU & distributed training
if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
else: # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
args.device = device

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

bertdata = BertData(args, lang2model)

dev_f1s = 0.0
test_f1s = 0.0
for idx in range(5):
    trainset = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
    devset = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
    testset = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
    xtrain, ytrain = list(trainset['sentence']), list(trainset['sentiment'])
    xdev, ydev = list(devset['sentence']), list(devset['sentiment'])
    xtest, ytest = list(testset['sentence']), list(testset['sentiment'])
    model = Model(args, device, logger, lang2model, lang2pad)
    model.to(args.device)
    train_dataset = bertdata.preprocess(xtrain, ytrain)
    dev_dataset = bertdata.preprocess(xdev, ydev)
    test_dataset = bertdata.preprocess(xtest, ytest)
    
    global_step, tr_loss, best_f1_dev, best_f1_test = model.process(train_dataset, dev_dataset, test_dataset)
    dev_f1s += best_f1_dev
    test_f1s += best_f1_test

print('End of Training 5-fold')
print('Dev set F1', dev_f1s/5.0)
print('Test set F1', test_f1s/5.0)