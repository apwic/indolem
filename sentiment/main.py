import logging
import argparse
import torch
import pandas as pd
from utils.utils import set_seed, setup_environment
from utils.preprocessing import SentimentData
from utils.models import SentimentModel
from utils.argparser import SentimentArgParser

logger = logging.getLogger(__name__)
lang2model = { 'id': 'indolem/indobert-base-uncased',
               'multi': 'bert-base-multilingual-uncased',
               'my': 'huseinzol05/bert-base-bahasa-cased' }
lang2pad = {'id': 0, 'multi': 0, 'my': 5}

if __name__ == "__main__":
    args_parser = SentimentArgParser()
    args = args_parser.parse_args()
    args = setup_environment(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(args)

    bertdata = SentimentData(args, lang2model)

    dev_f1s = 0.0
    test_f1s = 0.0
    for idx in range(5):
        trainset = pd.read_csv(args.data_path+'train'+str(idx)+'.csv')
        devset = pd.read_csv(args.data_path+'dev'+str(idx)+'.csv')
        testset = pd.read_csv(args.data_path+'test'+str(idx)+'.csv')
        xtrain, ytrain = list(trainset['sentence']), list(trainset['sentiment'])
        xdev, ydev = list(devset['sentence']), list(devset['sentiment'])
        xtest, ytest = list(testset['sentence']), list(testset['sentiment'])
        model = SentimentModel(args, logger, lang2model, lang2pad)
        model.to(args.device)
        train_dataset = bertdata.preprocess(xtrain, ytrain)
        dev_dataset = bertdata.preprocess(xdev, ydev)
        test_dataset = bertdata.preprocess(xtest, ytest)
        
        global_step, tr_loss, best_metrics = model.train_model(train_dataset, dev_dataset, test_dataset)
        dev_f1s += best_metrics['f1']
        test_f1s += best_metrics['test_f1']

    print('End of Training 5-fold')
    print('Dev set F1', dev_f1s/5.0)
    print('Test set F1', test_f1s/5.0)