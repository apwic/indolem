import logging
import argparse
import torch
import pandas as pd
from utils.utils import set_seed, setup_environment, read_data_tweet_ordering
from utils.preprocessing import TweetOrderingData
from utils.models import TweetOrderingModel
from utils.argparser import TweetOrderingArgParser

logger = logging.getLogger(__name__)
lang2model = { 'id': 'indolem/indobert-base-uncased',
               'multi': 'bert-base-multilingual-uncased',
               'my': 'huseinzol05/bert-base-bahasa-cased' }
lang2pad = {'id': 0, 'multi': 0, 'my': 5}

if __name__ == "__main__":
    args_parser = TweetOrderingArgParser()
    args = args_parser.parse_args()
    args = setup_environment(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    set_seed(args)

    bertdata = TweetOrderingData(args, lang2model)

dev_rankCorrs = 0.0
test_rankCorrs = 0.0
for idx in range(5):
    trainset = read_data_tweet_ordering(args.data_path+'train'+str(idx)+'.json')
    devset = read_data_tweet_ordering(args.data_path+'dev'+str(idx)+'.json')
    testset = read_data_tweet_ordering(args.data_path+'test'+str(idx)+'.json')
    model = TweetOrderingModel(args, logger, lang2model, lang2pad)
    model.to(args.device)
    train_dataset = bertdata.preprocess(trainset[0], trainset[1])
    dev_dataset = bertdata.preprocess(devset[0], devset[1])
    test_dataset = bertdata.preprocess(testset[0], testset[1])
    
    global_step, tr_loss, best_metrics = model.train_model(train_dataset, dev_dataset, test_dataset)
    dev_rankCorrs += best_metrics['rank_corr']
    test_rankCorrs += best_metrics['test_rank_corr']

print('End of Training 5-fold')
print('Dev set RankCorr', dev_rankCorrs/5.0)
print('Test set RankCorr', test_rankCorrs/5.0)