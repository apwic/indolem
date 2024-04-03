import argparse

def adapter_names_type(s):
    if s.lower() == 'none':
        return None
    return [item.strip() for item in s.split(',')]

class BaseArgParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--data_path', default='data/', help='path to all train/test/dev folds')
        self.add_argument('--bert_lang', default='id', choices=['id', 'multi', 'my'], help='select one of language')
        self.add_argument('--batch_size', type=int, default=30, help='batch size')
        self.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
        self.add_argument('--weight_decay', type=int, default=0, help='weight decay')
        self.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
        self.add_argument('--max_grad_norm', type=float, default=1.0)
        self.add_argument('--num_train_epochs', type=int, default=20, help='total epoch')
        self.add_argument('--warmup_steps', type=int, default=242, help='warmup_steps, the default value is 10% of total steps')
        self.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
        self.add_argument('--seed', type=int, default=2020)
        self.add_argument('--local_rank', type=int, default=-1)
        self.add_argument('--patience', type=int, default=5, help='patience for early stopping')
        self.add_argument('--no_cuda', default=False)
        self.add_argument('--world_size', type=int, default=1)
        self.add_argument('--adapters', type=adapter_names_type, default=None, 
                        help="Comma-separated list of adapter names to use, or 'None' for no adapters (default: None)")

class SentimentArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--max_token', type=int, default=200, help='maximum token allowed for 1 instance')

class NTPArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()
        self.add_argument('--max_token_premise', type=int, default=240, help='maximum token allowed for premise')
        self.add_argument('--max_token_nextTw', type=int, default=40, help='maximum token allowed for next tweet')