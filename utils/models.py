import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertModel, get_linear_schedule_with_warmup
from utils.utils import set_seed

class Batch():
    def __init__(self, data, idx, args, lang2pad):
        cur_batch = data[idx:idx+args.batch_size]
        src = torch.tensor([x[0] for x in cur_batch])
        seg = torch.tensor([x[1] for x in cur_batch])
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src!=lang2pad[args.bert_lang])
        
        self.src = src.to(args.device)
        self.seg= seg.to(args.device)
        self.label = label.to(args.device)
        self.mask_src = mask_src.to(args.device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src
    
class Model(nn.Module):
    def __init__(self, args, device, logger, lang2model, lang2pad):
        super(Model, self).__init__()
        self.args = args
        self.device = device
        self.bert = BertModel.from_pretrained(lang2model[args.bert_lang])
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(reduction='none') 
        self.logger = logger
        self.lang2model = lang2model
        self.lang2pad = lang2pad

    def forward(self, src, seg, mask_src):
        output = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = output.last_hidden_state
        top_vec = self.dropout(top_vec)
        top_vec *= mask_src.unsqueeze(dim=-1).float()
        top_vec = torch.sum(top_vec, dim=1) / mask_src.sum(dim=-1).float().unsqueeze(-1)
        conclusion = self.linear(top_vec).squeeze()
        return self.sigmoid(conclusion)
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())
    
    def prediction(self, dataset):
        preds = []
        golds = []
        self.eval()
        for j in range(0, len(dataset), self.args.batch_size):
            src, seg, label, mask_src = Batch(dataset, j, self.args, self.lang2pad).get()
            preds += self.predict(src, seg, mask_src)
            golds += label.cpu().data.numpy().tolist()
        return f1_score(golds, preds), accuracy_score(golds, preds)

    def predict(self, src, seg, mask_src):
        output = self.forward(src, seg, mask_src)
        prediction = output.cpu().data.numpy() > 0.5
        if type (prediction) == np.bool_:
            return [int(prediction)]
        return [int(x) for x in prediction]
    
    def process(self, train_dataset, dev_dataset, test_dataset):
        """ Train the model """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        t_total = len(train_dataset) // self.args.batch_size * self.args.num_train_epochs
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Train!
        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d", len(train_dataset))
        self.logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        self.logger.info("  Total optimization steps = %d", t_total)
        self.logger.info("  Warming up = %d", self.args.warmup_steps)
        self.logger.info("  Patience  = %d", self.args.patience)

        # Added here for reproductibility
        set_seed(self.args)
        tr_loss = 0.0
        global_step = 1
        best_f1_dev = 0
        best_f1_test = 0
        cur_patience = 0
        for i in range(int(self.args.num_train_epochs)):
            random.shuffle(train_dataset)
            epoch_loss = 0.0
            for j in range(0, len(train_dataset), self.args.batch_size):
                src, seg, label, mask_src = Batch(train_dataset, j, self.args, self.lang2pad).get()
                self.train()
                loss = self.get_loss(src, seg, label, mask_src)
                loss = loss.sum()/self.args.batch_size
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                global_step += 1
            
            self.logger.info("Finish epoch = %s, loss_epoch = %s", i+1, epoch_loss/global_step)
            dev_f1, dev_acc = self.prediction(dev_dataset, self.args)
            if dev_f1 > best_f1_dev:
                best_f1_dev = dev_f1
                test_f1, test_acc = self.prediction(test_dataset, self.args)
                best_f1_test = test_f1
                cur_patience = 0
                self.logger.info("Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)
            else:
                cur_patience += 1
                if cur_patience == self.args.patience:
                    self.logger.info("Early Stopping Not Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)
                    break
                else:
                    self.logger.info("Not Better, BEST F1 in DEV = %s & BEST F1 in test = %s.", best_f1_dev, best_f1_test)

        return global_step, tr_loss / global_step, best_f1_dev, best_f1_test
