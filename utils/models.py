import random
import numpy as np
import torch
import torch.nn as nn
import adapters
from scipy.stats import spearmanr
from itertools import permutations
from adapters import ConfigUnion, LoRAConfig, PrefixTuningConfig, IA3Config, BertAdapterModel
from sklearn.metrics import f1_score, accuracy_score
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils.utils import set_seed
from utils.batch import SentimentBatch, NTPBatch, TweetOrderingBatch

class BaseModel(nn.Module):
    def __init__(self, args, logger, lang2model, lang2pad):
        super().__init__()
        self.args = args
        self.device = args.device
        self.bert = BertModel.from_pretrained(lang2model[args.bert_lang])
        self.logger = logger
        self.lang2model = lang2model
        self.lang2pad = lang2pad
        self.Batch = None
        self.adapters = args.adapters
        self.primary_metric = None

        # Initialize adapters based on specified configurations
        if self.adapters:
            adapters.init(self.bert)
            config_list = []
            for adapter_name in self.adapters:
                config = self.get_adapter_config(adapter_name)
                if config is not None:
                    config_list.append(config)
            
            if config_list:
                # Create a ConfigUnion from the list of configurations
                config_union = ConfigUnion(*config_list)
                self.bert.add_adapter("adapters", config=config_union)
                self.bert.train_adapter("adapters")

    # TODO: Adding arguments for adapters hyperparameter
    def get_adapter_config(self, adapter_name):
        if adapter_name == "lora":
            return LoRAConfig(r=self.args.lora_rank, dropout=self.args.lora_dropout, alpha=self.args.lora_alpha)
        elif adapter_name == "prefix_tuning":
            return PrefixTuningConfig(flat=False, prefix_length=10)
        elif adapter_name == "IA3":
            return IA3Config()
        else:
            self.logger.warning(f"Adapter configuration for '{adapter_name}' not found.")
            return None
    
    def forward(self, src, seg, mask_src):
        raise NotImplementedError
    
    def predict(self, src, seg, mask_src):
        raise NotImplementedError
    
    def prediction(self, dataset):
        raise NotImplementedError
    
    def get_loss(self, src, seg, label, mask_src):
        raise NotImplementedError
    
    def evaluate_and_log(self, writer, dev_dataset, test_dataset, epoch, best_metrics, cur_patience):
        dev_result = self.prediction(dev_dataset)
        
        # log all metrics returned from the model
        for metric_name, metric_value in dev_result.items():
            writer.add_scalar(f'Validation {metric_name}', metric_value, epoch)

        primary_metric_value = dev_result.get(self.primary_metric)
        best_primary_metric_value = best_metrics.get(self.primary_metric, 0)

        # check for improvement
        if primary_metric_value > best_primary_metric_value:
            best_metrics[self.primary_metric] = primary_metric_value
            test_result = self.prediction(test_dataset)
            test_primary_metric_value = test_result.get(self.primary_metric)
            best_metrics[f'test_{self.primary_metric}'] = test_primary_metric_value
            cur_patience = 0
            self.logger.info(f"Better, BEST {self.primary_metric.upper()} in DEV = {primary_metric_value} & BEST {self.primary_metric.upper()} in TEST = {test_primary_metric_value}.")
        else:
            cur_patience += 1
            if cur_patience == self.args.patience:
                self.logger.info(f"Early Stopping Not Better, BEST {self.primary_metric.upper()} in DEV = {best_metrics[self.primary_metric]} & BEST {self.primary_metric.upper()} in TEST = {best_metrics[f'test_{self.primary_metric}']}.")
                return True, best_metrics, cur_patience
            else:
                self.logger.info("Not Better.")

        return False, best_metrics, cur_patience
    
    # TODO: Use Trainer and Dataset from huggingface
    def train_model(self, train_dataset, dev_dataset, test_dataset):
        """ Train the model """
        writer = SummaryWriter(f"runs/{self.args.name}")
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
        best_metrics = {}
        cur_patience = 0
        for epoch in range(int(self.args.num_train_epochs)):
            random.shuffle(train_dataset)
            epoch_loss = 0.0
            for batch in range(0, len(train_dataset), self.args.batch_size):
                batch_data = self.Batch(train_dataset, batch, self.args, self.lang2pad).get()
                self.train()
                loss = self.get_loss(**batch_data)
                loss = loss.sum()/self.args.batch_size
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                loss.backward()

                tr_loss += loss.item()
                epoch_loss += loss.item()
                nn.utils.clip_grad_norm_(self.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.zero_grad()
                global_step += 1

                writer.add_scalar('Training Loss', loss.item(), global_step)  # log training loss
            
            self.logger.info("Finish epoch = %s, loss_epoch = %s", epoch+1, epoch_loss/global_step)
            stop_early, best_metrics, cur_patience = self.evaluate_and_log(writer, dev_dataset, test_dataset, epoch, best_metrics, cur_patience)
            
            if stop_early:
                break

        writer.close()
        return global_step, tr_loss / global_step, best_metrics

class SentimentModel(BaseModel):
    def __init__(self, args, logger, lang2model, lang2pad):
        super().__init__(args, logger, lang2model, lang2pad)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(reduction='none') 
        self.Batch = SentimentBatch
        self.primary_metric = 'f1'

    def forward(self, src, seg, mask_src):
        output = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = output.last_hidden_state
        top_vec = self.dropout(top_vec)
        top_vec *= mask_src.unsqueeze(dim=-1).float()
        top_vec = torch.sum(top_vec, dim=1) / mask_src.sum(dim=-1).float().unsqueeze(-1)
        conclusion = self.linear(top_vec).squeeze()
        return self.sigmoid(conclusion)
    
    def predict(self, src, seg, mask_src, label=None):
        output = self.forward(src, seg, mask_src)
        prediction = output.cpu().data.numpy() > 0.5
        if type (prediction) == np.bool_:
            return [int(prediction)]
        return [int(x) for x in prediction]
    
    def prediction(self, dataset):
        preds = []
        golds = []
        self.eval()
        for j in range(0, len(dataset), self.args.batch_size):
            batch_data = self.Batch(dataset, j, self.args, self.lang2pad).get()
            preds += self.predict(**batch_data)
            golds += batch_data["label"].cpu().data.numpy().tolist()
        return {"f1": f1_score(golds, preds), "acc": accuracy_score(golds, preds)}
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())

class NTPModel(BaseModel):
    def __init__(self, args, logger, lang2model, lang2pad):
        super().__init__(args, logger, lang2model, lang2pad)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss(reduction='none') 
        self.Batch = NTPBatch
        self.primary_metric = 'acc'

    def forward(self, src, seg, mask_src):
        output = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = output.last_hidden_state
        clss = top_vec[:,0,:]
        final_rep = self.dropout(clss)
        conclusion = self.linear(final_rep).squeeze()
        return self.sigmoid(conclusion)
    
    def predict(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        batch_size = output.shape[0]
        assert batch_size%4 == 0
        output = output.view(int(batch_size/4), 4)
        prediction = torch.argmax(output, dim=-1).data.cpu().numpy().tolist()
        answer = label.view(int(batch_size/4), 4)
        answer = torch.argmax(answer, dim=-1).data.cpu().numpy().tolist()
        return answer, prediction
    
    def prediction(self, dataset):
        preds = []
        golds = []
        self.eval()
        assert len(dataset)%4==0
        assert self.args.batch_size%4==0
        for j in range(0, len(dataset), self.args.batch_size):
            batch_data = self.Batch(dataset, j, self.args, self.lang2pad).get()
            answer, prediction = self.predict(**batch_data)
            golds += answer
            preds += prediction
        return {"acc": accuracy_score(golds, preds)}
    
    def get_loss(self, src, seg, label, mask_src):
        output = self.forward(src, seg, mask_src)
        return self.loss(output, label.float())

class TweetOrderingModel(BaseModel):
    def __init__(self, args, logger, lang2model, lang2pad):
        super().__init__(args, logger, lang2model, lang2pad)
        self.linear = nn.Linear(self.bert.config.hidden_size, 5)
        self.dropout = nn.Dropout(0.2)
        self.loss = nn.CrossEntropyLoss(ignore_index=5, reduction='sum')
        self.Batch = TweetOrderingBatch
        self.primary_metric = 'rank_corr'

    def forward(self, src, seg, mask_src, cls_id, cls_mask):
        output = self.bert(input_ids=src, token_type_ids=seg, attention_mask=mask_src)
        top_vec = output.last_hidden_state
        
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), cls_id] #batch_size * 5 * dim
        sents_vec = sents_vec * cls_mask[:, :, None].float()
        final_rep = self.dropout(sents_vec)
        conclusion = self.linear(final_rep) #batch_size * 5 * 5
        return conclusion #batch_size * 5 * 5
    
    def compute(self, matrix, length):
        ids = list(permutations(np.arange(length),length))
        ids = [list(i) for i in ids]
        maxs = []; max_score = 0
        for x in ids:
            score = 0.0
            for j, i in enumerate(x):
                score += matrix[j][i]
            if score > max_score:
                max_score = score
                maxs = x
        return maxs

    def predict(self, src, seg, mask_src, label, cls_id, cls_mask):
        output = self.forward(src, seg, mask_src, cls_id, cls_mask)
        batch_size = output.shape[0]
        cors = []
        for idx in range(batch_size):
            limit = cls_mask[idx].sum()
            cur_output = output[idx][:limit]
            cur_prediction = torch.nn.Softmax(dim=-1)(cur_output.masked_fill(cls_mask[idx]==0, -np.inf))
            
            pred_rank = self.compute(cur_prediction.data.cpu().tolist(), limit.item())
            gold_rank = label[idx].data.cpu().tolist()[:limit]
            coef, _ = spearmanr(pred_rank, gold_rank)
            cors.append(coef)
        return cors
    
    def prediction(self, dataset):
        rank_cors = []
        self.eval()
        for j in range(0, len(dataset), self.args.batch_size):
            batch_data = self.Batch(dataset, j, self.args, self.lang2pad).get()
            cors = self.predict(**batch_data)
            rank_cors += cors
        return {"rank_corr" : np.mean(rank_cors)}
    
    def get_loss(self, src, seg, label, mask_src, cls_id, cls_mask):
        output = self.forward(src, seg, mask_src, cls_id, cls_mask)
        return self.loss(output.view(-1,5), label.view(-1))
