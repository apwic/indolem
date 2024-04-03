from transformers import BertTokenizer, AlbertTokenizer

class BaseData():
    def __init__(self, args, lang2model) -> None:
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'

        if args.bert_lang == 'id' or args.bert_lang == 'multi':
            self.tokenizer = BertTokenizer.from_pretrained(lang2model[args.bert_lang], do_lower_case=True)
            self.sep_vid = self.tokenizer.vocab[self.sep_token]
            self.cls_vid = self.tokenizer.vocab[self.cls_token]
            self.pad_vid = self.tokenizer.vocab[self.pad_token]
        elif args.bert_lang == 'my':
            self.tokenizer = AlbertTokenizer.from_pretrained(lang2model[args.bert_lang],
                    unk_token = '[UNK]', pad_token='[PAD]', do_lower_case=False)
            self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
            self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
            self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)
    
    def preprocess_one(self):
        raise NotImplementedError
    
    def preprocess(self):
        raise NotImplementedError

class SentimentData(BaseData):
    def __init__(self, args, lang2model) -> None:
        super().__init__(args, lang2model)
        self.MAX_TOKEN = args.max_token

    def preprocess_one(self, src_txt, label):
        src_subtokens = [self.cls_token] + self.tokenizer.tokenize(src_txt) + [self.sep_token]        
        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        
        if len(src_subtoken_idxs) > self.MAX_TOKEN:
            src_subtoken_idxs = src_subtoken_idxs[:self.MAX_TOKEN]
            src_subtoken_idxs[-1] = self.sep_vid
        else:
            src_subtoken_idxs += [self.pad_vid] * (self.MAX_TOKEN-len(src_subtoken_idxs))
        segments_ids = [0] * len(src_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, src_txts, labels):
        assert len(src_txts) == len(labels)
        output = []
        for idx in range(len(src_txts)):
            output.append(self.preprocess_one(src_txts[idx], labels[idx]))
        return output

class NTPData(BaseData):
    def __init__(self, args, lang2model) -> None:
        super().__init__(args, lang2model)
        self.MAX_TOKEN_PREMISE = args.max_token_premise
        self.MAX_TOKEN_NEXT = args.max_token_nextTw

    def preprocess_one(self, premise, nextTw, label):
        premise_subtokens = [self.cls_token] + self.tokenizer.tokenize(premise) + [self.sep_token]        
        premise_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(premise_subtokens)
        if len(premise_subtoken_idxs) > self.MAX_TOKEN_PREMISE:
            premise_subtoken_idxs = premise_subtoken_idxs[len(premise_subtoken_idxs)-self.MAX_TOKEN_PREMISE:]
            premise_subtoken_idxs[0] = self.cls_vid

        nextTw_subtokens = self.tokenizer.tokenize(nextTw) + [self.sep_token]
        nextTw_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(nextTw_subtokens)
        if len(nextTw_subtoken_idxs) > self.MAX_TOKEN_NEXT:
            nextTw_subtoken_idxs = nextTw_subtoken_idxs[:self.MAX_TOKEN_NEXT]
            nextTw_subtoken_idxs[-1] = self.sep_vid

        src_subtoken_idxs = premise_subtoken_idxs + nextTw_subtoken_idxs
        segments_ids = [0] * len(premise_subtoken_idxs) + [1] * len(nextTw_subtoken_idxs)
        assert len(src_subtoken_idxs) == len(segments_ids)
        return src_subtoken_idxs, segments_ids, label
    
    def preprocess(self, premises, nextTws, labels):
        assert len(premises) == len(nextTws) == len(labels)
        output = []
        for idx in range(len(premises)):
            output.append(self.preprocess_one(premises[idx], nextTws[idx], labels[idx]))
        return output