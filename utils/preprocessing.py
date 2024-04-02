from transformers import BertTokenizer, AlbertTokenizer

class BertData():
    def __init__(self, args, lang2model) -> None:
        self.MAX_TOKEN = args.max_token
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