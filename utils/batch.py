import torch
    
class SentimentBatch():
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
        return {
            "src": self.src,
            "seg": self.seg,
            "label": self.label,
            "mask_src": self.mask_src,
        }
    
class NTPBatch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    def __init__(self, data, idx, args, lang2pad):
        PAD_ID=lang2pad[args.bert_lang]
        cur_batch = data[idx:idx+args.batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src!=PAD_ID)
        
        self.src = src.to(args.device)
        self.seg= seg.to(args.device)
        self.label = label.to(args.device)
        self.mask_src = mask_src.to(args.device)

    def get(self):
        return {
            "src": self.src,
            "seg": self.seg,
            "label": self.label,
            "mask_src": self.mask_src,
        }
    
class TweetOrderingBatch():
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    # do padding here
    def __init__(self, data, idx, args, lang2pad):
        PAD_ID=lang2pad[args.bert_lang]
        cur_batch = data[idx:idx+args.batch_size]
        src = torch.tensor(self._pad([x[0] for x in cur_batch], PAD_ID))
        seg = torch.tensor(self._pad([x[1] for x in cur_batch], PAD_ID))
        cls_id = torch.tensor([x[2] for x in cur_batch])
        cls_mask = torch.tensor([x[3] for x in cur_batch])
        label = torch.tensor([x[4] for x in cur_batch])
        mask_src = 0 + (src!=PAD_ID)
        
        self.src = src.to(args.device)
        self.seg= seg.to(args.device)
        self.label = label.to(args.device)
        self.mask_src = mask_src.to(args.device)
        self.cls_id = cls_id.to(args.device)
        self.cls_mask = cls_mask.to(args.device)

    def get(self):
        return {
            "src": self.src,
            "seg": self.seg,
            "label": self.label,
            "mask_src": self.mask_src,
            "cls_id": self.cls_id,
            "cls_mask": self.cls_mask,
        }