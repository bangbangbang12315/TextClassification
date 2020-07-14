import random
import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

class TranslateData():
    def __init__(self, pad=0):
        self.pad = pad

    def collate_fn(self, batch):
        src = list(map(lambda x: x['src'], batch))
        tgt = list(map(lambda x: x['tgt'], batch))
        src_len = list(map(lambda x: x['src_len'], batch))
        src = torch.transpose(pad_sequence(src, padding_value=self.pad), 0, 1)
        src_len = torch.stack(src_len)
        tgt = torch.stack(tgt)
        return {'src': src, 'tgt': tgt, 'src_len': src_len}

    def translate_data(self, subs, obj):
        import re
        import unicodedata
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )

        def normalizeString(s):
            s = unicodeToAscii(s.lower().strip())
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s

        tgt, src = subs
        if len(src) > obj.max_src_length:
            return None
        src_length = len(src)
        src_ids = [obj.src_vocab.word2idx[w] for w in src]
        tgt = int(tgt)
        return {"src": torch.LongTensor(src_ids), 
                "tgt": torch.LongTensor([tgt]), 
                "src_len": torch.LongTensor([src_length])}


class Short_text_Dataset(Dataset):
    def __init__(self, data_fp, transform_fuc, src_vocab, max_src_length):
        self.datasets = []
        self.src_vocab = src_vocab
        self.max_src_length = max_src_length
        
        loaded = 0
        data_monitor = 0
        with open(data_fp, 'r') as f:
            for line in tqdm(f, desc="Load Data: "):
                subs = line.strip().split('\t')
                loaded += 1
                if not data_monitor: data_monitor = len(subs)
                else: assert data_monitor == len(subs)
                item = transform_fuc(subs, self)
                if item: self.datasets.append(item)

        print(f"{loaded} paris loaded. {len(self.datasets)} are valid. Rate {1.0 * len(self.datasets)/loaded:.4f}")

    
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx]