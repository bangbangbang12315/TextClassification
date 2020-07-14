from collections import defaultdict
from tqdm import tqdm
class Genindex():
    def __init__(self):
        self.word2idx = defaultdict()
        self.idx2word = defaultdict()
    def get_vocab(self, data,vocab_num=40000):
        vocab_dict = defaultdict(int)
        for text in tqdm(data):
            words = text.strip().split(' ')
            for w in words:
                if not w:
                    continue
                else:
                    vocab_dict[w] += 1
        vocab = [k for k, _ in sorted(vocab_dict.items(), key=lambda x: -x[1])]
        vocab = vocab[:vocab_num]
        return vocab 

    def gen_index(self, data,vocab):
        index = [[] for i in range(len(vocab))]
        for i, w in enumerate(vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w
        for idx, text in tqdm(enumerate(data)):
            words = text.strip().split(' ')
            for w in words:
                if w in vocab:
                    if idx not in index[self.word2idx[w]]:
                        index[self.word2idx[w]].append(idx)
                else:
                    continue
        return index

    def store_index(self, index, fp):
        with open(fp, 'w') as f:
            for idx in tqdm(index):
                idx = list(map(lambda x: str(x), idx))
                f.write(' '.join(idx) + '\n')
        f.close()

    def store_vocab(self, fp, data):
        print(f"Save {fp}")
        with open(fp, 'w') as f:
            for text in tqdm(data):
                f.write(text + '\n')

    def index_pipeline(self, data, vocab_fp, index_fp, vocab_num=40000):
        vocab = self.get_vocab(data, vocab_num)
        self.store_vocab(vocab_fp, vocab)
        index = self.gen_index(data, vocab)
        self.store_index(index, index_fp)
