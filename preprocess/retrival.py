from collections import defaultdict
from tqdm import tqdm

class Retrieval():
    def __init__(self):
        self.word2idx = defaultdict()
        self.idx2word = defaultdict()
        self.vocab = []
        self.index = []
        self.srcdata = []
    def load_src(self, src_fp):
        with open(src_fp, 'r') as f_src:
            for text in tqdm(f_src):
                self.srcdata.append(''.join(text.strip()))

    def load_vocab_index(self, vocab_fp, index_fp):
        with open(vocab_fp, 'r') as fp_v, open(index_fp, 'r') as fp_i:
            for word in tqdm(fp_v):
                self.vocab.append(word.strip())
            for idx in tqdm(fp_i):
                idxlist = idx.strip().split(' ')
                idxlist = list(map(lambda x: int(x), idxlist))
                self.index.append(idxlist)
            for i, w in enumerate(self.vocab):
                self.word2idx[w] = i
                self.idx2word[i] = w

    def mergetwo(self, p1, p2):
        result = []
        idx1, idx2 = 0, 0
        if not p1:
            return p2
        if not p2:
            return p1
        while idx1 < len(p1) and idx2 < len(p2):
            if p1[idx1] == p2[idx2]:
                result.append(p1[idx1])
                idx1 += 1
                idx2 += 1
            elif p1[idx1] < p2[idx2]:
                idx1 += 1
            else:
                idx2 += 1
        return result
    def intersect(self, candidate, result=None):
        terms = sorted(candidate, key = lambda x: len(x))
        result = []
        for t in terms:
            result = self.mergetwo(result, t)
        return result

    def search(self, keywords):
        ans = []
        candidate = []
        result = []
        for key in keywords:
            if key in self.vocab:
                candidate.append(self.index[self.word2idx[key]])
            else:
                return ["No Answer"]
        if candidate:
            ans = self.intersect(candidate)
        if ans:
            for i in ans:
                result.append(self.srcdata[i])
        return result if result else ["No Answer"]
