import pkuseg
from tqdm import tqdm
class Prepross():
    def __init__(self,sw_fp):
        self.cutter = pkuseg.pkuseg()
        self.stopwords = self.load_stop_words(sw_fp)

    def load_stop_words(self,fp):
        stopwords = []
        with open(fp, 'r') as sw:
            for word in sw:
                stopwords.append(word.strip())
        return stopwords

    def cut_text(self,text):
        return self.cutter.cut(text)

    def filter_stop_words(self, line):
        newline = []
        for word in line:
            if word in self.stopwords:
                pass
            else:
                newline.append(word)
        return newline 
    
    def prepross(self,text):
        label, text = text.strip().split('\t')
        line = self.cut_text(text)
        line = self.filter_stop_words(line)
        if line:
            return label, line