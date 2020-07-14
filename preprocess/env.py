import pkuseg
import os
from tqdm import  tqdm
from  multi import *
from collections import defaultdict
from retrival import Retrieval
from prepross import Prepross
from genindex import Genindex

def storedata_list(data,fp):
    with open(fp, 'w') as fout:
        for text in tqdm(data):
            fout.write(' '.join(text) + '\n')
    fout.close()


def loaddata(fp):
    data = []
    with open(fp,'r') as ftext:
        for text in tqdm(ftext):
            data.append(text)
        return data

def prepross_query(query, cutter):
    text = cutter.cut(query)
    return text

if __name__ == "__main__":
    data_dir = '../data'
    fsrc = os.path.join(data_dir, '带标签短信.txt')
    fdata = os.path.join(data_dir, 'clean_data.txt')
    sw_fp = os.path.join(data_dir, 'stopwords.txt')
    vocab_fp = os.path.join(data_dir, 'vocab')
    index_fp = os.path.join(data_dir, 'index')
    P = Prepross(sw_fp)
    worker = Worker(fsrc,fdata, P.prepross)
    mp = MultiProcessor(worker, 8)
    mp.run()
    print("All Processes Done.")
    worker.merge_result(keep_pid_file=False)
    # data = loaddata(fdata)
    # gen = Genindex()
    # gen.index_pipeline(data,vocab_fp,index_fp,40000)

    # R = Retrieval()
    # R.load_src(fdata)
    # R.load_vocab_index(vocab_fp, index_fp)
    # cutter = pkuseg.pkuseg()
    # while True:
    #     query = input()
    #     if query == 'exit':
    #         break
    #     query = prepross_query(query, cutter)
    #     result = R.search(query)
    #     if result[0][0] != 'N':
    #         print('已查询到{}条相关文档'.format(str(len(result))))
    #     else:
    #         print('已查询到{}条相关文档'.format(str(0)))
    #     for r in result:
    #         print(r)
    #     print('==' * 30)
    





