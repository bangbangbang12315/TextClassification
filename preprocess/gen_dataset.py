from tqdm import tqdm
import os
import random
def load_data(data_fp):
    data = []
    with open(data_fp, 'r') as f_data:
        for text in tqdm(f_data):
            data.append(text.strip())
    return data
def statistic(data):
    cnt = 0
    for text in data:
        label = text.split('\t')[0]
        if label == '0':
            cnt += 1
    print("非垃圾短信占比为:{}".format(cnt / len(data)))


def save_data(data, save_fp):
    with open(save_fp, 'w') as f_save:
        for text in tqdm(data):
            f_save.write(text + '\n')

def split_train_dev_test(data, dev_num=10000, test_num=10000, shuffle=True):
    if shuffle:
        random.shuffle(data)
    data_dir = '../data'
    dev_fp = os.path.join(data_dir, 'dev.txt')
    test_fp = os.path.join(data_dir, 'test.txt')
    train_fp = os.path.join(data_dir, 'train.txt')
    dev = data[:dev_num]
    statistic(dev)
    save_data(dev,dev_fp)
    test = data[dev_num:dev_num+test_num]
    statistic(test)
    save_data(test, test_fp)
    train = data[dev_num+test_num:]
    statistic(train)
    save_data(train, train_fp)

if __name__ == '__main__':
    data_dir = '../data'
    data_fp = os.path.join(data_dir, 'clean_data.txt')
    data = load_data(data_fp)
    split_train_dev_test(data)
