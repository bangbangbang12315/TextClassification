import torch
import torch.optim as optim
import torch.nn as nn
from vocab import VocabField
import os
import random
from tqdm import tqdm
import logging
from torch.utils.data.dataloader import DataLoader
import torch.utils.data.distributed as dist
from optim import Optimizer
from supervised_trainer import SupervisedTrainer
from data import *
from textrcnn import TextRCNN
from transformer import TransformerModel
from configParser import opt
from textrcnn import TextRCNN
from rnn import RNN
import numpy as np

def get_last_checkpoint(model_dir):
    checkpoints_fp = os.path.join(model_dir, "checkpoints")
    try:
        with open(checkpoints_fp, 'r') as f:
            checkpoint = f.readline().strip()
    except:
        return None
    return checkpoint

if opt.random_seed is not None: 
    torch.cuda.manual_seed_all(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)

LOG_FORMAT = '%(asctime)s %(levelname)-8s %(message)s'
if opt.phase == 'train':
    logging.basicConfig(format=LOG_FORMAT, 
                        level=getattr(logging, opt.log_level.upper()),
                        filename=os.path.join(opt.model_dir, opt.log_file),
                        filemode='a' if opt.resume else 'w')
    logger = logging.getLogger('train')
    logger.info(f"Train Log")
    logger.info(opt)
else:
    logging.basicConfig(format=LOG_FORMAT, 
                        filename=os.path.join(opt.model_dir, opt.log_file),
                        level=getattr(logging, opt.log_level.upper()))
    logger = logging.getLogger('test')
    logger.info(f"Test Log")
    logger.info(opt)

device = torch.device(f"cuda:{opt.device}" if opt.device.isdigit() else 'cpu')
if __name__ == '__main__':
    src_vocab_list = VocabField.load_vocab(opt.src_vocab_file)
    src_vocab = VocabField(src_vocab_list, vocab_size=opt.src_vocab_size)

    if opt.model_name == 'textrcnn':
        model = TextRCNN(opt.src_vocab_size, opt.embedding_size, opt.hidden_size)
    elif opt.model_name == 'rnn':
        model = RNN(opt.src_vocab_size, opt.embedding_size, opt.hidden_size)
    elif opt.model_name == 'transformer':
        model = TransformerModel(opt.src_vocab_size, opt.embedding_size, opt.hidden_size)

    last_checkpoint = None
    if opt.resume and not opt.load_checkpoint:
        last_checkpoint = get_last_checkpoint(opt.best_model_dir)
    if last_checkpoint:
        opt.load_checkpoint = os.path.join(opt.model_dir, last_checkpoint)
        opt.skip_steps = int(last_checkpoint.strip('.pt').split('/')[-1])

    if opt.load_checkpoint:
        model.load_state_dict(torch.load(opt.load_checkpoint))
        opt.skip_steps = int(opt.load_checkpoint.strip('.pt').split('/')[-1])
        logger.info(f"\nLoad from {opt.load_checkpoint}\n")
    else:
        for param in model.parameters():
            param.data.uniform_(-opt.init_weight, opt.init_weight)

    optimizer = optim.Adam(model.parameters())
    optimizer = Optimizer(optimizer, max_grad_norm=opt.clip_grad)
    loss = nn.CrossEntropyLoss()
    model = model.to(device)
    loss = loss.to(device)

    if opt.phase == 'train':
        trans_data = TranslateData()
        train_set = Short_text_Dataset(opt.train_path,
                                  trans_data.translate_data,
                                  src_vocab,
                                  max_src_length=opt.max_src_length)
        train = DataLoader(train_set, 
                           batch_size=opt.batch_size, 
                           shuffle=False,
                           drop_last=True,
                           collate_fn=trans_data.collate_fn)

        dev_set = Short_text_Dataset(opt.dev_path,
                                trans_data.translate_data,
                                src_vocab,
                                max_src_length=opt.max_src_length)
        dev = DataLoader(dev_set, 
                        batch_size=opt.batch_size, 
                        shuffle=False, 
                        collate_fn=trans_data.collate_fn)

        t = SupervisedTrainer(loss=loss, 
                                model_dir=opt.model_dir,
                                best_model_dir=opt.best_model_dir,
                                batch_size=opt.batch_size,
                                checkpoint_every=opt.checkpoint_every,
                                print_every=opt.print_every,
                                max_epochs=opt.max_epochs,
                                max_steps=opt.max_steps,
                                max_checkpoints_num=opt.max_checkpoints_num,
                                device=device,
                                logger=logger)

        model = t.train(model, 
                            data=train,
                            start_step=opt.skip_steps, 
                            dev_data=dev,
                            optimizer=optimizer)

    if opt.phase == 'test':
        trans_data = TranslateData()
        test_set = Short_text_Dataset(opt.test_path,
                                  trans_data.translate_data,
                                  src_vocab,
                                  max_src_length=opt.max_src_length)
        test = DataLoader(test_set, 
                           batch_size=opt.batch_size, 
                           shuffle=False,
                           drop_last=True,
                           collate_fn=trans_data.collate_fn)

        with torch.no_grad():
            match = 0
            total = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0

            for batch in test:
                src_variables = batch['src'].to(device)
                tgt_variables = batch['tgt'].view(-1).to(device)
                src_lens = batch['src_len'].view(-1).to(device)

                output = model(src_variables, src_lens.tolist(), tgt_variables)
                lossb = loss(output, tgt_variables)
                pred =  torch.argmax(output, dim=1)
                for pre,tgt in zip(pred, tgt_variables):
                    total += 1
                    pre = pre.item()
                    tgt = tgt.item()
                    if pre == tgt:
                        match += 1
                    if pre == 0 and tgt == 0:
                        true_positives += 1
                    if pre == 0 and tgt == 1:
                        false_positives += 1
                    if pre == 1 and tgt == 0:
                        false_negatives += 1

        if total == 0:
            accuracy = float('nan')
            F1 = float('nan')
        else:
            accuracy = match / total
            precision = true_positives / (true_positives + false_positives + 1)
            recall    = true_positives / (true_positives + false_negatives + 1)
            F1 = 2 * (precision * recall) / (precision + recall)

        logger.info(f"{opt.model_name}, accuracy: {accuracy}, F1: {F1}")
        
        


        
    
    


