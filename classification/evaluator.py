from __future__ import print_function, division

import torch
import torch.nn as nn
import textrcnn


class Evaluator(object):
    """ Class to evaluate models with given datasets.

    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=nn.CrossEntropyLoss(), batch_size=64, device=None):
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.

        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        device = self.device

        match = 0
        total = 0

        with torch.no_grad():
            for batch in data:
                src_variables = batch['src'].to(device)
                tgt_variables = batch['tgt'].view(-1).to(device)
                src_lens = batch['src_len'].view(-1).to(device)

                output = model(src_variables, src_lens.tolist(), tgt_variables)
                lossb = loss(output, tgt_variables)

                # Evaluation
                pred =  torch.argmax(output, dim=1)
                for pre,tgt in zip(pred, tgt_variables):
                    total += 1
                    if pre == tgt:
                        match += 1
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return lossb, accuracy
