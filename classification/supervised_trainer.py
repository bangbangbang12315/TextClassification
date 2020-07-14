from __future__ import division
import os
import logging
import random
import time
import torch
from torch import optim
import torch.nn as nn
from evaluator import Evaluator
class SupervisedTrainer(object):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.

    Args:
        model_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, 
                 model_dir='experiment',
                 best_model_dir='experiment/best',
                 loss=nn.CrossEntropyLoss(), 
                 batch_size=64, 
                 checkpoint_every=100, 
                 print_every=100, 
                 max_epochs=5,
                 max_steps=10000, 
                 max_checkpoints_num=5, 
                 device=None,
                 logger=None):
        self._trainer = "Simple Trainer"
        self.loss = loss
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_checkpoints_num = max_checkpoints_num
        self.device = device
        self.best_acc = 0.0
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, device=device)

        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.getcwd(), model_dir)
        self.model_dir = model_dir
        
        if not os.path.isabs(best_model_dir):
            best_model_dir = os.path.join(os.getcwd(), best_model_dir)
        self.best_model_dir = best_model_dir

        if not os.path.exists(self.best_model_dir): os.makedirs(self.best_model_dir)
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)

        self.model_checkpoints = []
        self.best_model_checkpoints = []

        self.logger = logger if logger is not None else logging.getLogger(__name__)

    def save_model(self, model, steps, dev_acc=None):
        model_fn = f"{steps}.pt"
        model_fp = os.path.join(self.model_dir, model_fn)

        # save model checkpoints
        while len(self.model_checkpoints) >= self.max_checkpoints_num:
            os.system(f"rm {self.model_checkpoints[0]}")
            self.model_checkpoints = self.model_checkpoints[1:]
        torch.save(model.state_dict(), model_fp)
        self.model_checkpoints.append(model_fp)

        # update checkpoints file
        with open(os.path.join(self.model_dir, "checkpoints"), 'w') as f:
            f.write('\n'.join(self.model_checkpoints[::-1]))

        if not dev_acc: return None
        # save best model checkpoints
        if dev_acc > self.best_acc:
            self.best_acc = dev_acc
            self.logger.info(f"Best model dev acc {dev_acc}.")
            while len(self.best_model_checkpoints) >= self.max_checkpoints_num:
                os.system(f"rm {self.best_model_checkpoints[0]}")
                self.best_model_checkpoints = self.best_model_checkpoints[1:]
            
            best_model_fp = os.path.join(self.best_model_dir, model_fn)
            os.system(f"cp {model_fp} {best_model_fp}")
            self.best_model_checkpoints.append(best_model_fp)
        else:
            self.logger.info(f"Current learning rate: {self.optimizer.optimizer.param_groups[0]['lr']}")


    def _train_batch(self, input_variable, input_lengths, label, model):
        loss = self.loss
        # Forward propagation
        output = model(input_variable, input_lengths, label)
        # Get loss
        # print(output, label)
        lossb = loss(output, label)

        # Backward propagation
        model.zero_grad()
        lossb.backward()
        self.optimizer.step()

        return lossb

    def _train_epoches(self, data, model, start_step, dev_data=None):
        device = self.device
        log = self.logger
        max_epochs = self.max_epochs
        max_steps = self.max_steps

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        step = 0
        steps_per_epoch = len(data)
        start_epoch = (start_step - step) // steps_per_epoch
        step = start_epoch * steps_per_epoch
        for batch in data:
            if step >= start_step: break
            step += 1
        if start_epoch or start_step:
            log.info(f"Resume from Epoch {start_epoch}, Step {start_step}")

        for epoch in range(start_epoch, max_epochs):
            model.train(True)
            for batch in data:
                step += 1
                src_variables = batch['src'].to(device)
                tgt_variables = batch['tgt'].view(-1).to(device)
                src_lens = batch['src_len'].view(-1).to(device)


                loss = self._train_batch(src_variables, src_lens.tolist(), tgt_variables, model)

                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = f"Process {100.0*(step%steps_per_epoch)/steps_per_epoch:.2f}% of Epoch {epoch}, Total step {step},Train: {print_loss_avg:.4f}" 
                    # if not multi_gpu or hvd.rank() == 0:
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0:
                    dev_loss = None
                    if dev_data is not None:
                        dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                        self.optimizer.update(dev_loss, epoch)
                        log_msg = f"Dev: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"
                        log.info(log_msg)
                        model.train(mode=True)

                    self.save_model(model, step, dev_acc=accuracy)

                if step >= max_steps:
                    break

            if step >= max_steps:
                log.info(f"Finish max steps {max_steps} at Epoch {epoch}.")
                break

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = f"Finished Epoch {epoch}, Train: {epoch_loss_avg:.4f}"
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += f", Dev: {dev_loss:.4f}, Accuracy: {accuracy:.4f}"
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)
            self.save_model(model, step, dev_acc=dev_loss)
            log.info(log_msg)
            log.info(f"Finish Epoch {epoch}, Total steps {step}.")

    def train(self, model, data, start_step=0, dev_data=None, optimizer=None, teacher_forcing_ratio=0):
        """ Run training for a given model.

        Args:
            model (seq2seq.models): model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data (seq2seq.dataset.dataset.Dataset): dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data (seq2seq.dataset.dataset.Dataset, optional): dev Dataset (default None)
            optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
            teacher_forcing_ratio (float, optional): teaching forcing ratio (default 0)
        Returns:
            model (seq2seq.models): trained model.
        """
        
        if optimizer is None:
            optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
        self.optimizer = optimizer

        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

        self._train_epoches(data, model, start_step, dev_data=dev_data)
        return model