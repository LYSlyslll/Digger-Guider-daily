import copy
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from common.utils import pprint, AverageMeter
from common.functions import get_loss_fn, get_metric_fn

from sacred import Ingredient
model_ingredient = Ingredient('cnn_rnn_v2')


@model_ingredient.config
def model_config():
    # architecture
    input_shape = [6, 20]  # [因子数, 时间窗口天数]
    rnn_type = 'LSTM'  # LSTM/GRU
    rnn_layer = 2
    hid_size = 64
    dropout = 0
    dropout_2 = 0.1
    # optimization
    optim_method = 'Adam'
    optim_args = {'lr': 1e-3}
    optim_args_2 = {'lr': 1e-4, 'weight_decay': 1e-6}
    loss_fn = 'mse'
    eval_metric = 'corr'
    verbose = 500
    max_steps = 50
    early_stopping_rounds = 5
    output_path = "/your_own_path/out"


class Day_Model_1(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout=0,
                 optim_method='Adam',
                 optim_args={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr'):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_size = input_shape[0]
        self.input_day = input_shape[1]
        self.dropout = dropout
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)
        self.net = nn.Sequential()
        self.net.add_module('fc_in', nn.Linear(in_features=self.input_size, out_features=self.hid_size))
        self.net.add_module('act', nn.Tanh())
        self.rnn = klass(input_size=self.hid_size,
                         hidden_size=self.hid_size,
                         num_layers=self.rnn_layer,
                         batch_first=True,
                         dropout=self.dropout)
        self.fc_final = nn.Linear(in_features=self.hid_size, out_features=1)

    def forward(self, inputs):
        inputs = inputs.view(-1, self.input_size, self.input_day)
        inputs = inputs.permute(0, 2, 1)  # [batch, input_size, seq_len] -> [batch, seq_len, input_size]
        fc_hid = self.net(inputs)
        out, _ = self.rnn(fc_hid)
        out_seq = self.fc_final(out[:, -1, :])  # [batch, input_day, hid_size] -> [batch, 1]

        return fc_hid, out_seq[..., 0]  # fc_hid:[batch, input_day, hid_size]


    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            run=None,
            max_steps=50,
            early_stopping_rounds=10,
            verbose=100):

        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())
        for step in range(max_steps):

            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            train_hids = dict()
            for i, (idx, data, label) in enumerate(train_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                train_hid, pred = self(data)
                loss = self.loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                train_loss.update(loss_, len(data))
                train_eval.update(eval_)
                train_hids[idx] = train_hid.cpu().detach()
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            valid_hids = dict()
            for i, (idx, data, label) in enumerate(valid_set):
                data = torch.tensor(data, dtype=torch.float)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    valid_hid, pred = self(data)

                loss = self.loss_fn(pred, label)
                valid_loss_ = loss.item()
                valid_eval_ = self.metric_fn(pred, label).item()
                valid_loss.update(valid_loss_, len(data))
                valid_eval.update(valid_eval_)
                valid_hids[idx] = valid_hid.cpu().detach()
            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)
            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
        # restore
        self.load_state_dict(best_params)
        return train_hids, valid_hids  # train_hid: [batch, input_day, hid_size]

    def predict(self, test_set):
        self.eval()
        preds = []
        test_hids = dict()
        for _, (idx, data, _) in enumerate(test_set):
            data = torch.tensor(data, dtype=torch.float)
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                test_hid, pred = self(data)
            test_hids[idx] = test_hid.cpu().detach()
            preds.append(pred.cpu().detach().numpy())
        return np.concatenate(preds), test_hids

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)


class Day_Model_2(nn.Module):
    @model_ingredient.capture
    def __init__(self,
                 input_shape,
                 rnn_type='LSTM',
                 rnn_layer=2,
                 hid_size=64,
                 dropout_2=0,
                 optim_method='Adam',
                 optim_args_2={'lr': 1e-3},
                 loss_fn='mse',
                 eval_metric='corr'):

        super().__init__()

        # Architecture
        self.hid_size = hid_size
        self.input_shape = input_shape
        self.input_size = input_shape[0]
        self.input_day = input_shape[1]
        self.dropout = dropout_2
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type

        self._build_model()
        # Optimization
        self.optimizer = getattr(optim, optim_method)(self.parameters(),
                                                      **optim_args_2)
        self.loss_fn = get_loss_fn(loss_fn)
        self.metric_fn = get_metric_fn(eval_metric)

        if torch.cuda.is_available():
            self.cuda()

    def _build_model(self):

        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception:
            raise ValueError('unknown rnn_type `%s`' % self.rnn_type)

        self.net = nn.Sequential()
        self.net.add_module('fc_in', nn.Linear(in_features=self.hid_size, out_features=self.hid_size))
        self.net.add_module('act', nn.Tanh())

        self.rnn = klass(input_size=self.hid_size + self.input_size,
                hidden_size=self.hid_size + self.input_size,
                num_layers=self.rnn_layer,
                batch_first=True,
                dropout=self.dropout)
        self.fc_final = nn.Linear(in_features=self.hid_size + self.input_size, out_features=1)

    def forward(self, inputs, data):
        fc_hid = self.net(inputs)
        out, _ = self.rnn(torch.cat([fc_hid, data], dim=2))
        out_seq = self.fc_final(out[:, -1, :])  # [batch, seq_len, num_directions * hidden_size] -> [batch, 1]
        return fc_hid, out_seq[..., 0]

    @model_ingredient.capture
    def fit(self,
            train_set,
            valid_set,
            train_day_reps,
            valid_day_reps,
            run=None,
            max_steps=100,
            early_stopping_rounds=10,
            verbose=100):
        best_score = np.inf
        stop_steps = 0
        best_params = copy.deepcopy(self.state_dict())

        for step in range(max_steps):

            pprint('Step:', step)
            if stop_steps >= early_stopping_rounds:
                if verbose:
                    pprint('\tearly stop')
                break
            stop_steps += 1
            # training
            self.train()
            tune_train_reps = dict()
            train_loss = AverageMeter()
            train_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(train_set):
                train_day_rep = train_day_reps[idx]
                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_size, self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    train_day_rep, label, data = train_day_rep.cuda(), label.cuda(), data.cuda()
                tune_train_rep, pred = self(train_day_rep, data)
                loss = self.loss_fn(pred, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_train_reps[idx] = tune_train_rep.cpu().detach()
                train_loss.update(loss_, len(label))
                train_eval.update(eval_)
                if verbose and i % verbose == 0:
                    pprint('iter %s: train_loss %.6f, train_eval %.6f' %
                           (i, train_loss.avg, train_eval.avg))
            # evaluation
            self.eval()
            tune_valid_reps = dict()
            valid_loss = AverageMeter()
            valid_eval = AverageMeter()
            for i, (idx, data, label) in enumerate(valid_set):
                valid_day_rep = valid_day_reps[idx]
                data = torch.tensor(data, dtype=torch.float)
                data = data.view(-1, self.input_shape[0], self.input_day)
                data = data.permute(0, 2, 1)
                label = torch.tensor(label, dtype=torch.float)
                if torch.cuda.is_available():
                    valid_day_rep, label, data = valid_day_rep.cuda(), label.cuda(), data.cuda()
                with torch.no_grad():
                    tune_valid_rep, pred = self(valid_day_rep, data)
                loss = self.loss_fn(pred, label)
                loss_ = loss.item()
                eval_ = self.metric_fn(pred, label).item()
                tune_valid_reps[idx] = tune_valid_rep.cpu().detach()
                valid_loss.update(loss_, len(label))
                valid_eval.update(eval_)

            if run is not None:
                run.add_scalar('Train/Loss', train_loss.avg, step)
                run.add_scalar('Train/Eval', train_eval.avg, step)
                run.add_scalar('Valid/Loss', valid_loss.avg, step)
                run.add_scalar('Valid/Eval', valid_eval.avg, step)

            if verbose:
                pprint("current step: train_loss {:.6f}, valid_loss {:.6f}, "
                       "train_eval {:.6f}, valid_eval {:.6f}".format(
                           train_loss.avg, valid_loss.avg, train_eval.avg,
                           valid_eval.avg))
            if valid_eval.avg < best_score:
                if verbose:
                    pprint(
                        '\tvalid update from {:.6f} to {:.6f}, save checkpoint.'
                        .format(best_score, valid_eval.avg))
                best_score = valid_eval.avg
                stop_steps = 0
                best_params = copy.deepcopy(self.state_dict())
                best_tune_train_reps = tune_train_reps
                best_tune_valid_reps = tune_valid_reps

        # restore
        self.load_state_dict(best_params)
        return best_tune_train_reps, best_tune_valid_reps

    def predict(self, test_set, test_day_reps):
        self.eval()
        preds = []
        tune_test_reps = dict()
        for _, (idx, data, _) in enumerate(test_set):
            test_day_rep = test_day_reps[idx]

            data = torch.tensor(data, dtype=torch.float)
            data = data.view(-1, self.input_size, self.input_day)
            data = data.permute(0, 2, 1)
            if torch.cuda.is_available():
                test_day_rep, data = test_day_rep.cuda(), data.cuda()
            with torch.no_grad():
                tune_test_rep, pred = self(test_day_rep, data)
                tune_test_reps[idx] = tune_test_rep.cpu().detach()
                preds.append(pred.cpu().detach().numpy())
        return np.concatenate(preds), tune_test_reps

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)

    def load(self, model_path, strict=True):
        self.load_state_dict(torch.load(model_path), strict=strict)
