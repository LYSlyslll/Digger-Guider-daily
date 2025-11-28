# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import pickle
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

import loaders as loaders_module
import models as models_module

from common.utils import (pprint, set_random_seed, create_output_path,
                          zscore, robust_zscore, count_num_params)
from common.functions import K, rmse, mae  # BUG: sacred cannot save source files used in ingredients

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('hft_pred',
                ingredients=[
                    loaders_module.daily_loader_v3.data_ingredient,
                    models_module.cnn_rnn_v2.model_ingredient
                ])

create_output_path('my_runs')
ex.observers.append(
    FileStorageObserver("my_runs", "my_runs/resource", "my_runs/source",
                        "my_runs/templete"))

def Day_model_1(output_path, day_model_1, day_train_set, day_valid_set,
                day_test_set):
    pprint('training day_model_1 ...')
    train_hids, valid_hids = day_model_1.fit(day_train_set,
                                             day_valid_set,
                                             run=None)
    _, test_hids = day_model_1.predict(day_test_set)
    pprint('inference...')
    pprint('validation set day_model_1 :')
    inference(dset=day_valid_set, model=day_model_1)
    pprint('testing set day_model_1 :')
    inference(dset=day_test_set, model=day_model_1)
    pprint('done.')

    return train_hids, valid_hids, test_hids

def Day_model_2(_run,
                output_path,
                day_model_2,
                day_train_set,
                day_valid_set,
                day_test_set,
                train_day_reps,
                valid_day_reps,
                test_day_reps,
                pred_path=None):
    pprint('training day_model_2')
    tune_train_reps, tune_valid_reps = day_model_2.fit(day_train_set,
                                                       day_valid_set,
                                                       train_day_reps,
                                                       valid_day_reps,
                                                       run=_run)

    pprint('inference...')
    pprint('validation set day_model_2:')
    inference(dset=day_valid_set,
              model=day_model_2,
              day_rep=valid_day_reps,
              pred_path=None)

    pprint('testing set day_model_2:')
    rmse, mae   = inference(dset=day_test_set,
                             model=day_model_2,
                             day_rep=test_day_reps,
                             pred_path=pred_path)
    return rmse, mae

def inference(dset, model,day_rep=None, pred_path=None):
    pred = pd.DataFrame(index=dset.index)
    pred['label'] = dset.label
    if day_rep is not None:
        pred['score'], _ = model.predict(dset, day_rep)
    else:
        pred['score'], _ = model.predict(dset)
    if pred_path is not None:
        pred.to_pickle(pred_path)
        ex.add_artifact(pred_path)

    rank_ic = pred.groupby(level='datetime').apply(
        lambda x: x.label.corr(x.score, method='spearman'))
    pprint('Rank IC:', rank_ic.mean(), ',', rank_ic.mean() / rank_ic.std())
    ori_label = pd.read_pickle('/your_own_path/data/%s.pkl'%day_loader.dset)['LABEL%s'%day_loader.label_id]
    RMSE = rmse(pred.score, pred.ori_label)
    MAE = mae(pred.score, pred.ori_label)
    pprint('RMSE:', RMSE)
    pprint('MAE:', MAE)

    return RMSE, MAE

@ex.config
def run_config():
    seed = 2
    output_path = '/your_own_path/out'
    loader_name = 'daily_loader_v3'
    model_name = 'rnn_v3'
    comt = 'rnn_60_1.0'
    run_on = False
    dsets = ["day_csi300"]


@ex.main
def main(_run, seed, loader_name, model_name, output_path, comt, run_on,
         dsets):
    # path
    output_path = create_output_path(output_path)

    pprint('output path:', output_path)
    model_path = output_path + '/model.bin'
    pprint('create loader `%s` and model `%s`...' % (loader_name, model_name))

    ###### Daily Model and Data Prepare #########
    set_random_seed(seed)
    global day_loader
    day_loader = getattr(loaders_module, loader_name).DataLoader(dset=dsets[0])
    day_model_1 = getattr(models_module, model_name).Day_Model_1()
    super_model = getattr(models_module, model_name)
    pprint(f'''
        Day_Model_1: {count_num_params(super_model.Day_Model_1())},
        Day_model_2: {count_num_params(super_model.Day_Model_2())}
        ''')
    pprint('load daily data...')
    day_train_set, day_valid_set, day_test_set = day_loader.load_data()

    ###### Day Model 1#######
    train_hids, valid_hids, test_hids = Day_model_1(output_path, day_model_1,
                                         day_train_set, day_valid_set,
                                         day_test_set)

    ####### Day Model 2########
    set_random_seed(seed)
    _run = SummaryWriter(comment='_day_model2') if run_on else None
    day_model_2 = getattr(models_module, model_name).Day_Model_2()
    pred_path = output_path+'/pred_%s.pkl' %(model_name)
    rmse_min,  mae_min = Day_model_2(
        _run,
        output_path,
        day_model_2,
        day_train_set,
        day_valid_set,
        day_test_set,
        train_hids,
        valid_hids,
        test_hids,
        pred_path)

    pprint('###################')
    pprint(f'Final RMSE: {rmse_min}, MAE: {mae_min}')

if __name__ == '__main__':

    ex.run_commandline()
