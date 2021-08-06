# -*- coding：utf-8 -*-
# Author: juzstu
# Time: 2021/8/7 18:19

from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import logging

logger = logging.getLogger()

formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler = logging.FileHandler('cbt_train.log', 'w')
fhandler.setLevel(logging.DEBUG)
fhandler.setFormatter(formatter)

chandler = logging.StreamHandler()
chandler.setLevel(logging.INFO)
chandler.setFormatter(formatter)

logger.addHandler(fhandler)
logger.addHandler(chandler)
logger.setLevel(logging.INFO)


def calc_uauc(df, label):
    df = df.groupby('userid')[[label, f'pred_{label}']].agg(list).reset_index()
    df['auc'] = df.apply(lambda x: roc_auc_score(x[label], x[f'pred_{label}']), axis=1)
    return df['auc'].mean()


def single_cbt_train(train_: pd.DataFrame, valid_: pd.DataFrame, use_train_feats: list, label: str, eta: float):
    print('data shape:\ntrain--{}\nvalid--{}\n'.format(train_.shape, valid_.shape))
    print('Use {} features ...'.format(len(use_train_feats)))
    print('Use catboost to train ...')

    params = {
        'iterations': 50000,
        'learning_rate': eta,
        'max_depth': 6,
        'random_seed': 1,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'early_stopping_rounds': 1000,
        'use_best_model': True,
        'verbose': 100
    }

    train_pool = Pool(train_[use_train_feats], train_[label], feature_names=use_train_feats)
    valid_pool = Pool(valid_[use_train_feats], valid_[label], feature_names=use_train_feats)

    clf = CatBoostClassifier(**params)
    clf.fit(train_pool, eval_set=valid_pool)

    imp_df = pd.DataFrame()
    imp_df['feats'] = clf.feature_names_
    imp_df['imp'] = clf.feature_importances_
    imp_df.sort_values('imp', ascending=False, inplace=True)
    print(imp_df.head(50))

    valid_['pred_{}'.format(label)] = clf.predict_proba(valid_[use_train_feats])[:, 1]
    tmp_auc = calc_uauc(valid_, label)
    print(f'UAUC of {label}: {tmp_auc}')
    return tmp_auc, clf


def online_cbt_train(train_: pd.DataFrame, test_: pd.DataFrame, use_train_feats: list, label: str, eta: float,
                     n_rounds: int):
    print('data shape:\ntrain--{}\ntest--{}\n'.format(train_.shape, test_.shape))
    print('Use {} features ...'.format(len(use_train_feats)))
    print('Use catboost to train ...')

    params = {
        'iterations': n_rounds,
        'learning_rate': eta,
        'max_depth': 6,
        'random_seed': 1,
        'task_type': 'GPU',
        'verbose': 100
    }

    train_pool = Pool(train_[use_train_feats], train_[label], feature_names=use_train_feats)

    clf = CatBoostClassifier(**params)
    clf.fit(train_pool)
    test_[label] = clf.predict_proba(test_[use_train_feats])[:, 1]
    test_[['userid', 'feedid', label]].to_csv(f'{sub_path}/r2_testa_cbt_{label}.csv', index=False, encoding='utf8')
    clf.save_model(f'{model_path}/cbt_{label}.cbm')


if __name__ == "__main__":
    sub_path = './sub'
    feat_path = './feats'
    model_path = './tree_model'

    label_cols = ['like', 'click_avatar', 'forward', 'follow', 'favorite', 'read_comment', 'comment']
    rounds_dict = {}
    auc_score = []
    for n in label_cols:
        logger.info(f'start train {n} ...')
        train_set = pd.read_feather(f'{feat_path}/for_train_{n}.feather')
        valid_set = pd.read_feather(f'{feat_path}/for_valid_{n}.feather')

        use_cols = [i for i in valid_set.columns if i not in ['feedid', 'userid', 'date_'] + label_cols]
        auc_, cbt_model = single_cbt_train(train_set, valid_set, use_cols, n, 0.05)
        auc_score.append(auc_)
        rounds_dict[n] = cbt_model.get_best_iteration()
        logger.info('#' * 100)

    for n in zip(label_cols, auc_score):
        logger.info(f'AUC of {n[0]}: {n[1]}, rounds-{rounds_dict[n[0]]}')

    logger.info(f'score: {np.average(auc_score, weights=[3, 2, 1, 1, 1, 4, 1])}')

    test_set = pd.read_feather(feat_path + '/round2_test_feats.feather')
    for n in label_cols:
        now_model_path = f'{model_path}/cbt_{n}.cbm'
        train_set = pd.read_feather(f'{feat_path}/all_train_{n}.feather')
        use_cols = [i for i in train_set.columns if i not in ['feedid', 'userid', 'date_'] + label_cols]
        if os.path.exists(now_model_path):
            logger.info(f'find exist cbt model of {n}.')
            model = CatBoostClassifier().load_model(now_model_path)
            del train_set
            test_set[n] = model.predict_proba(test_set[use_cols])[:, 1]
        else:
            logger.info(f'start train {n} ...')
            online_cbt_train(train_set, test_set, use_cols, n, 0.05, rounds_dict[n]+1000)
        logger.info('#' * 100)

    test_set[['userid', 'feedid'] + label_cols].to_csv(f'{sub_path}/r2_testa_cbt_baseline.csv', index=False, encoding='utf8')
