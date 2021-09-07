# -*- coding：utf-8 -*-
# Author: juzstu
# Time: 2021/8/7 18:19

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from collections import defaultdict
import warnings
import faiss
import pickle
from gensim.models import Word2Vec
import multiprocessing

warnings.filterwarnings('ignore')


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# 全局统计
def cnt_stat(df, group_cols, target_col, use_cnt=True, use_nunique=True):
    if isinstance(group_cols, list):
        col_name = '_'.join(group_cols)
    else:
        col_name = 'global_' + group_cols
    if use_cnt:
        df[f'{col_name}_cnt'] = df.groupby(group_cols)[target_col].transform('count')
    if use_nunique:
        df[f'{col_name}_dcnt'] = df.groupby(group_cols)[target_col].transform('nunique')
    return df


def count2vec(input_values, output_num, output_prefix, seed=1024):
    count_enc = CountVectorizer(lowercase=True, ngram_range=(1, 1))
    count_vec = count_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=10, random_state=seed).fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['svd_cntvec_{}_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def tf_idf(input_values, output_num, output_prefix, seed=1024):
    tfidf_enc = TfidfVectorizer(lowercase=True, ngram_range=(1, 1))
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=10, random_state=seed).fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['svd_tfidf_{}_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def gen_svd_df(df, group_target, num=5, seed=1024):
    tfidf_tmp = tf_idf(df[group_target], num, group_target, seed)
    count_tmp = count2vec(df[group_target], num, group_target, seed)
    df = pd.concat([df, tfidf_tmp, count_tmp], axis=1)
    del df[group_target]
    return df


def list_tm(df, df_, col, cnt_, read_, comment_, like_, click_, forward_, follow_, favorite_, use_hist=False):
    df_ = df_.reset_index(drop=True)
    if use_hist:
        cnt_dict = cnt_
        read_dict = read_
        comment_dict = comment_
        like_dict = like_
        click_dict = click_
        forward_dict = forward_
        follow_dict = follow_
        favorite_dict = favorite_
    else:
        cnt_dict = defaultdict(int)
        read_dict = defaultdict(int)
        comment_dict = defaultdict(int)
        like_dict = defaultdict(int)
        click_dict = defaultdict(int)
        forward_dict = defaultdict(int)
        follow_dict = defaultdict(int)
        favorite_dict = defaultdict(int)
    list_ids = [col, 'read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']
    for (i, read_comment, comment, like, click_avatar, forward, follow, favorite) in df[list_ids].values:
        if ';' in i:
            user, ids = i.split('|')
            for j in ids.split(';'):
                cnt_dict[f'{user}|{j}'] += 1
                read_dict[f'{user}|{j}'] += read_comment
                comment_dict[f'{user}|{j}'] += comment
                like_dict[f'{user}|{j}'] += like
                click_dict[f'{user}|{j}'] += click_avatar
                forward_dict[f'{user}|{j}'] += forward
                follow_dict[f'{user}|{j}'] += follow
                favorite_dict[f'{user}|{j}'] += favorite
    read_dict_ = {k: v / cnt_dict[k] for k, v in read_dict.items()}
    comment_dict_ = {k: v / cnt_dict[k] for k, v in comment_dict.items()}
    like_dict_ = {k: v / cnt_dict[k] for k, v in like_dict.items()}
    click_dict_ = {k: v / cnt_dict[k] for k, v in click_dict.items()}
    forward_dict_ = {k: v / cnt_dict[k] for k, v in forward_dict.items()}
    follow_dict_ = {k: v / cnt_dict[k] for k, v in follow_dict.items()}
    favorite_dict_ = {k: v / cnt_dict[k] for k, v in favorite_dict.items()}
    tmp_list = []
    for f, d, c in df_[['feedid', 'date_', col]].values:
        user, ids = c.split('|')
        if ';' in ids:
            for j in ids.split(';'):
                tmp_list.append([user, f, d, f'{user}|{j}'])
        else:
            tmp_list.append([user, f, d, np.nan])
    tmp_list = pd.DataFrame(tmp_list, columns=['userid', 'feedid', 'date_', col])
    for c in zip(list_ids[1:], [read_dict_, comment_dict_, like_dict_, click_dict_, forward_dict_, follow_dict_, favorite_dict_]):
        tmp_list[c[0]] = tmp_list[col].map(c[1])
    del tmp_list[col]
    stat_cols = ['min', 'max', 'mean', 'std', 'count']
    tmp_list = tmp_list.groupby(['userid', 'feedid', 'date_'])[list_ids[1:]].agg(stat_cols).reset_index()
    tmp_list.columns = ['userid', 'feedid', 'date_'] + [f'{col}_{i}_{j}' for i in list_ids[1:] for j in stat_cols]
    for i in list_ids[1:]:
        tmp_list[f'{col}_{i}_std'] = np.sqrt(
            tmp_list[f'{col}_{i}_std'] ** 2 * (tmp_list[f'{col}_{i}_count'] - 1) / tmp_list[f'{col}_{i}_count'])
        del tmp_list[f'{col}_{i}_count']

    return tmp_list, (cnt_dict, read_dict, comment_dict, like_dict, click_dict, forward_dict, follow_dict, favorite_dict)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# 多个tag, keyword的点击率统计特征
def gen_tag_mean(train_path, feed_info_path, sava_path):
    user_action = pd.read_feather(train_path)
    m_list = ['machine_keyword_list', 'manual_tag_list']

    feed_info = pd.read_csv(feed_info_path, usecols=['feedid'] + m_list)

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    for d in m_list:
        user_action[d] = user_action['userid'].astype(str) + '|' + user_action[d].astype(str)
    del feed_info

    for m in m_list:
        print(f'start deal {m} ...')
        cnt_u, read_u, comment_u, like_u, click_u, forward_u, follow_u, favorite_u = None, None, None, None, None, None, None, None
        for d in range(2, 15):
            print(f'date of {d}')
            prev_tmp = user_action[user_action['date_'] == d - 1]
            now_tmp = user_action[user_action['date_'] == d]
            flg = True if d > 2 else False
            now_tmp, all_dict = list_tm(prev_tmp, now_tmp, m, cnt_u, read_u, comment_u, like_u, click_u, forward_u, follow_u, favorite_u, use_hist=flg)
            if d == 14:
                save_obj(all_dict, f'{sava_path}/{m}_user_ctr_dict.pkl')
            cnt_u, read_u, comment_u, like_u, click_u, forward_u, follow_u, favorite_u = all_dict
            print(f'the shape of df_{m}_{d}:{now_tmp.shape[0]}')
            now_tmp['userid'] = now_tmp['userid'].astype(int)
            now_tmp.to_feather(f'{sava_path}/{m}_user_ctr_{d}.feather')


# 常规全局统计特征
def gen_base_feats(train_path, feed_info_path, train_save_path, base_hist_path):
    user_action = pd.read_feather(train_path)
    r1_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_a.csv', usecols=['userid', 'feedid'])
    r1_test_b = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_b.csv', usecols=['userid', 'feedid'])
    r2_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data2/test_a.csv', usecols=['userid', 'feedid'])

    test = pd.concat([r1_test_a, r1_test_b, r2_test_a], axis=0, ignore_index=True)
    del r1_test_a, r1_test_b, r2_test_a
    test['date_'] = 15
    for i in ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']:
        test[i] = -1
    user_action = user_action[test.columns].append(test)
    del test

    feed_info = pd.read_csv(feed_info_path, usecols=['feedid', 'authorid', 'videoplayseconds'])

    feed_info['vs_ratio'] = feed_info['videoplayseconds'] / feed_info.groupby('authorid')['videoplayseconds'].transform(
        'mean')

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    del feed_info

    user_action = cnt_stat(user_action, 'feedid', 'userid')
    user_action = cnt_stat(user_action, ['feedid', 'date_'], 'userid', use_nunique=False)
    user_action = cnt_stat(user_action, 'userid', 'feedid')
    user_action = cnt_stat(user_action, ['userid', 'date_'], 'feedid', use_nunique=False)

    user_action = cnt_stat(user_action, 'userid', 'authorid', use_cnt=False)
    user_action = cnt_stat(user_action, 'authorid', 'userid')
    user_action = cnt_stat(user_action, ['authorid', 'date_'], 'userid', use_nunique=False)

    user_action = cnt_stat(user_action, ['userid', 'authorid'], 'feedid', use_nunique=False)
    user_action = cnt_stat(user_action, ['authorid', 'feedid'], 'userid', use_nunique=False)
    user_action = cnt_stat(user_action, ['userid', 'authorid', 'date_'], 'feedid', use_nunique=False)
    user_action = cnt_stat(user_action, ['authorid', 'feedid', 'date_'], 'userid', use_nunique=False)

    user_action['userid_authorid_ratio'] = user_action['userid_authorid_cnt'] / user_action['global_userid_cnt']
    user_action['feedid_authorid_ratio'] = user_action['authorid_feedid_cnt'] / user_action['global_authorid_cnt']

    user_action['userid_date_ratio'] = user_action['userid_date__cnt'] / user_action['global_userid_cnt']
    user_action['feedid_date_ratio'] = user_action['feedid_date__cnt'] / user_action['global_feedid_cnt']
    user_action['authorid_date_ratio'] = user_action['authorid_date__cnt'] / user_action['global_authorid_cnt']

    for t in ['feedid', 'authorid']:
        user_action[f'date_first_{t}'] = user_action.groupby(t)['date_'].transform('min')
        user_action[f'diff_first_{t}'] = user_action['date_'] - user_action[f'date_first_{t}']

    train_ = user_action[(user_action['date_'] < 15) & (user_action['date_'] > 1)]
    train_ = train_.drop(['authorid', 'date_first_feedid', 'date_first_authorid'], axis=1)
    train_ = reduce_mem_usage(train_)
    train_ = train_.reset_index(drop=True)
    train_.to_feather(train_save_path)

    feed_list = (['feedid'], ['feedid', 'global_feedid_cnt', 'global_feedid_dcnt', 'date_first_feedid'])
    user_list = (['userid'], ['userid', 'global_userid_cnt', 'global_userid_dcnt'])
    author_list = (['authorid'], ['authorid', 'global_authorid_cnt', 'global_authorid_dcnt', 'date_first_authorid'])
    user_author_list = (['userid', 'authorid'], ['userid', 'authorid', 'userid_authorid_cnt'])
    author_feed_list = (['authorid', 'feedid'], ['authorid', 'feedid', 'authorid_feedid_cnt'])

    for i in [feed_list, user_list, author_list, user_author_list, author_feed_list]:
        file_name = '_'.join(i[0])
        tmp_df = user_action[i[1]].drop_duplicates(subset=i[0], keep='first').reset_index(drop=True)
        print(f'for {file_name} now rows:', tmp_df.shape[0])
        tmp_df = reduce_mem_usage(tmp_df)
        tmp_df.to_feather(base_hist_path + f'_{file_name}.feather')


def gen_tm_ratio(train_path, feed_info_path, train_save_path, hist_path):
    user_action = pd.read_feather(train_path)
    del user_action['device'], user_action['stay'], user_action['play']
    feed_info = pd.read_csv(feed_info_path, usecols=['feedid', 'authorid'])

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    del feed_info

    user_action['user_authorid'] = user_action['userid'].astype(str) + '_' + user_action['authorid'].astype(str)

    print(f'train shape:{user_action.shape[0]}')

    tm_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']
    tm_ids = ['userid', 'feedid', 'authorid', 'user_authorid']

    for d in tqdm(tm_ids):
        grp_tmp = user_action.groupby(d)[tm_cols].agg('sum').reset_index()
        now_cols = [f'{d}_{i}_ratio' for i in tm_cols]
        grp_tmp.columns = [d] + now_cols
        grp_tmp['tmp_sum'] = grp_tmp[now_cols].sum(axis=1)
        for i in now_cols:
            grp_tmp[i] = grp_tmp[i] / grp_tmp['tmp_sum']
        del grp_tmp['tmp_sum']
        if 'user' in d:
            for t in tm_cols:
                tmp_df = user_action[user_action[t] == 1]
                tmp_df = tmp_df.groupby(d)['date_'].agg([[f'hist_{d}_{t}_date_diff', 'max']])
                tmp_df[f'hist_{d}_{t}_date_diff'] = 15 - tmp_df[f'hist_{d}_{t}_date_diff']
                grp_tmp = grp_tmp.merge(tmp_df, on=d, how='left')
        grp_tmp = reduce_mem_usage(grp_tmp)
        grp_tmp.to_feather(f'{hist_path}_{d}.feather')

    for t in tqdm(range(2, 15)):
        prev_tmp = user_action[user_action['date_'] < t]
        now_tmp = user_action[user_action['date_'] == t]
        for d in tm_ids:
            grp_tmp = prev_tmp.groupby(d)[tm_cols].agg(['sum']).reset_index()
            now_cols = [f'{d}_{i}_ratio' for i in tm_cols]
            grp_tmp.columns = [d] + now_cols
            grp_tmp['tmp_sum'] = grp_tmp[now_cols].sum(axis=1)
            for i in now_cols:
                grp_tmp[i] = grp_tmp[i] / grp_tmp['tmp_sum']
            del grp_tmp['tmp_sum']
            if 'user' in d:
                for p in tm_cols:
                    tmp_df = prev_tmp[user_action[p] == 1]
                    tmp_df = tmp_df.groupby(d)['date_'].agg([[f'hist_{d}_{p}_date_diff', 'max']])
                    grp_tmp = grp_tmp.merge(tmp_df, on=d, how='left')
                    grp_tmp[f'hist_{d}_{p}_date_diff'] = t - grp_tmp[f'hist_{d}_{p}_date_diff']

            now_tmp = now_tmp.merge(grp_tmp, on=d, how='left')
            if d not in ['userid', 'feedid']:
                del now_tmp[d]
        for c in tm_cols:
            del now_tmp[c]
        now_tmp.to_feather(f'{train_save_path}_{t}.feather')


# 各个ID类特征的点击率
def gen_tm_feats(train_path, feed_info_path, km_path, train_save_path, hist_path):
    user_action = pd.read_feather(train_path)
    feed_info = pd.read_csv(feed_info_path, usecols=['feedid', 'authorid', 'videoplayseconds', 'manual_tag_list', 'machine_tag_list'])
    km = pd.read_csv(km_path)
    feed_info = feed_info.merge(km, on='feedid', how='left')

    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
        lambda x: x.split(';') if isinstance(x, str) else x)
    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
        lambda x: ';'.join([i.split(' ')[0] for i in x]) if isinstance(x, list) else np.nan)
    for m in tqdm(['manual_tag_list', 'machine_tag_list']):
        feed_info[m] = feed_info[m].astype(str).apply(lambda x: ';'.join(sorted(x.split(';'))))

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    user_action['play'] = user_action['play'] / 1000 / user_action['videoplayseconds']
    user_action['stay'] = user_action['stay'] / 1000 / user_action['videoplayseconds']
    del feed_info

    del user_action['device'], user_action['videoplayseconds']
    tm_cols = ['authorid', 'manual_tag_list', 'machine_tag_list', 'km_label']
    connect_cols = ['authorid', 'manual_tag_list', 'machine_tag_list', 'km_label']
    for n in tqdm(tm_cols):
        user_action[f'user_{n}'] = user_action['userid'].astype(str) + '_' + user_action[n].astype(str)
        connect_cols.append(f'user_{n}')

    print(f'train shape:{user_action.shape[0]}')

    tm_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite', 'play', 'stay']
    tm_ids = ['userid', 'feedid'] + connect_cols

    for d in tqdm(tm_ids):
        grp_tmp = user_action.groupby(d)[tm_cols].agg('mean').reset_index()
        grp_tmp.columns = [d] + [f'{d}_{i}_tm' for i in tm_cols]
        grp_tmp = reduce_mem_usage(grp_tmp)
        grp_tmp.to_feather(f'{hist_path}_{d}.feather')

    for t in tqdm(range(2, 15)):
        prev_tmp = user_action[user_action['date_'] < t]
        now_tmp = user_action[user_action['date_'] == t]
        for d in tm_ids:
            grp_tmp = prev_tmp.groupby(d)[tm_cols].agg(['mean']).reset_index()
            grp_tmp.columns = [d] + [f'{d}_{i}_tm' for i in tm_cols]
            now_tmp = now_tmp.merge(grp_tmp, on=d, how='left')
            if d not in ['userid', 'feedid']:
                del now_tmp[d]
        for c in tm_cols:
            del now_tmp[c]
        now_tmp.to_feather(f'{train_save_path}_{t}.feather')


def gen_svd_feats(train_path, feed_info_path, train_save_path, hist_path):
    user_action = pd.read_feather(train_path)
    r1_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_a.csv', usecols=['userid', 'feedid'])
    r1_test_b = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_b.csv', usecols=['userid', 'feedid'])
    r2_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data2/test_a.csv', usecols=['userid', 'feedid'])

    test = pd.concat([r1_test_a, r1_test_b, r2_test_a], axis=0, ignore_index=True)
    del r1_test_a, r1_test_b, r2_test_a
    test['date_'] = 15
    user_action = user_action[test.columns].append(test)
    del test

    m_list = ['manual_keyword_list', 'machine_keyword_list', 'manual_tag_list', 'machine_tag_list']
    n_list = ['description']
    # , 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char']
    feed_info = pd.read_csv(feed_info_path, usecols=['feedid'] + m_list + n_list)
    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
        lambda x: x.split(';') if isinstance(x, str) else x)
    feed_info['machine_tag_list'] = feed_info['machine_tag_list'].apply(
        lambda x: ';'.join([i.split(' ')[0] for i in x]) if isinstance(x, list) else np.nan)

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    del feed_info

    for m in tqdm(m_list + n_list):
        user_action[m] = user_action[m].astype(str)
        if m in m_list:
            user_action[m] = user_action[m].apply(lambda x: x.replace(';', ' '))
        user_action = gen_svd_df(user_action, m)

    train_ = user_action[(user_action['date_'] < 15) & (user_action['date_'] > 1)]
    train_ = reduce_mem_usage(train_)
    train_ = train_.reset_index(drop=True)
    train_.to_feather(train_save_path)

    del user_action['userid'], user_action['date_']

    user_action = user_action.drop_duplicates(subset='feedid', keep='first')
    user_action = reduce_mem_usage(user_action)
    user_action = user_action.reset_index(drop=True)
    user_action.to_feather(hist_path)


# 原始512emb svd降维
def svd_feed(feed_emb_path, save_path, output_num=10, seed=1024):
    feed_emb = pd.read_csv(feed_emb_path)
    feed_emb['feed_embedding'] = feed_emb['feed_embedding'].apply(
        lambda x: np.array(x.strip().split(' '), dtype='float32'))
    feed_embedding = np.array([i for i in feed_emb['feed_embedding'].values])
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=10, random_state=seed).fit_transform(feed_embedding)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['feed_emb_{}'.format(i) for i in range(output_num)]
    svd_tmp = pd.concat([feed_emb[['feedid']], svd_tmp], axis=1)
    svd_tmp.to_feather(save_path)


# 原始512emb聚类获取类别特征，便于后续的target encoding
def get_kmeans_label(feed_emb_path, save_path, num):
    feed_emb = pd.read_csv(feed_emb_path)
    feed_emb['feed_embedding'] = feed_emb['feed_embedding'].apply(
        lambda x: np.array(x.strip().split(' '), dtype='float32'))
    feed_embedding = np.array([i for i in feed_emb['feed_embedding'].values])
    del feed_emb['feed_embedding']
    km = faiss.Kmeans(512, num, niter=500, verbose=True, gpu=False)
    km.train(feed_embedding)
    print('train finished.')
    _, labels = km.index.search(feed_embedding, 1)
    feed_emb['km_label'] = labels
    feed_emb.to_csv(save_path, index=False)


def w2v_feat(df, feat, length, w2v_model_path):
    if os.path.exists(w2v_model_path):
        print('find existing w2v model: {}'.format(w2v_model_path))
        model = Word2Vec.load(w2v_model_path)
    else:
        print('start word2vec, use cpu count: {} ...'.format(multiprocessing.cpu_count()))
        model = Word2Vec(df[feat].values, vector_size=length, window=5, min_count=1, sg=1, hs=0, negative=5,
                         workers=multiprocessing.cpu_count() * 2, epochs=10, seed=1)
        model.save(w2v_model_path)
    return model


# 用户浏览feed聚合后进行word2vec，获取feed与用户的交互emd特征
def get_user_feed_w2v(train_path, w2v_path, save_path, num):
    user_action = pd.read_feather(train_path)
    r1_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_a.csv', usecols=['userid', 'feedid'])
    r1_test_b = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_b.csv', usecols=['userid', 'feedid'])
    r2_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data2/test_a.csv', usecols=['userid', 'feedid'])

    test = pd.concat([r1_test_a, r1_test_b, r2_test_a], axis=0, ignore_index=True)
    del r1_test_a, r1_test_b, r2_test_a
    test['date_'] = 15
    user_action = user_action[test.columns].append(test)
    del test

    user_action['feedid'] = user_action['feedid'].astype(str)

    feed_list = user_action['feedid'].unique()

    user_action = user_action.groupby('userid')['feedid'].agg(list).reset_index()

    model = w2v_feat(user_action, 'feedid', num, w2v_path)

    tmp_list = []
    for x in tqdm(feed_list):
        tmp_list.append(model.wv[x])
    tmp_list = pd.DataFrame(tmp_list, columns=[f'fu_w2v_{w}' for w in range(num)])
    tmp_list['feedid'] = feed_list
    tmp_list['feedid'] = tmp_list['feedid'].astype(int)
    tmp_list = reduce_mem_usage(tmp_list)
    tmp_list.to_feather(save_path)


def get_user_feed_svd(train_path, save_path):
    user_action = pd.read_feather(train_path)
    r1_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_a.csv', usecols=['userid', 'feedid'])
    r1_test_b = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/test_b.csv', usecols=['userid', 'feedid'])
    r2_test_a = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data2/test_a.csv', usecols=['userid', 'feedid'])

    test = pd.concat([r1_test_a, r1_test_b, r2_test_a], axis=0, ignore_index=True)
    del r1_test_a, r1_test_b, r2_test_a
    test['date_'] = 15
    user_action = user_action[test.columns].append(test)
    del test

    user_action['userid'] = user_action['userid'].astype(str)

    user_action = user_action.groupby('feedid')['userid'].agg(lambda x: ' '.join(list(x))).reset_index()
    user_action.columns = ['feedid', 'ulist']

    user_action = gen_svd_df(user_action, 'ulist')

    user_action = reduce_mem_usage(user_action)
    user_action.to_feather(save_path)


# 当前feed与历史feed emb均值做差，获取emb的差异性特征
def gen_hist_emb_diff(train_path, emb_path, train_save_path, hist_path):
    user_action = pd.read_feather(train_path)
    del user_action['device'], user_action['play'], user_action['stay']

    feed_emb = pd.read_feather(emb_path)
    emb_cols = [e for e in feed_emb.columns if e != 'feedid']
    hist_cols = [f'hist_mean_{c}' for c in emb_cols]

    user_action = user_action.merge(feed_emb, on='feedid', how='left')
    del feed_emb

    tm_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']

    for d in tqdm(tm_cols):
        grp_tmp = user_action[user_action[d] == 1].groupby('userid')[emb_cols].agg('mean').reset_index()
        grp_tmp.columns = ['userid'] + [f'hist_mean_{d}_{i}' for i in emb_cols]
        grp_tmp = reduce_mem_usage(grp_tmp)
        grp_tmp.to_feather(f'{hist_path}_{d}_emb_mean.feather')

    train_df = []
    for t in tqdm(range(2, 15)):
        prev_tmp = user_action[user_action['date_'] < t]
        now_tmp = user_action[user_action['date_'] == t]
        for d in tm_cols:
            grp_tmp = prev_tmp[prev_tmp[d] == 1].groupby('userid')[emb_cols].agg('mean').reset_index()
            grp_tmp.columns = ['userid'] + hist_cols
            now_tmp = now_tmp.merge(grp_tmp, on='userid', how='left')
            for i in emb_cols:
                now_tmp[f'{d}_{i}'] = now_tmp[i] - now_tmp[f'hist_mean_{i}']
                del now_tmp[f'hist_mean_{i}']
            del now_tmp[d]
        now_tmp = now_tmp.drop(emb_cols, axis=1)
        train_df.append(now_tmp)

    train_ = pd.concat(train_df, axis=0, ignore_index=True)
    train_['feedid'] = train_['feedid'].astype(int)
    print(f'train shape:{train_.shape[0]}')
    train_ = reduce_mem_usage(train_)
    train_.to_feather(train_save_path)


def gen_hist_w2v_mean(train_path, emb_path, save_train_path):
    user_action = pd.read_feather(train_path)
    user_action = user_action[['userid', 'feedid', 'date_', 'like']]

    feed_info = pd.read_feather(emb_path)
    emb_cols = [e for e in feed_info.columns if e != 'feedid']

    user_action = user_action.merge(feed_info, on='feedid', how='left')
    del feed_info

    tm_cols = ['like']

    train_df = []
    for t in tqdm(range(2, 15)):
        prev_tmp = user_action[user_action['date_'] < t]
        now_tmp = user_action[user_action['date_'] == t][['userid', 'feedid', 'date_']]
        for d in tm_cols:
            grp_tmp = prev_tmp[prev_tmp[d] == 1].groupby('userid')[emb_cols].agg('mean').reset_index()
            grp_tmp.columns = ['userid'] + [f'hist_mean_{d}_{i}' for i in emb_cols]
            now_tmp = now_tmp.merge(grp_tmp, on='userid', how='left')
        train_df.append(now_tmp)

    train_ = pd.concat(train_df, axis=0, ignore_index=True)
    print(f'train shape:{train_.shape}')
    train_ = reduce_mem_usage(train_)
    train_.to_feather(save_train_path)


def gen_feats(base_path, ori_train_path, save_path):
    ori_feed_path = f'{base_path}/feed_info.csv'
    ori_emb_path = f'{base_path}/feed_embeddings.csv'
    b_path = f'{save_path}/base'
    s_path = f'{save_path}/svd'
    t_path = f'{save_path}/base_tm'
    dec_path = f'{save_path}/feed_emb_dec.feather'
    km_path = f'{save_path}/feed_km.csv'
    fu_w2v_path = f'{save_path}/user_feed.w2v'
    feed_w2v_path = f'{save_path}/fu_w2v.feather'
    w_path1 = f'{save_path}/diff_emb'
    w_path2 = f'{save_path}/diff_fu'
    l_path = f'{save_path}/hist_mean_w2v'

    gen_base_feats(ori_train_path, ori_feed_path, f'{b_path}_train.feather',  f'{b_path}_hist')
    print('the generation of base feats finished.')
    gen_tag_mean(ori_train_path, ori_feed_path, save_path)
    print('the generation of tag ctr feats finished.')
    gen_svd_feats(ori_train_path, ori_feed_path, f'{s_path}_train.feather',  f'{s_path}_hist.feather')
    print('the generation of svd feats finished.')
    get_kmeans_label(ori_emb_path, km_path, 1000)
    print('the generation of emb kmeans label finished.')
    svd_feed(ori_emb_path, dec_path)
    print('origin emb svd finished.')
    get_user_feed_w2v(ori_train_path, fu_w2v_path, feed_w2v_path, 10)
    print('the generation of w2v feats finished.')
    gen_tm_feats(ori_train_path, ori_feed_path, km_path, f'{t_path}_train', f'{t_path}_hist')
    print('the generation of target mean feats finished.')

    gen_hist_emb_diff(ori_train_path, dec_path, f'{w_path1}_train.feather', f'{w_path1}_hist')
    gen_hist_emb_diff(ori_train_path, feed_w2v_path, f'{w_path2}_train.feather', f'{w_path2}_hist')
    print('the generation of hist emb diff feats finished.')
    get_user_feed_svd(ori_train_path,  f'{save_path}/svd_under_feed.feather')
    print('the generation of hist user feed svd finished.')
    gen_hist_w2v_mean(ori_train_path, feed_w2v_path, f'{l_path}_train.feather')
    print('the generation of hist emb mean feats finished.')


def get_ori_train():
    train2 = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data2/user_action.csv')
    train1 = pd.read_csv('./wbdc2021/data/wedata/wechat_algo_data1/user_action.csv')
    train2 = train2.append(train1)
    del train1
    train2 = train2.reset_index(drop=True)
    train2 = reduce_mem_usage(train2)
    print(train2.shape)
    train2.to_feather('./feats/user_action.feather')
    train2 = train2.sort_values('date_')
    train2 = train2.drop_duplicates(subset=['userid', 'feedid'], keep='first').reset_index(drop=True)
    print(train2.shape)
    train2.to_feather('./feats/user_action_drop_repeat.feather')


def merge_file(path):
    tmp_list = []
    for i in tqdm(range(2, 15)):
        tmp_df = pd.read_feather(f'{path}_{i}.feather')
        tmp_list.append(tmp_df)
    tmp_list = pd.concat(tmp_list, axis=0, ignore_index=True)
    print(f'train shape:{tmp_list.shape[0]}')
    tmp_list = reduce_mem_usage(tmp_list)
    tmp_list.to_feather(f'{path}_all.feather')


def tag_ctr_filter(path, tag):
    cnt_dict, read_dict, comment_dict, like_dict, click_dict, forward_dict, follow_dict, favorite_dict = load_obj(f'{path}/{tag}_user_ctr_dict.pkl')
    tmp_list = []
    for d in [read_dict, comment_dict, like_dict, click_dict, forward_dict, follow_dict, favorite_dict]:
        d = {k: v / cnt_dict[k] for k, v in d.items() if v > 0}
        tmp_list.append(d)
    save_obj(tmp_list, f'{path}/{tag}_user_ctr_7dict.pkl')


def load_train_file(base_path):
    for_train = pd.read_feather(f'{base_path}/base_train.feather')
    all_kw_tm = pd.read_feather(f'{base_path}/machine_keyword_list_user_ctr_all.feather')
    all_tag_tm = pd.read_feather(f'{base_path}/manual_tag_list_user_ctr_all.feather')
    train_svd = pd.read_feather(f'{base_path}/svd_train.feather')
    train_base_tm = pd.read_feather(f'{base_path}/base_tm_train_all.feather')
    train_emb = pd.read_feather(f'{base_path}/diff_emb_train.feather')
    train_w2v = pd.read_feather(f'{base_path}/diff_fu_train.feather')
    fu_w2v = pd.read_feather(f'{base_path}/fu_w2v.feather')
    train_wm = pd.read_feather(f'{base_path}/hist_mean_w2v_train.feather')
    feed_svd = pd.read_feather(f'{base_path}/svd_under_feed.feather')

    for_train = for_train.merge(train_base_tm, on=['feedid', 'userid', 'date_'], how='left')
    del train_base_tm

    for_train = for_train.merge(train_svd, on=['feedid', 'userid', 'date_'], how='left')
    del train_svd

    for_train = for_train.merge(all_tag_tm, on=['feedid', 'userid', 'date_'], how='left')
    del all_tag_tm

    for_train = for_train.merge(all_kw_tm, on=['feedid', 'userid', 'date_'], how='left')
    del all_kw_tm

    for_train = for_train.merge(train_emb, on=['feedid', 'userid', 'date_'], how='left')
    del train_emb

    for_train = for_train.merge(fu_w2v, on='feedid', how='left')
    del fu_w2v

    for_train = for_train.merge(train_w2v, on=['feedid', 'userid', 'date_'], how='left')
    del train_w2v

    for_train = for_train.merge(feed_svd, on='feedid', how='left')
    del feed_svd

    for_train = for_train.merge(train_wm, on=['feedid', 'userid', 'date_'], how='left')
    del train_wm

    for_train.to_feather(f'{base_path}/round2_train_feats.feather')


def load_test_file(base_path, test_path):
    feed_info = pd.read_feather(f'{base_path}/feed_info.feather')

    for_test = pd.read_csv(test_path)
    for_test = for_test.merge(feed_info, on='feedid', how='left')

    base_list = [['feedid'], ['userid'], ['authorid'], ['userid', 'authorid'], ['authorid', 'feedid']]
    for b in base_list:
        col_name = '_'.join(b) + '_date__cnt'
        for_test[col_name] = for_test.groupby(b)['device'].transform('count')
    del for_test['device']
    feed_list = ['feedid']
    user_list = ['userid']
    author_list = ['authorid']
    user_author_list = ['userid', 'authorid']
    author_feed_list = ['authorid', 'feedid']

    for i in [feed_list, user_list, author_list, user_author_list, author_feed_list]:
        file_name = '_'.join(i)
        tmp_df = pd.read_feather(f'{base_path}/base_hist_{file_name}.feather')
        for_test = for_test.merge(tmp_df, on=i, how='left')

    # 冷启动填充
    for n in ['feedid', 'authorid']:
        for_test.loc[for_test[f'global_{n}_cnt'].isnull(), f'global_{n}_cnt'] = for_test.loc[
            for_test[f'global_{n}_cnt'].isnull(), f'{n}_date__cnt']

        for_test.loc[for_test[f'global_{n}_dcnt'].isnull(), f'global_{n}_dcnt'] = for_test.loc[
            for_test[f'global_{n}_dcnt'].isnull(), f'{n}_date__cnt']

    for n in ['userid_authorid', 'authorid_feedid']:
        for_test.loc[for_test[f'{n}_cnt'].isnull(), f'{n}_cnt'] = for_test.loc[
            for_test[f'{n}_cnt'].isnull(), f'{n}_date__cnt']

    for_test['userid_authorid_ratio'] = for_test['userid_authorid_cnt'] / for_test['global_userid_cnt']
    for_test['feedid_authorid_ratio'] = for_test['authorid_feedid_cnt'] / for_test['global_authorid_cnt']

    for_test['userid_date_ratio'] = for_test['userid_date__cnt'] / for_test['global_userid_cnt']
    for_test['feedid_date_ratio'] = for_test['feedid_date__cnt'] / for_test['global_feedid_cnt']
    for_test['authorid_date_ratio'] = for_test['authorid_date__cnt'] / for_test['global_authorid_cnt']

    for t in ['feedid', 'authorid']:
        for_test[f'diff_first_{t}'] = 15 - for_test[f'date_first_{t}']
        del for_test[f'date_first_{t}']
        for_test[f'diff_first_{t}'] = for_test[f'diff_first_{t}'].fillna(0)

    m_list = ['machine_keyword_list', 'manual_tag_list']
    label_list = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']
    for m in m_list:
        print(m)
        tmp_list = []
        list_ids = [f'tmp_{m}'] + label_list
        for_test[f'tmp_{m}'] = for_test['userid'].astype(str) + '|' + for_test[m].astype(str)
        read_, comment_, like_, click_, forward_, follow_, favorite_ = load_obj(f'{base_path}/{m}_user_ctr_7dict.pkl')
        for f, c in tqdm(for_test[['feedid', f'tmp_{m}']].values):
            user, ids = c.split('|')
            if ';' in ids:
                for j in ids.split(';'):
                    tmp_list.append([user, f, f'{user}|{j}'])
            else:
                tmp_list.append([user, f, np.nan])
        tmp_list = pd.DataFrame(tmp_list, columns=['userid', 'feedid', f'tmp_{m}'])
        for c in zip(list_ids[1:], [read_, comment_, like_, click_, forward_, follow_, favorite_]):
            tmp_list[c[0]] = tmp_list[f'tmp_{m}'].map(c[1]).fillna(0)
        del tmp_list[f'tmp_{m}'], for_test[f'tmp_{m}']
        stat_cols = ['min', 'max', 'mean', 'std', 'count']
        tmp_list = tmp_list.groupby(['userid', 'feedid'])[list_ids[1:]].agg(stat_cols).reset_index()
        tmp_list.columns = ['userid', 'feedid'] + [f'{m}_{i}_{j}' for i in list_ids[1:] for j in stat_cols]
        for i in list_ids[1:]:
            tmp_list[f'{m}_{i}_std'] = np.sqrt(
                tmp_list[f'{m}_{i}_std'] ** 2 * (tmp_list[f'{m}_{i}_count'] - 1) / tmp_list[f'{m}_{i}_count'])
            del tmp_list[f'{m}_{i}_count']

        tmp_list['userid'] = tmp_list['userid'].astype(int)
        for_test = for_test.merge(tmp_list, on=['userid', 'feedid'], how='left')

    del for_test['machine_keyword_list']
    test_svd = pd.read_feather(f'{base_path}/svd_hist.feather')
    for_test = for_test.merge(test_svd, on='feedid', how='left')

    svd_dict = {'svd_tfidf_manual_keyword_list_0': 0.292,
                'svd_tfidf_manual_keyword_list_1': 0.04446,
                'svd_tfidf_manual_keyword_list_2': 0.0377,
                'svd_tfidf_manual_keyword_list_3': 0.009636,
                'svd_tfidf_manual_keyword_list_4': 0.01441,
                'svd_cntvec_manual_keyword_list_0': 0.292,
                'svd_cntvec_manual_keyword_list_1': 0.11945,
                'svd_cntvec_manual_keyword_list_2': 0.08264,
                'svd_cntvec_manual_keyword_list_3': 0.002274,
                'svd_cntvec_manual_keyword_list_4': 0.04883,
                'svd_tfidf_machine_keyword_list_0': 0.6016,
                'svd_tfidf_machine_keyword_list_1': 0.0257,
                'svd_tfidf_machine_keyword_list_2': 0.0202,
                'svd_tfidf_machine_keyword_list_3': 0.00846,
                'svd_tfidf_machine_keyword_list_4': 0.01064,
                'svd_cntvec_machine_keyword_list_0': 0.04886,
                'svd_cntvec_machine_keyword_list_1': 0.6016,
                'svd_cntvec_machine_keyword_list_2': 0.0418,
                'svd_cntvec_machine_keyword_list_3': 0.0287,
                'svd_cntvec_machine_keyword_list_4': 0.03296,
                'svd_tfidf_manual_tag_list_0': 0.1848,
                'svd_tfidf_manual_tag_list_1': 0.0705,
                'svd_tfidf_manual_tag_list_2': 0.04874,
                'svd_tfidf_manual_tag_list_3': 0.06216,
                'svd_tfidf_manual_tag_list_4': -0.01532,
                'svd_cntvec_manual_tag_list_0': 0.7734,
                'svd_cntvec_manual_tag_list_1': 0.04358,
                'svd_cntvec_manual_tag_list_2': -0.02547,
                'svd_cntvec_manual_tag_list_3': -0.03415,
                'svd_cntvec_manual_tag_list_4': 0.0614,
                'svd_tfidf_machine_tag_list_0': 0.2861,
                'svd_tfidf_machine_tag_list_1': -0.0294,
                'svd_tfidf_machine_tag_list_2': 0.0158,
                'svd_tfidf_machine_tag_list_3': 0.04144,
                'svd_tfidf_machine_tag_list_4': -0.0001535,
                'svd_cntvec_machine_tag_list_0': 1.076,
                'svd_cntvec_machine_tag_list_1': 0.04974,
                'svd_cntvec_machine_tag_list_2': -0.0243,
                'svd_cntvec_machine_tag_list_3': 0.009315,
                'svd_cntvec_machine_tag_list_4': 0.02483,
                'svd_tfidf_description_0': 0.1698,
                'svd_tfidf_description_1': 0.02164,
                'svd_tfidf_description_2': -0.02055,
                'svd_tfidf_description_3': -0.0198,
                'svd_tfidf_description_4': 0.01101,
                'svd_cntvec_description_0': 3.06,
                'svd_cntvec_description_1': -1.725,
                'svd_cntvec_description_2': 0.444,
                'svd_cntvec_description_3': 0.01752,
                'svd_cntvec_description_4': -0.2338}

    for k, v in svd_dict.items():
        for_test[k] = for_test[k].fillna(v)

    del test_svd

    tm_cols = ['authorid', 'manual_tag_list', 'machine_tag_list', 'km_label']
    connect_cols = ['authorid', 'manual_tag_list', 'machine_tag_list', 'km_label']
    for n in tqdm(tm_cols):
        for_test[f'user_{n}'] = for_test['userid'].astype(str) + '_' + for_test[n].astype(str)
        connect_cols.append(f'user_{n}')

    tm_ids = ['userid', 'feedid'] + connect_cols

    for d in tqdm(tm_ids):
        grp_tmp = pd.read_feather(f'{base_path}/base_tm_hist_{d}.feather')
        for_test = for_test.merge(grp_tmp, on=d, how='left')
        if d not in ['userid', 'feedid', 'user_authorid', 'authorid']:
            del for_test[d]

    fu_w2v = pd.read_feather(f'{base_path}/fu_w2v.feather')
    feed_emb = pd.read_feather(f'{base_path}/feed_emb_dec.feather')

    fu_dict = {'fu_w2v_0': -0.4192, 'fu_w2v_1': -0.7144, 'fu_w2v_2': -1.131, 'fu_w2v_3': -0.279, 'fu_w2v_4': 1.125,
               'fu_w2v_5': 0.841, 'fu_w2v_6': -0.3801, 'fu_w2v_7': 0.1775, 'fu_w2v_8': -1.9795, 'fu_w2v_9': -1.745}

    for mode, emb in zip(['emb', 'fu'], [feed_emb, fu_w2v]):
        emb_cols = [e for e in emb.columns if e != 'feedid']
        for_test = for_test.merge(emb, on='feedid', how='left')
        if mode == 'fu':
            for k, v in fu_dict.items():
                for_test[k] = for_test[k].fillna(v)

        for d in label_list:
            grp_tmp = pd.read_feather(f'{base_path}/diff_{mode}_hist_{d}_emb_mean.feather')
            for_test = for_test.merge(grp_tmp, on='userid', how='left')
            for i in emb_cols:
                for_test[f'{d}_{i}'] = for_test[i] - for_test[f'hist_mean_{d}_{i}']
                if d != 'like' or mode == 'emb':
                    del for_test[f'hist_mean_{d}_{i}']
        if mode == 'emb':
            for_test = for_test.drop(emb_cols, axis=1)

    feed_svd = pd.read_feather(f'{base_path}/svd_under_feed.feather')
    for_test = for_test.merge(feed_svd, on='feedid', how='left')
    ulist_dict = {'svd_tfidf_ulist_0': 0.03958, 'svd_tfidf_ulist_1': -0.0002842, 'svd_tfidf_ulist_2': 0.001081,
                  'svd_tfidf_ulist_3': 0.002897, 'svd_tfidf_ulist_4': 0.01572, 'svd_cntvec_ulist_0': 0.972,
                  'svd_cntvec_ulist_1': -0.02539, 'svd_cntvec_ulist_2': -0.1755, 'svd_cntvec_ulist_3': -0.4963,
                  'svd_cntvec_ulist_4': 0.107}

    for k, v in ulist_dict.items():
        for_test[k] = for_test[k].fillna(v)

    del feed_svd
    for_test.to_feather(f'{base_path}/cbt_round2_test_feats.feather')


def split_train_valid():
    train_set = pd.read_feather(feat_path + '/round2_train_feats.feather')
    print('origin shape', train_set.shape)

    label_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']
    use_cols = [i for i in train_set.columns if i not in ['feedid', 'userid', 'date_'] + label_cols]
    valid_set = train_set[train_set['date_'] == 14]
    train_set = train_set[train_set['date_'] < 14]

    emb_comment_cols = ['comment_feed_emb_0', 'comment_feed_emb_1', 'comment_feed_emb_2', 'comment_feed_emb_3',
                        'comment_feed_emb_4', 'comment_feed_emb_5', 'comment_feed_emb_6', 'comment_feed_emb_7',
                        'comment_feed_emb_8', 'comment_feed_emb_9']

    fu_comment_cols = ['comment_fu_w2v_0', 'comment_fu_w2v_1', 'comment_fu_w2v_2', 'comment_fu_w2v_3',
                       'comment_fu_w2v_4', 'comment_fu_w2v_5', 'comment_fu_w2v_6', 'comment_fu_w2v_7',
                       'comment_fu_w2v_8', 'comment_fu_w2v_9']

    read_comment_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                         and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                         and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                         and i not in fu_comment_cols]

    comment_cols = [i for i in use_cols if ('read_comment_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i) and (
                                'like_feed_emb' not in i)
                    and ('read_comment_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                    and ('follow_fu' not in i) and ('favorite_fu' not in i) and ('like_fu' not in i)]

    like_cols = [i for i in use_cols if ('read_comment_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                 and ('read_comment_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                 and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                 and i not in fu_comment_cols]

    click_avatar_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('read_comment_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                         and ('like_fu' not in i) and ('read_comment_fu' not in i) and ('forward_fu' not in i)
                         and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                         and i not in fu_comment_cols]

    forward_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                    and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                    and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                    and i not in fu_comment_cols]

    follow_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('forward_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                   and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                   and ('forward_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                   and i not in fu_comment_cols]

    favorite_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('forward_feed_emb' not in i)
                     and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                     and ('follow_fu' not in i) and ('forward_fu' not in i) and i not in emb_comment_cols
                     and i not in fu_comment_cols]

    train_feats = [read_comment_cols, comment_cols, like_cols, click_avatar_cols, forward_cols, follow_cols, favorite_cols]

    for i, j in zip(label_cols, train_feats):
        train_set['num_label'] = train_set.groupby('userid')[i].transform('nunique')
        tmp_train = train_set[train_set['num_label'] == 2]
        del tmp_train['num_label']
        tmp_train = tmp_train[['feedid', 'userid', 'date_'] + [i] + j].reset_index(drop=True)
        print(f'{i} now train shape:', tmp_train.shape)
        tmp_train.to_feather(f'{feat_path}/for_train_{i}.feather')
        del tmp_train

        valid_set['num_label'] = valid_set.groupby('userid')[i].transform('nunique')
        tmp_valid = valid_set[valid_set['num_label'] == 2]
        tmp_valid = tmp_valid[['feedid', 'userid', 'date_'] + [i] + j].reset_index(drop=True)
        print(f'{i} now valid shape:', tmp_valid.shape)
        tmp_valid.to_feather(f'{feat_path}/for_valid_{i}.feather')
        del tmp_valid


def split_all():
    train_set = pd.read_feather(feat_path + '/round2_train_feats.feather')
    print('origin shape', train_set.shape)

    label_cols = ['read_comment', 'comment', 'like', 'click_avatar', 'forward', 'follow', 'favorite']
    use_cols = [i for i in train_set.columns if i not in ['feedid', 'userid', 'date_'] + label_cols]

    emb_comment_cols = ['comment_feed_emb_0', 'comment_feed_emb_1', 'comment_feed_emb_2', 'comment_feed_emb_3',
                        'comment_feed_emb_4', 'comment_feed_emb_5', 'comment_feed_emb_6', 'comment_feed_emb_7',
                        'comment_feed_emb_8', 'comment_feed_emb_9']

    fu_comment_cols = ['comment_fu_w2v_0', 'comment_fu_w2v_1', 'comment_fu_w2v_2', 'comment_fu_w2v_3',
                       'comment_fu_w2v_4', 'comment_fu_w2v_5', 'comment_fu_w2v_6', 'comment_fu_w2v_7',
                       'comment_fu_w2v_8', 'comment_fu_w2v_9']

    read_comment_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                         and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                         and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                         and i not in fu_comment_cols]

    comment_cols = [i for i in use_cols if
                    ('read_comment_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
                            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and (
                                'favorite_feed_emb' not in i) and (
                            'like_feed_emb' not in i)
                    and ('read_comment_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                    and ('follow_fu' not in i) and ('favorite_fu' not in i) and ('like_fu' not in i)]

    like_cols = [i for i in use_cols if ('read_comment_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                 and ('read_comment_fu' not in i) and ('click_avatar_fu' not in i) and ('forward_fu' not in i)
                 and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                 and i not in fu_comment_cols]

    click_avatar_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('read_comment_feed_emb' not in i) and (
            'forward_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                         and ('like_fu' not in i) and ('read_comment_fu' not in i) and ('forward_fu' not in i)
                         and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                         and i not in fu_comment_cols]

    forward_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                    and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                    and ('follow_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                    and i not in fu_comment_cols]

    follow_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('forward_feed_emb' not in i) and ('favorite_feed_emb' not in i)
                   and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                   and ('forward_fu' not in i) and ('favorite_fu' not in i) and i not in emb_comment_cols
                   and i not in fu_comment_cols]

    favorite_cols = [i for i in use_cols if ('like_feed_emb' not in i) and ('click_avatar_feed_emb' not in i) and (
            'read_comment_feed_emb' not in i) and ('follow_feed_emb' not in i) and ('forward_feed_emb' not in i)
                     and ('like_fu' not in i) and ('click_avatar_fu' not in i) and ('read_comment_fu' not in i)
                     and ('follow_fu' not in i) and ('forward_fu' not in i) and i not in emb_comment_cols
                     and i not in fu_comment_cols]

    train_feats = [read_comment_cols, comment_cols, like_cols, click_avatar_cols, forward_cols, follow_cols,
                   favorite_cols]

    for i, j in zip(label_cols, train_feats):
        train_set['num_label'] = train_set.groupby('userid')[i].transform('nunique')
        tmp_all_train = train_set[train_set['num_label'] == 2]
        del train_set['num_label']
        tmp_all_train = tmp_all_train[['feedid', 'userid', 'date_'] + [i] + j].reset_index(drop=True)
        print(f'{i} now all train shape:', tmp_all_train.shape)
        tmp_all_train.to_feather(f'{feat_path}/all_train_{i}.feather')


if __name__ == "__main__":
    file_path = './feats/user_action_drop_repeat.feather'
    we_path = './wbdc2021/data/wedata/wechat_algo_data2'
    sub_path = './sub'
    feat_path = './feats'
    if not os.path.exists(sub_path):
        os.mkdir(sub_path)

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)

    get_ori_train()
    gen_feats(we_path, file_path, feat_path)
    tag_ctr_filter(feat_path, 'machine_keyword_list')
    tag_ctr_filter(feat_path, 'manual_tag_list')
    merge_file(feat_path + '/base_tm_train')
    merge_file(feat_path + '/manual_tag_list_user_ctr')
    merge_file(feat_path + '/machine_keyword_list_user_ctr')
    merge_file(feat_path + '/tm_ratio_train')
    load_train_file(feat_path)
    load_test_file(feat_path, we_path + '/test_a.csv')
    split_train_valid()
    split_all()
