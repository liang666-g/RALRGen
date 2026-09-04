"""
Nearest Neighbor GEN (k=5)
Feature: Term frequence, BLEU-4
"""

from sklearn.neighbors import NearestNeighbors
from gensim.corpora import Dictionary
import gensim
import numpy as np
import subprocess
import os
from scipy import sparse
from gensim.parsing.preprocessing import STOPWORDS
from nltk.translate.bleu_score import sentence_bleu
import time
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

def compute_blue4(tar, pred_lst):
    """
    compute bleu4
    :param tar_lst:
    :param pred_lst:
    :return:
    """
    score_vec = np.zeros(len(pred_lst))
    for i, pred in enumerate(pred_lst):
        score_vec[i] = sentence_bleu([tar], pred, weights=(0.25, 0.25, 0.25, 0.25))
    return np.argmax(score_vec)


def evaluate(review_lst, tar_lst, pred_lst):
    """
    仅将 NNGen 的生成结果保存为纯文本，后续用 Python 脚本统一算分。
    """ 
    import os
    def process_data(reviews, replies, fn):
        text_w_lst = []
        for rev, rep in zip(reviews, replies):
            # 将 token 列表拼接成完整的句子字符串
            rev_str = " ".join(rev)
            rep_str = " ".join(rep)
            # 采用和 NMT 完全一样的拼接格式
            text_w_lst.append(f"{rev_str}-[split]-{rep_str}\n")
            
        with open(fn, "w", encoding="utf-8") as fout:
            fout.writelines(text_w_lst)

    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tar_fn = os.path.join(current_dir, "tar.txt")
    pred_fn = os.path.join(current_dir, "pred.txt")
    
    process_data(review_lst, tar_lst, tar_fn)
    process_data(review_lst, pred_lst, pred_fn)
    
    print(f"✅ 参考答案已保存至: {tar_fn}")
    print(f"✅ 预测结果已保存至: {pred_fn}")
    print("💡 格式已对齐 NMT！请使用现有的 Python 评测脚本去读取并计算分数！")


def build_dict(data_lst):
    dictionary = Dictionary(data_lst)
    dictionary.filter_tokens(list(map(dictionary.token2id.get, STOPWORDS)))
    dictionary.filter_extremes(no_below=3)  #, keep_n=10000)
    dictionary.compactify()
    return dictionary


def get_bow(text_data, dictionary):
    row = []
    col = []
    val = []
    r_l = 0
    for text in text_data:
        for i, j in dictionary.doc2bow(text):
            row.append(r_l)
            col.append(i)
            val.append(j)
        r_l += 1
    bow_mat = sparse.coo_matrix((val, (row, col)), shape=(r_l, len(dictionary)))
    return bow_mat


def read_data(fn):
    review_lst = []
    reply_lst = []
    with open(fn, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split("-[split]-")
            review = terms[4]
            reply = terms[5]
            review_lst.append(list(gensim.utils.tokenize(review, lower=True)))
            reply_lst.append(list(gensim.utils.tokenize(reply, lower=True)))
    return review_lst, reply_lst

if __name__ == '__main__':
    train_review_lst, train_reply_lst = read_data("./data/train.txt")
    test_review_lst, test_reply_lst = read_data("./data/test.txt")
    assert len(train_review_lst) == len(train_reply_lst)
    assert len(test_review_lst) == len(test_reply_lst)
    dictionary = build_dict(train_review_lst)
    train_bow = get_bow(train_review_lst, dictionary)
    test_bow = get_bow(test_review_lst, dictionary)

    start_t = time.time()
    print("building NN module...")
    model = NearestNeighbors(n_neighbors=5).fit(train_bow)
    inds = model.kneighbors(test_bow, return_distance=False)

    topK_lst = []
    for ins in inds:
        topK_ = [train_review_lst[i] for i in ins.ravel()]
        topK_lst.append(topK_)

    print("counting bleu match...")
    pred_reply_lst = []
    for topK_candidate, target_review, ins in zip(topK_lst, test_review_lst, inds):
        ind = compute_blue4(target_review, topK_candidate)
        pred_reply = train_reply_lst[ins[ind]]
        pred_reply_lst.append(pred_reply)

    end_t = time.time()
    assert len(pred_reply_lst) == len(test_reply_lst)
    print("evaluating ...")
    # 把 test_review_lst 传给 evaluate 函数
    evaluate(test_review_lst, test_reply_lst, pred_reply_lst)
    print("Elapse time: %s" % time.strftime("%H:%M:%S", time.gmtime(end_t - start_t)))





