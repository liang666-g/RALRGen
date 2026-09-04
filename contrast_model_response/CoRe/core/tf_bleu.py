"""
Nearest Neighbor GEN (k=5)
Feature: Term frequence, BLEU-4
"""

from sklearn.neighbors import NearestNeighbors
from gensim.corpora import Dictionary
import gensim
import numpy as np
import os
from scipy import sparse
from gensim.parsing.preprocessing import STOPWORDS
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
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
    smooth = SmoothingFunction().method1
    score_vec = np.zeros(len(pred_lst))
    for i, pred in enumerate(pred_lst):
        score_vec[i] = sentence_bleu(
            [tar],
            pred,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth
        )
    return np.argmax(score_vec)


def evaluate(tar_lst, pred_lst):
    """
    compute bleu in corpus level
    :param tar_lst:
    :param pred_lst:
    :return:
    """
    smooth = SmoothingFunction().method1

    # corpus_bleu 需要：
    # references = [[ref1_tokens], [ref2_tokens], ...]
    # hypotheses = [pred1_tokens, pred2_tokens, ...]
    references = [[tar] for tar in tar_lst]
    hypotheses = pred_lst

    bleu1 = corpus_bleu(
        references,
        hypotheses,
        weights=(1.0, 0, 0, 0),
        smoothing_function=smooth
    )
    bleu2 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smooth
    )
    bleu3 = corpus_bleu(
        references,
        hypotheses,
        weights=(1 / 3, 1 / 3, 1 / 3, 0),
        smoothing_function=smooth
    )
    bleu4 = corpus_bleu(
        references,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )

    print("BLEU-1: %.6f" % bleu1)
    print("BLEU-2: %.6f" % bleu2)
    print("BLEU-3: %.6f" % bleu3)
    print("BLEU-4: %.6f" % bleu4)


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
    with open(fn, encoding='utf-8-sig') as fin:
        lines = fin.readlines()
        for line in lines:
            terms = line.strip().split("-[split]-")
            review = terms[4]
            reply = terms[5]
            review_lst.append(list(gensim.utils.tokenize(review, lower=True)))
            reply_lst.append(list(gensim.utils.tokenize(reply, lower=True)))
    return review_lst, reply_lst

if __name__ == '__main__':
    train_review_lst, train_reply_lst = read_data("./data/rrgen_train_data.txt")
    test_review_lst, test_reply_lst = read_data("./data/rrgen_test_data.txt")
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
    evaluate(test_reply_lst, pred_reply_lst)
    print("Elapse time: %s" % time.strftime("%H:%M:%S", time.gmtime(end_t - start_t)))





