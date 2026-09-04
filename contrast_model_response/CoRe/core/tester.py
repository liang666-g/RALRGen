import os, sys
from tqdm import tqdm
from translate import translate
from configuration import file_space

from metrics.nmt_bleu import compute_bleu
from configuration import runfig
from metrics.evaluation import compute_rouge, compute_meteor
import random


def _load_test_data(test_file, desc_file, test_simi_file):
    # load test data
    test_src_texts = []
    test_tgt_texts = []
    test_desc_texts = []
    test_simi_texts = []
    test_extra_review_texts = []

    train_tgt_texts = []
    train_src_texts = []

    desc_map = {}
    with open(desc_file, encoding='utf-8-sig') as f:
        for sent in tqdm(f.readlines()):
            terms = sent.split('-[split]-')
            desc_map[terms[0]] = [token for token in terms[1].split()]

    with open(test_file, encoding='utf-8-sig') as f:
        for sent in tqdm(f.readlines()):
            terms = sent.split('-[split]-')
            if len(terms) < 8:  # check term length
                continue
            app_id = terms[0]
            test_desc_texts.append(desc_map[app_id])
            test_src_texts.append(terms[4])
            test_tgt_texts.append(terms[5])

    with open(file_space.train_file, encoding='utf-8-sig') as f:
        for sent in tqdm(f.readlines()):
            terms = sent.split('-[split]-')
            if len(terms) < 8:  # check term length
                continue
            train_tgt_texts.append(terms[5])
            train_src_texts.append(terms[4])

    with open(test_simi_file, encoding='utf-8-sig') as f:
        for sent in tqdm(f.readlines()):
            terms = sent.split()
            test_extra_review_texts.append(train_src_texts[int(terms[0])])
            if runfig.tfidf_N == 1:
                test_simi_texts.append(train_tgt_texts[int(terms[0])])
            else:
                temp_test_simi_text = []
                for i in range(runfig.tfidf_N):
                    temp_test_simi_text.append(train_tgt_texts[int(terms[i])])
                test_simi_texts.append(temp_test_simi_text)

    return test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts


def _valid_test(test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts,
                train_dataset, encoder, decoder,
                max_seq_len):
    # --------------------------
    # Start translation
    # --------------------------

    # test_src_texts = [line.split('-[split]-')[4] for line in test_fr.readlines()]
    # test_tgt_texts = [ for line in test_fr.readlines()]
    print(len(test_src_texts), len(test_tgt_texts))
    print("==============Start evaluation on test data==============")
    references = []
    candidates = []
    out_texts = []
    attention_src_weights = []
    attention_simi_weights = []
    attention_simi_weights_1 = []
    attention_simi_weights_2 = []
    attention_desc_weights = []

    for idx, src_text in tqdm(enumerate(test_src_texts)):
        _, out_text, attention_src_weight, attention_simi_weight, attention_desc_weight = translate(src_text.strip(),
                                                                                                    test_desc_texts[
                                                                                                        idx],
                                                                                                    test_simi_texts[
                                                                                                        idx],
                                                                                                    test_extra_review_texts[
                                                                                                        idx],
                                                                                                    train_dataset,
                                                                                                    encoder, decoder,
                                                                                                    max_seq_len=max_seq_len)
        references.append([test_tgt_texts[idx].strip().split()])
        candidates.append(out_text.split())
        out_texts.append(out_text)

        attention_src_weights.append(attention_src_weight.numpy())

        if (runfig.mode == 1 or runfig.mode == 2):
            if (runfig.tfidf_N == 1):
                attention_simi_weights.append(attention_simi_weight.numpy())
            else:
                attention_simi_weights_1.append(attention_simi_weight[0].numpy())
                attention_simi_weights_2.append(attention_simi_weight[1].numpy())

        if (runfig.mode == 1 or runfig.mode == 3):
            attention_desc_weights.append(attention_desc_weight.numpy())

    if (runfig.tfidf_N != 1):
        attention_simi_weights = [attention_simi_weights_1, attention_simi_weights_2]
    bleu_4, pls, _, _, _, _ = compute_bleu(references, candidates)
    rouge_score = compute_rouge(test_tgt_texts, out_texts)
    meteor_score = compute_meteor(test_tgt_texts, out_texts)
    return bleu_4, pls, rouge_score, meteor_score, out_texts, attention_src_weights, attention_simi_weights, attention_desc_weights
