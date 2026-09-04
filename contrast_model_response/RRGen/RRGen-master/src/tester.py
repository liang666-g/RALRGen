## load test data and output BLEU score
import os, sys
from tqdm import tqdm
from translate import translate
from metrics.nmt_bleu import compute_bleu

def _load_test_data(test_filename, train_dataset):
    test_src_texts = []
    test_tgt_texts = []
    test_rates = []
    test_cates = []
    test_sentis = []

    FIELD_SEP = "-[split]-"

    with open(test_filename, "r", encoding="utf-8-sig", errors="replace") as test_fr:
        for line in test_fr:
            terms = line.rstrip("\n").split(FIELD_SEP)

            if len(terms) < 8:
                continue

            senti_sent = terms[7].strip().split()
            try:
                if int(senti_sent[1]) * 1.5 + int(senti_sent[0]) < 0:
                    test_sentis.append(senti_sent[1])
                else:
                    test_sentis.append(senti_sent[0])
            except Exception:
                continue

            try:
                test_cates.append(train_dataset.app_cate_dict[terms[0]])
                test_rates.append(terms[1])
                test_src_texts.append(terms[4])
                test_tgt_texts.append(terms[5])
            except KeyError as e:
                print("[!] App not found in app_category.txt:", e)
                continue

    test_rate_sents = [train_dataset.rate_vocab.token2id[i] for i in test_rates]
    test_cate_sents = [train_dataset.cate_vocab.token2id[i] for i in test_cates]
    test_senti_sents = [train_dataset.senti_vocab.token2id[i] for i in test_sentis]

    return test_src_texts, test_tgt_texts, test_rate_sents, test_cate_sents, test_senti_sents

def _valid_test(test_src_texts, test_tgt_texts, test_rate_sents, test_cate_sents, test_senti_sents, train_dataset, encoder, decoder, max_seq_len):
    # --------------------------
    ### Start translation
    # --------------------------


    # test_src_texts = [line.split('-[split]-')[4] for line in test_fr.readlines()]
    # test_tgt_texts = [ for line in test_fr.readlines()]
    print(len(test_src_texts), len(test_tgt_texts))
    print("==============Start evaluation on test data==============")
    references = []
    candidates = []
    out_texts = []
    attention_weights = []
    eval_bar = tqdm(
        enumerate(test_src_texts),
        total=len(test_src_texts),
        desc="Evaluating",
        file=sys.stdout,
        ncols=100,
        dynamic_ncols=False,
        leave=True,
        mininterval=1.0,
        position=0
    )

    for idx, src_text in eval_bar:
        _, out_text, attention_weight = translate(src_text.strip(), test_rate_sents[idx], test_cate_sents[idx],
                                                  test_senti_sents[idx], train_dataset, encoder, decoder,
                                                  max_seq_len=max_seq_len)
        references.append([test_tgt_texts[idx].strip().split()])
        candidates.append(out_text.split())
        out_texts.append(out_text)
        attention_weights.append(attention_weight.numpy())
    bleu_4, pls, _, _, _, _ = compute_bleu(references, candidates)
    return bleu_4, pls, attention_weights, out_texts
