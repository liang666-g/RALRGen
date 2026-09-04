from tqdm import tqdm
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from collections import Counter
from configuration import special_identifier as si
from configuration import runfig,encoder_config,decoder_config
import torch
import random


class Dataset():
    def __init__(self, data_file, desc_file, simi_file, max_vocab_size=10000):
        self.tgt_sents_text = []
        self.max_vocab_size = max_vocab_size
        self.src_sents, self.tgt_sents, self.app_ids = self.load_sents(data_file)
        self.desc_map = self.load_desc(desc_file)
        self.simi = self.load_simi_file(simi_file)
        self.input_counter = self.build_counter(self.src_sents + self.tgt_sents)
        self.output_counter = self.build_counter(self.tgt_sents)
        self.desc_counter = self.build_desc_counter(self.desc_map)
        self.input_vocab = self.build_vocab(self.input_counter, self.max_vocab_size)
        self.output_vocab = self.build_vocab(self.output_counter, self.max_vocab_size)

    def load_sents(self, fpaths):
        src_sents = []
        tgt_sents = []
        app_ids = []
        with open(fpaths, encoding='utf-8-sig') as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split('-[split]-')
                if len(terms) < 8:  # check term length
                    continue
                app_id = terms[0]
                src_sent = terms[4]
                tgt_sent = terms[5]
                self.tgt_sents_text.append(tgt_sent)
                src_tokens = [token for token in src_sent.split()]
                tgt_tokens = [token for token in tgt_sent.split()]
                app_ids.append(app_id)
                src_sents.append(src_tokens)
                tgt_sents.append(tgt_tokens)
        return src_sents, tgt_sents, app_ids

    def load_desc(self, desc_file):
        id2desc = {}
        with open(desc_file, encoding='utf-8-sig') as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split('-[split]-')
                id2desc[terms[0]] = [token for token in terms[1].split()]
        return id2desc

    def build_desc_counter(self, desc_map):
        counter = Counter()
        for v in tqdm(desc_map.values()):
            counter.update(v)
        return counter

    def get_desc(self, app_id):
        return self.desc_map[app_id]

    def __getitem__(self, index):
        src_sent = self.src_sents[index]
        src_seq = self.tokens2ids(src_sent, self.input_vocab.token2id, append_BOS=False, append_EOS=True)

        if len(src_seq) > encoder_config.max_seq_len:
            src_seq = src_seq[:encoder_config.max_seq_len-1]
            src_seq.append(si.EOS)

        desc_sent = self.desc_map[self.app_ids[index]]

        desc_seq = self.tokens2ids(desc_sent, self.input_vocab.token2id, append_BOS=False, append_EOS=True)

        if len(desc_seq) > encoder_config.max_seq_len:
            desc_seq = desc_seq[:encoder_config.max_seq_len-1]
            desc_seq.append(si.EOS)

        simi_numbers = self.simi[index]

        if runfig.tfidf_N == 1:
            simi = int(simi_numbers[0])
            simi_sent = self.tgt_sents[simi]
            simi_seq = self.tokens2ids(simi_sent, self.input_vocab.token2id, append_BOS=False, append_EOS=True)
            if len(simi_seq) > encoder_config.max_seq_len:
                simi_seq = simi_seq[:encoder_config.max_seq_len - 1]
                simi_seq.append(si.EOS)

        else:
            simi = []
            for i in range(runfig.tfidf_N):
                try:
                    simi.append(int(simi_numbers[i]))
                except:
                    simi.append(simi[-1])
            simi_sent = []
            for i in simi:
                simi_sent.append(self.tgt_sents[i])
            simi_seq = []
            for i in simi_sent:
                cur_simi_seq = self.tokens2ids(i, self.input_vocab.token2id, append_BOS=False, append_EOS=True)
                if len(cur_simi_seq) > encoder_config.max_seq_len:
                    cur_simi_seq = cur_simi_seq[:encoder_config.max_seq_len - 1]
                    cur_simi_seq.append(si.EOS)
                simi_seq.append(cur_simi_seq)

        tgt_sent = self.tgt_sents[index]
        tgt_seq = self.tokens2ids(tgt_sent, self.output_vocab.token2id, append_BOS=False, append_EOS=True)
        if len(tgt_seq) > decoder_config.max_seq_len:
            tgt_seq = tgt_seq[:decoder_config.max_seq_len - 1]
            tgt_seq.append(si.EOS)


        extra_review_sent = self.src_sents[int(simi_numbers[0])]
        extra_review_seq = self.tokens2ids(extra_review_sent, self.input_vocab.token2id, append_BOS=False, append_EOS=True)

        if len(extra_review_seq) > encoder_config.max_seq_len:
            extra_review_seq = extra_review_seq[:encoder_config.max_seq_len - 1]
            extra_review_seq.append(si.EOS)

        return src_seq, desc_seq, simi_seq, tgt_seq, extra_review_seq

    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(si.BOS)
        seq.extend([token2id.get(token, si.UNK) for token in tokens])
        if append_EOS: seq.append(si.EOS)
        return seq

    def __len__(self):
        return len(self.src_sents)

    def load_simi_file(self, filename):
        similarit = []
        with open(filename, encoding='utf-8-sig') as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split()
                similarit.append(terms)
        return similarit

    def build_counter(self, sents):
        counter = Counter()
        for sent in tqdm(sents):
            counter.update(sent)
        return counter

    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': si.PAD, '<BOS>': si.BOS, '<EOS>': si.EOS, '<UNK>': si.UNK}
        for _id, (token, count) in tqdm(enumerate(counter.most_common(max_vocab_size))):
            vocab.token2id.update({token: _id + 4})
        vocab.id2token = {v: k for k, v in tqdm(vocab.token2id.items())}
        return vocab

def tfidf(srcfile, inputfile, outputfile, n):
    src_sent_app = {}
    response = {}
    vm_app = {}
    i = 0
    with open(srcfile, encoding='utf-8-sig') as f:
        for sent in tqdm(f.readlines()):
            terms = sent.split('-[split]-')
            if len(terms) < 8:  # check term length
                continue
            app_id = terms[0]
            src_sent = terms[4]

            if (app_id not in src_sent_app.keys()):
                src_sent_app[app_id] = []
            src_sent_app[app_id].append(src_sent)

            if (app_id not in response.keys()):
                response[app_id] = []
            response[app_id].append(i)
            i = i + 1
    print("字典完成")

    for app, text in src_sent_app.items():
        for i in range(len(text)):
            text[i] = re.sub(r"[.,!?<>():;\[\]]", " ", text[i])
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(text)
        if (app not in vm_app.keys()):
            vm_app[app] = [vectorizer, matrix]

    print("vectorizer和matrix完成")

    index = []
    m = 1
    with open(inputfile, encoding='utf-8-sig') as f:
        for sent in f.readlines():
            terms = sent.split('-[split]-')
            if len(terms) < 8:  # check term length
                continue
            src_sent = terms[4]
            app_id = terms[0]

            vectorizer, matrix = vm_app[app_id]

            query_tfidf = vectorizer.transform([src_sent])
            nums = list(cosine_similarity(query_tfidf, matrix).flatten())

            temp = []
            for value in heapq.nlargest(n + 1, nums):
                i = nums.index(value)
                while (i in temp):
                    try:
                        i = nums.index(value, i + 1)
                    except:
                        print(temp)
                        print(heapq.nlargest(n + 1, nums))
                        for p, kk in enumerate(nums):
                            if (kk == value):
                                print(p)
                        raise ValueError('wrong')
                temp.append(i)
            index.append([response[app_id][i] for i in temp])
            if ((m % 1000) == 0):
                print("完成" + str(m))
            m = m + 1
    with open(outputfile, "wt", encoding='utf-8', newline='') as f:
        for temp in index:
            for t in temp:
                f.write(str(t) + " ")
            f.write("\n")


def collate_fn(data):
    """
        Creates mini-batch tensors from (src_sent, tgt_sent, src_seq, tgt_seq).
        We should build a custom collate_fn rather than using default collate_fn,
        because merging sequences (including padding) is not supported in default.
        Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).

        Args:
            data: list of tuple (src_sents, tgt_sents, src_seqs, tgt_seqs)
            - src_sents, tgt_sents: batch of original tokenized sentences
            - src_seqs, tgt_seqs: batch of original tokenized sentence ids
        Returns:
            - src_sents, tgt_sents (tuple): batch of original tokenized sentences
            - src_seqs, tgt_seqs (variable): (max_src_len, batch_size)
            - src_lens, tgt_lens (tensor): (batch_size)
    """

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    # Sort a list by *source* sequence length (descending order) to use `pack_padded_sequence`.
    # The *target* sequence is not sorted <-- It's ok, cause `pack_padded_sequence` only takes
    # *source* sequence, which is in the EncoderRNN
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # Seperate source and target sequences.
    src_seq, desc_seq, simi_seq, tgt_seq, extra_review_seq = zip(*data)

    # Merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lens = _pad_sequences(src_seq)

    extra_review_seqs, extra_review_lens = _pad_sequences(extra_review_seq)

    if(runfig.tfidf_N==1):
        simi_seqs, simi_lens = _pad_sequences(simi_seq)
    else:
        simi_seqs = []
        simi_lens = []
        simi_seq1 = []
        simi_seq2 = []
        simi_seq3 = []
        simi_seq4 = []
        simi_seq5 = []
        for i in simi_seq:
            if(runfig.tfidf_N > 1):
                simi_seq1.append(i[0])
                simi_seq2.append(i[1])
            if(runfig.tfidf_N > 2):
                simi_seq3.append(i[2])
            if(runfig.tfidf_N > 3):
                simi_seq4.append(i[3])
            if(runfig.tfidf_N > 4):
                simi_seq5.append(i[4])
        simi_seq = [simi_seq1, simi_seq2, simi_seq3, simi_seq4, simi_seq5]
        for i in range(runfig.tfidf_N):
            cur_simi_seq, cur_simi_len = _pad_sequences(simi_seq[i])
            simi_seqs.append(cur_simi_seq)
            simi_lens.append(cur_simi_len)
        
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seq)
    desc_seqs, desc_lens = _pad_sequences(desc_seq)

    # (batch, seq_len) => (seq_len, batch)
    src_seqs = src_seqs.transpose(0, 1)

    if(runfig.tfidf_N==1):
        simi_seqs = simi_seqs.transpose(0, 1)
    else:
        for i in range(len(simi_seqs)):
            simi_seqs[i] = simi_seqs[i].transpose(0, 1)


    tgt_seqs = tgt_seqs.transpose(0, 1)
    desc_seqs = desc_seqs.transpose(0, 1)
    extra_review_seqs = extra_review_seqs.transpose(0, 1)

    return src_seqs, simi_seqs, desc_seqs, tgt_seqs, extra_review_seqs, src_lens, simi_lens, desc_lens, tgt_lens, extra_review_lens


class AttrDict(dict):
    """ Access dictionary keys like attribute
        https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
    """

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self