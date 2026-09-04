from tqdm import tqdm
import numpy as np
import torch
from configuration import special_identifier as si


class Embedding():
    """
    Load embedding file
    """

    def __init__(self, filename, embedding_size):
        self.filename = filename
        self.embedding_size = embedding_size

    # word_to_index保存了work到index的映射，这个映射用于在word_vectors中取出向量
    def load_word_vectors(self):
        word_to_index = {}
        word_vectors = []

        with open(self.filename, encoding='utf-8') as fp:
            for line in tqdm(fp, leave=False):
                line = line.split(" ")

                word = line[0]  # 取出一个单词
                word_to_index[word] = len(word_to_index)  # 按照取出顺序编号

                vec = np.array([float(x) for x in line[1:]])  # 取出对应的向量
                word_vectors.append(vec)  # 按顺序添加对应的向量

        return word_to_index, word_vectors

    # 根据word_vocab构造embeding
    def load_embedding(self, word_vocab):
        '''
        :param word_vocab: Word to index dictionary
        :param embedding_size: The size of word embedding
        :return:
        '''
        word_to_index, word_vectors = self.load_word_vectors()
        vocab_size = len(word_vocab.token2id)
        embedding = np.zeros((vocab_size, self.embedding_size))  # 根据vocab的size构建word对应的向量表
        unk_count = 0
        for word, emb_index in tqdm(word_vocab.token2id.items()):  # 取出vocab中的word和他们的依次的顺序
            if word == word_vocab.id2token[si.PAD]:  # 如果得出word是PAD就取零向量
                continue
            elif word in [word_vocab.id2token[si.BOS], word_vocab.id2token[si.EOS],
                          word_vocab.id2token[si.UNK]]:  # bos/eos/unk为随机向量
                embedding[emb_index, :] = np.random.rand(self.embedding_size, )
            elif word in word_to_index:
                glove_index = word_to_index[word.lower()]
                glove_vec = torch.FloatTensor(word_vectors[glove_index])  ##cuda.
                embedding[emb_index, :] = glove_vec
            else:
                embedding[emb_index, :] = embedding[si.UNK]
                unk_count += 1

        print('- Unknown word count: {}'.format(unk_count))
        print('=' * 100 + '\n')
        return torch.from_numpy(embedding).float()  # 根据的是从vocab取出的word的顺序，结合glove文件得到向量