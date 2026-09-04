import torch
import torch.nn as nn
from util import sequence_mask
from configuration import runfig
import os
import pickle
import numpy as np


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, encoder, embedding=None, attention=True, bias=True, dropout=0.3,
                 tie_ext_feature=False, ext_rate_embedding=None, ext_appcate_embedding=None, ext_seqlen_embedding=None,
                 ext_senti_embedding=None):
        """ General attention in `Effective Approaches to Attention-based Neural Machine Translation`
            Ref: https://arxiv.org/abs/1508.04025

            Share input and output embeddings:
            Ref:
                - "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
                   https://arxiv.org/abs/1608.05859
                - "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
                   https://arxiv.org/abs/1611.01462
        """
        super(LuongAttnDecoderRNN, self).__init__()

        self.hidden_size = encoder.hidden_size * encoder.num_directions
        self.num_layers = encoder.num_layers
        self.dropout = dropout
        self.embedding = embedding
        self.attention = attention

        self.vocab_size = self.embedding.num_embeddings
        self.word_vec_size = self.embedding.embedding_dim

        self.rnn_type = encoder.rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=self.word_vec_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout)

        if self.attention:
            self.W_a = nn.Linear(encoder.hidden_size * encoder.num_directions,
                                 self.hidden_size, bias=bias)
            self.W_c = nn.Linear(encoder.hidden_size * encoder.num_directions + self.hidden_size,
                                 self.hidden_size, bias=bias)
            self.W_H = nn.Linear(encoder.hidden_size * encoder.num_directions, 1, bias=bias)

            self.W_S = nn.Linear(encoder.hidden_size * encoder.num_directions, 1, bias=bias)

            self.W_X = nn.Linear(self.word_vec_size, 1, bias=bias)

        self.W_s = nn.Linear(self.hidden_size, self.vocab_size, bias=bias)

        input_size = self.hidden_size

        self.W_r = nn.Linear(input_size, self.hidden_size, bias=bias)

    def atten(self, decoder_output, encoder_outputs, lens):

        attention_scores = torch.bmm(decoder_output, self.W_a(encoder_outputs).transpose(0, 1).transpose(1, 2))

        # attention_mask: (batch_size, seq_len=1, max_src_len)
        attention_mask = sequence_mask(lens).unsqueeze(1)
        # Fills elements of tensor with `-float('inf')` where `mask` is 1.

        attention_scores.data.masked_fill_(~attention_mask.data, -float('inf'))

        # attention_weights: (batch_size, seq_len=1, max_src_len) => (batch_size, max_src_len) for `F.softmax`
        # => (batch_size, seq_len=1, max_src_len)
        try:  # torch 0.3.x
            attention_weights = torch.softmax(attention_scores.squeeze(1), dim=1).unsqueeze(1)
        except:
            attention_weights = torch.softmax(attention_scores.squeeze(1)).unsqueeze(1)

        # context_vector:
        # (batch_size, seq_len=1, max_src_len) * (batch_size, max_src_len, encoder_hidden_size * num_directions)
        # => (batch_size, seq_len=1, encoder_hidden_size * num_directions)
        context_vector = torch.bmm(attention_weights, encoder_outputs.transpose(0, 1))
        return attention_weights, context_vector


    def forward(self, input_seq, decoder_hidden, encoder_src_outputs, encoder_simi_outputs, encoder_desc_outputs,
                encoder_extra_review_outputs, src_lens, simi_lens, desc_lens, extra_review_lens):
        """ Args:
            - input_seq      : (batch_size)
            - decoder_hidden : (t=0) last encoder hidden state (num_layers * num_directions, batch_size, hidden_size)
                               (t>0) previous decoder hidden state (num_layers, batch_size, hidden_size)
            - encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)

            Returns:
            - output           : (batch_size, vocab_size)
            - decoder_hidden   : (num_layers, batch_size, hidden_size)
            - attention_weights: (batch_size, max_src_len)
        """
        # (batch_size) => (seq_len=1, batch_size)
        input_seq = input_seq.unsqueeze(0)

        # (seq_len=1, batch_size) => (seq_len=1, batch_size, word_vec_size)
        emb = self.embedding(input_seq)

        # rnn returns:
        # - decoder_output: (seq_len=1, batch_size, hidden_size)
        # - decoder_hidden: (num_layers, batch_size, hidden_size)

        # decoder_hidden = torch.cat((decoder_hidden, ext_rate_embedding, ext_appcate_embedding, ext_seqlen_embedding, ext_senti_embedding), 2)
        decoder_hidden = torch.tanh(self.W_r(decoder_hidden))
        decoder_output, decoder_hidden = self.rnn(emb, decoder_hidden)

        # (seq_len=1, batch_size, hidden_size) => (batch_size, seq_len=1, hidden_size)
        decoder_output = decoder_output.transpose(0, 1)
        device = decoder_output.device
        """ 
        ------------------------------------------------------------------------------------------
        Notes of computing attention scores
        ------------------------------------------------------------------------------------------
        # For-loop version:

        max_src_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        attention_scores = Variable(torch.zeros(batch_size, max_src_len))

        # For every batch, every time step of encoder's hidden state, calculate attention score.
        for b in range(batch_size):
            for t in range(max_src_len):
                # Loung. eq(8) -- general form content-based attention:
                attention_scores[b,t] = decoder_output[b].dot(attention.W_a(encoder_outputs[t,b]))

        ------------------------------------------------------------------------------------------
        # Vectorized version:

        1. decoder_output: (batch_size, seq_len=1, hidden_size)
        2. encoder_outputs: (max_src_len, batch_size, hidden_size * num_directions)
        3. W_a(encoder_outputs): (max_src_len, batch_size, hidden_size)
                        .transpose(0,1)  : (batch_size, max_src_len, hidden_size) 
                        .transpose(1,2)  : (batch_size, hidden_size, max_src_len)
        4. attention_scores: 
                        (batch_size, seq_len=1, hidden_size) * (batch_size, hidden_size, max_src_len) 
                        => (batch_size, seq_len=1, max_src_len)
        """
        attention_src_weights = None
        attention_simi_weights = None
        attention_desc_weights = None

        batch_size = decoder_output.size(0)
        if self.attention:
            if (runfig.mode == 1):
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)
                attention_extra_review_weights, context_extra_review_vector = self.atten(decoder_output,
                                                                                         encoder_extra_review_outputs,
                                                                                         extra_review_lens)
                if (runfig.tfidf_N == 1):
                    attention_simi_weights, context_simi_vector = self.atten(decoder_output, encoder_simi_outputs,
                                                                             simi_lens)
                else:

                    attention_simi_weights = []
                    context_simi_vector = []
                    for i in range(len(encoder_simi_outputs)):
                        cur_attention_simi_weights, cur_context_simi_vector = self.atten(decoder_output,
                                                                                         encoder_simi_outputs[i],
                                                                                         simi_lens[i])
                        attention_simi_weights.append(cur_attention_simi_weights)
                        context_simi_vector.append(cur_context_simi_vector)

                    union_context_simi = torch.cat(context_simi_vector, dim=1)
                    _union_context_simi = union_context_simi.transpose(0, 1)
                    _union_context_simi_lens = torch.LongTensor([runfig.tfidf_N] * batch_size).to(device)
                    atten_simi_beta, context_simi_vector = self.atten(decoder_output, _union_context_simi,
                                                                      _union_context_simi_lens)

                attention_desc_weights, context_desc_vector = self.atten(decoder_output, encoder_desc_outputs,
                                                                         desc_lens)

                # encoder_copy_outputs = torch.cat([context_simi_vector,context_desc_vector],dim=2).transpose(0,1)
                encoder_copy_outputs = ((context_simi_vector + context_desc_vector) / 2).transpose(0, 1)
                # batch_size = encoder_copy_outputs.size(1)
                copy_lens = torch.LongTensor([1] * batch_size).to(device)
                attention_copy_weights, context_copy_vector = self.atten(decoder_output, encoder_copy_outputs,
                                                                         copy_lens)
                
                # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_simi_input = torch.cat([context_simi_vector, decoder_output], -1)
                concat_desc_input = torch.cat([context_desc_vector, decoder_output], -1)
                concat_extra_review_input = torch.cat([context_extra_review_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_simi_output = torch.tanh(self.W_c(concat_simi_input))
                concat_desc_output = torch.tanh(self.W_c(concat_desc_input))
                concat_extra_review_output = torch.tanh(self.W_c(concat_extra_review_input))
                # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)

                gama = attention_copy_weights
                gama_ = torch.ones(batch_size, 1, 1, device=device) - gama

                # concat_copy_output = torch.bmm(gama, self.W_s(concat_simi_output)) + torch.bmm(gama_, self.W_s(concat_desc_output))

                p_gen = torch.sigmoid(
                    self.W_H(context_copy_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))

                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                # output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, concat_copy_output)
                concat_simi_dist = torch.bmm(p_gen_, torch.bmm(gama, self.W_s(concat_simi_output)))
                concat_desc_dist = torch.bmm(p_gen_, torch.bmm(gama_, self.W_s(concat_desc_output)))
                if runfig.additional_review:
                    review_dist = torch.bmm(p_gen, self.W_s(concat_src_output) + self.W_s(concat_extra_review_output))
                else:
                    review_dist = torch.bmm(p_gen, self.W_s(concat_src_output))
                output = concat_simi_dist + concat_desc_dist + review_dist


            elif (runfig.mode == 2):
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)
                if (runfig.tfidf_N == 1):
                    attention_simi_weights, context_simi_vector = self.atten(decoder_output, encoder_simi_outputs,
                                                                             simi_lens)
                else:
                    attention_simi_weights = []
                    context_simi_vector = []
                    for i in range(len(encoder_simi_outputs)):
                        cur_attention_simi_weights, cur_context_simi_vector = self.atten(decoder_output,
                                                                                         encoder_simi_outputs[i],
                                                                                         simi_lens[i])
                        attention_simi_weights.append(cur_attention_simi_weights)
                        context_simi_vector.append((cur_context_simi_vector))

                    temp_context_simi_vector = context_simi_vector[0]
                    for i in range(1, len(context_simi_vector)):
                        temp_context_simi_vector += context_simi_vector[i]
                    temp_context_simi_vector = temp_context_simi_vector / len(context_simi_vector)
                    temp_context_simi_vector = temp_context_simi_vector.transpose(0, 1)
                    temp_batch_size = temp_context_simi_vector.size(1)
                    simi_lens = torch.LongTensor([1] * temp_batch_size).to(device)
                    attention_simi_beta, context_simi_vector = self.atten(decoder_output, temp_context_simi_vector,
                                                                             simi_lens)

                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_simi_input = torch.cat([context_simi_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_simi_output = torch.tanh(self.W_c(concat_simi_input))

                batch_size = context_simi_vector.transpose(0, 1).size(1)
                p_gen = torch.sigmoid(
                    self.W_H(context_simi_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))
                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, self.W_s(concat_simi_output))

            elif (runfig.mode == 3):
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)

                attention_desc_weights, context_desc_vector = self.atten(decoder_output, encoder_desc_outputs,
                                                                         desc_lens)

                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_desc_input = torch.cat([context_desc_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_desc_output = torch.tanh(self.W_c(concat_desc_input))

                p_gen = torch.sigmoid(
                    self.W_H(context_desc_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))

                batch_size = context_desc_vector.transpose(0, 1).size(1)
                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, self.W_s(concat_desc_output))
            elif (runfig.mode == 4):
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)

                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))

                output = self.W_s(concat_src_output)
            elif runfig.mode == 5:
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)

                attention_extra_review_weights, context_extra_review_vector = self.atten(decoder_output, encoder_extra_review_outputs,
                                                                         extra_review_lens)

                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_extra_review_input = torch.cat([context_extra_review_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_extra_review_output = torch.tanh(self.W_c(concat_extra_review_input))

                p_gen = torch.sigmoid(
                    self.W_H(context_extra_review_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))

                batch_size = context_extra_review_vector.transpose(0, 1).size(1)
                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, self.W_s(concat_extra_review_output))
            elif runfig.mode == 6:
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)
                attention_extra_review_weights, context_extra_review_vector = self.atten(decoder_output,
                                                                                         encoder_extra_review_outputs,
                                                                                         extra_review_lens)
                if (runfig.tfidf_N == 1):
                    attention_simi_weights, context_simi_vector = self.atten(decoder_output, encoder_simi_outputs,
                                                                             simi_lens)
                else:

                    attention_simi_weights = []
                    context_simi_vector = []
                    for i in range(len(encoder_simi_outputs)):
                        cur_attention_simi_weights, cur_context_simi_vector = self.atten(decoder_output,
                                                                                         encoder_simi_outputs[i],
                                                                                         simi_lens[i])
                        attention_simi_weights.append(cur_attention_simi_weights)
                        context_simi_vector.append(cur_context_simi_vector)

                    union_context_simi = torch.cat(context_simi_vector, dim=1)
                    _union_context_simi = union_context_simi.transpose(0, 1)
                    _union_context_simi_lens = torch.LongTensor([runfig.tfidf_N] * batch_size).to(device)
                    atten_simi_beta, context_simi_vector = self.atten(decoder_output, _union_context_simi,
                                                                      _union_context_simi_lens)


                # encoder_copy_outputs = torch.cat([context_simi_vector,context_desc_vector],dim=2).transpose(0,1)
                encoder_copy_outputs = ((context_simi_vector + context_extra_review_vector) / 2).transpose(0, 1)
                # batch_size = encoder_copy_outputs.size(1)
                copy_lens = torch.LongTensor([1] * batch_size).to(device)
                attention_copy_weights, context_copy_vector = self.atten(decoder_output, encoder_copy_outputs,
                                                                         copy_lens)

                # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_simi_input = torch.cat([context_simi_vector, decoder_output], -1)
                concat_extra_review_input = torch.cat([context_extra_review_vector, decoder_output], -1)


                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_simi_output = torch.tanh(self.W_c(concat_simi_input))
                concat_extra_review_output = torch.tanh(self.W_c(concat_extra_review_input))
                # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)

                gama = attention_copy_weights
                gama_ = torch.ones(batch_size, 1, 1, device=device) - gama

                # concat_copy_output = torch.bmm(gama, self.W_s(concat_simi_output)) + torch.bmm(gama_, self.W_s(concat_desc_output))

                p_gen = torch.sigmoid(
                    self.W_H(context_copy_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))

                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                # output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, concat_copy_output)
                concat_simi_dist = torch.bmm(p_gen_, torch.bmm(gama, self.W_s(concat_simi_output)))
                concat_extra_review_dist = torch.bmm(p_gen_, torch.bmm(gama_, self.W_s(concat_extra_review_output)))

                review_dist = torch.bmm(p_gen, self.W_s(concat_src_output))
                output = concat_simi_dist + concat_extra_review_dist + review_dist
            elif runfig.mode == 7:
                attention_src_weights, context_src_vector = self.atten(decoder_output, encoder_src_outputs, src_lens)
                attention_extra_review_weights, context_extra_review_vector = self.atten(decoder_output,
                                                                                         encoder_extra_review_outputs,
                                                                                         extra_review_lens)
                attention_desc_weights, context_desc_vector = self.atten(decoder_output, encoder_desc_outputs,
                                                                         desc_lens)

                # encoder_copy_outputs = torch.cat([context_simi_vector,context_desc_vector],dim=2).transpose(0,1)
                encoder_copy_outputs = ((context_desc_vector + context_extra_review_vector) / 2).transpose(0, 1)
                # batch_size = encoder_copy_outputs.size(1)
                copy_lens = torch.LongTensor([1] * batch_size).to(device)
                attention_copy_weights, context_copy_vector = self.atten(decoder_output, encoder_copy_outputs,
                                                                         copy_lens)

                # concat_input: (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size)
                concat_src_input = torch.cat([context_src_vector, decoder_output], -1)
                concat_desc_input = torch.cat([context_desc_vector, decoder_output], -1)
                concat_extra_review_input = torch.cat([context_extra_review_vector, decoder_output], -1)

                concat_src_output = torch.tanh(self.W_c(concat_src_input))
                concat_desc_output = torch.tanh(self.W_c(concat_desc_input))
                concat_extra_review_output = torch.tanh(self.W_c(concat_extra_review_input))
                # (batch_size, seq_len=1, encoder_hidden_size * num_directions + decoder_hidden_size) => (batch_size, seq_len=1, decoder_hidden_size)

                gama = attention_copy_weights
                gama_ = torch.ones(batch_size, 1, 1, device=device) - gama

                # concat_copy_output = torch.bmm(gama, self.W_s(concat_simi_output)) + torch.bmm(gama_, self.W_s(concat_desc_output))

                p_gen = torch.sigmoid(
                    self.W_H(context_copy_vector) + self.W_S(decoder_output) + self.W_X(emb.transpose(0, 1)))

                p_gen_ = torch.ones(batch_size, 1, 1, device=device) - p_gen

                # output = torch.bmm(p_gen, self.W_s(concat_src_output)) + torch.bmm(p_gen_, concat_copy_output)
                concat_desc_dist = torch.bmm(p_gen_, torch.bmm(gama, self.W_s(concat_desc_output)))
                concat_extra_review_dist = torch.bmm(p_gen_, torch.bmm(gama_, self.W_s(concat_extra_review_output)))

                review_dist = torch.bmm(p_gen, self.W_s(concat_src_output))
                output = concat_desc_dist + concat_extra_review_dist + review_dist
            else:
                raise Exception('set mode with wrong value')

        else:
            concat_output = decoder_output
            output = self.W_s(concat_output)

        # (batch_size, seq_len=1, decoder_hidden_size) => (batch_size, seq_len=1, vocab_size)

        # Prepare returns:
        # (batch_size, seq_len=1, vocab_size) => (batch_size, vocab_size)
        output = output.squeeze(1)

        del src_lens

        return output, decoder_hidden, attention_src_weights, attention_simi_weights, attention_desc_weights
