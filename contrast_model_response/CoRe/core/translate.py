import torch
from torch.autograd import Variable
from configuration import special_identifier as si
from util import USE_CUDA, detach_hidden
import pandas as pd
from configuration import runfig

def translate(src_text, desc_text, simi_text, extra_review_text, train_dataset, encoder, decoder, max_seq_len, replace_unk=True):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Like dataset's `__getitem__()` and dataloader's `collate_fn()`.
    src_sent = src_text.split()
    desc_sent = desc_text
    extra_review_sent = extra_review_text.split()
    if(runfig.tfidf_N == 1):
        simi_sent = simi_text.strip().split()
    else:
        simi_sent = []
        for cur_simi_text in simi_text:
            simi_sent.append(cur_simi_text.strip().split())

    src_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=src_sent, token2id=train_dataset.input_vocab.token2id,
                                                          append_BOS=False, append_EOS=True)]).transpose(0, 1)
    extra_review_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=extra_review_sent,
                                                           token2id=train_dataset.input_vocab.token2id,
                                                           append_BOS=False, append_EOS=True)]).transpose(0, 1)
    desc_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=desc_sent,
                                                           token2id=train_dataset.input_vocab.token2id,
                                                           append_BOS=False, append_EOS=True)]).transpose(0, 1)
    if(runfig.tfidf_N == 1):
        simi_seqs = torch.LongTensor([train_dataset.tokens2ids(tokens=simi_sent,
                                                               token2id=train_dataset.input_vocab.token2id,
                                                               append_BOS=False, append_EOS=True)]).transpose(0, 1)
    else:
        simi_seqs = []
        for cur_simi_sent in simi_sent:
            simi_seqs.append(torch.LongTensor([train_dataset.tokens2ids(tokens=cur_simi_sent,
                                                               token2id=train_dataset.input_vocab.token2id,
                                                               append_BOS=False, append_EOS=True)]).transpose(0, 1))

    src_lens = [len(src_seqs)]
    desc_lens = [len(desc_seqs)]
    extra_review_lens = [len(extra_review_seqs)]

    if(runfig.tfidf_N == 1):
        simi_lens = [len(simi_seqs)]
    else:
        simi_lens = []
        for cur_simi_seq in simi_seqs:
            simi_lens.append([len(cur_simi_seq)])

    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)

    # Pack tensors to variables for neural network inputs (in order to autograd)
    src_seqs = Variable(src_seqs)
    src_lens = Variable(torch.LongTensor(src_lens))
    if(runfig.tfidf_N == 1):
        simi_lens = torch.LongTensor(simi_lens)
    else:
        for i in range(len(simi_lens)):
            simi_lens[i] = torch.LongTensor(simi_lens[i])
    desc_lens = torch.LongTensor(desc_lens)
    extra_review_lens = torch.LongTensor(extra_review_lens)
    # Decoder's input
    input_seq = Variable(torch.LongTensor([si.BOS] * batch_size))

    # Store output words and attention states
    out_sent = []

    attention_src_weights = torch.zeros(max_seq_len, len(src_seqs))
    attention_simi_weights = None
    if(runfig.tfidf_N == 1):
        attention_simi_weights = torch.zeros(max_seq_len, len(simi_seqs))
    else:
        attention_simi_weights1 = torch.zeros(max_seq_len, len(simi_seqs[0]))
        attention_simi_weights2 = torch.zeros(max_seq_len, len(simi_seqs[1]))
    attention_desc_weights = torch.zeros(max_seq_len, len(desc_seqs))
    # Move variables from CPU to GPU.
    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        src_lens = src_lens.cuda()

        if(runfig.tfidf_N == 1):
            simi_lens = simi_lens.cuda()
            simi_seqs = simi_seqs.cuda()
        else:
            for i in range(len(simi_lens)):
                simi_lens[i] = simi_lens[i].cuda()
                simi_seqs[i] = simi_seqs[i].cuda()

        desc_seqs = desc_seqs.cuda()
        desc_lens = desc_lens.cuda()

        extra_review_seqs = extra_review_seqs.cuda()
        extra_review_lens = extra_review_lens.cuda()

        input_seq = input_seq.cuda()

    # -------------------------------------
    # Evaluation mode (disable dropout)
    # -------------------------------------
    encoder.eval()
    decoder.eval()

    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_src_outputs, encoder_src_hidden = encoder(src_seqs, src_lens.data.tolist())
    encoder_extra_review_outputs, encoder_extra_review_hidden = encoder(extra_review_seqs, extra_review_lens.data.tolist())
    if(runfig.mode== 1 or runfig.mode==2 or runfig.mode==6):
        if(runfig.tfidf_N == 1):
            encoder_simi_outputs, encoder_simi_hidden = encoder(simi_seqs, simi_lens.data.tolist())
        else:
            encoder_simi_outputs = []
            encoder_simi_hidden = []
            for i in range(len(simi_seqs)):
                cur_encoder_simi_outputs, cur_encoder_simi_hidden = encoder(simi_seqs[i], simi_lens[i].data.tolist())
                encoder_simi_outputs.append(cur_encoder_simi_outputs)
                encoder_simi_hidden.append(cur_encoder_simi_hidden)
    else:
        encoder_simi_outputs = None
    if (runfig.mode == 1 or runfig.mode == 3 or runfig.mode==7):
        encoder_desc_outputs, encoder_desc_hidden = encoder(desc_seqs, desc_lens.data.tolist())
    else:
        encoder_desc_outputs = None
    # -------------------------------------
    # Forward decoder
    # -------------------------------------
    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_src_hidden
    if USE_CUDA:
        decoder_hidden = decoder_hidden.cuda()
    # Run through decoder one time step at a time.

    for t in range(max_seq_len):

        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden,attention_src_weight,attention_simi_weight,attention_desc_weight= decoder(input_seq, decoder_hidden, encoder_src_outputs, encoder_simi_outputs,
                                                 encoder_desc_outputs,encoder_extra_review_outputs, src_lens, simi_lens, desc_lens,extra_review_lens)

        '''
        if p_gen:
            cur_pgen = p_gen.squeeze(0).cpu().data.numpy().tolist()
            pgens.append(cur_pgen[0][0])
        if gama:
            cur_gama = gama.squeeze(0).cpu().data.numpy().tolist()
            gamas.append(cur_gama[0][0])
        '''
        # Store attention weights.
        # .squeeze(0): remove `batch_size` dimension since batch_size=1
        if runfig.mode==1 or runfig.mode==2 or runfig.mode==3:
            attention_src_weights[t] = attention_src_weight.squeeze(0).cpu().data
        if(runfig.mode==1 or runfig.mode==3):
            attention_desc_weights[t] = attention_desc_weight.squeeze(0).cpu().data
        if(runfig.mode==1 or runfig.mode==2):
            if (runfig.tfidf_N == 1):
                attention_simi_weights[t] = attention_simi_weight.squeeze(0).cpu().data
            else:
                attention_simi_weights1[t] = attention_simi_weight[0].squeeze(0).cpu().data
                attention_simi_weights2[t] = attention_simi_weight[1].squeeze(0).cpu().data
        # Choose top word from decoder's output
        prob, token_id = decoder_output.data.topk(1)
        token_id = token_id[0][0].item()  # get value

        if token_id == si.EOS:
            break
        else:
            try:
                token = train_dataset.output_vocab.id2token[token_id]
            except KeyError:
                print('token_id is ', token_id)

            out_sent.append(token)

        # Next input is chosen word
        input_seq = Variable(torch.LongTensor([token_id]))
        if USE_CUDA: input_seq = input_seq.cuda()

        # Repackage hidden state (may not need this, since no BPTT)
        detach_hidden(decoder_hidden)

    attention_src_weights = attention_src_weights[0:len(out_sent)]

    if (runfig.mode == 1 or runfig.mode == 2):
        if (runfig.tfidf_N == 1):
            attention_simi_weights = attention_simi_weights[0:len(out_sent)]
        else:
            attention_simi_weights = [attention_simi_weights1[0:len(out_sent)], attention_simi_weights2[0:len(out_sent)]]

    if (runfig.mode == 1 or runfig.mode == 3):
        attention_desc_weights = attention_desc_weights[0:len(out_sent)]

    src_text = ' '.join([train_dataset.input_vocab.id2token[token_id] for token_id in src_seqs.data.squeeze(1).tolist()])
    out_text = ' '.join(out_sent)


    # all_attention_weights: (out_len, src_len)
    return src_text, out_text,attention_src_weights,\
           attention_simi_weights,attention_desc_weights