import torch
import torch.nn as nn
from torch.autograd import Variable
from configuration import special_identifier as si, decoder_config, runfig
import torch.nn.functional as F
from util import detach_hidden, masked_cross_entropy, USE_CUDA


def compute_grad_norm(parameters, norm_type=2):
    """ Ref: http://pytorch.org/docs/0.3.0/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    return total_norm


# src_sents, tgt_sents, src_seqs, tgt_seqs, src_lens, tgt_lens, encoder, decoder, encoder_optim, decoder_optim
# src_sents, tgt_sents, src_seqs, tgt_input_seqs,tgt_output_seqs,desc_seqs,src_lens,tgt_input_lens,tgt_output_lens,desc_lens, encoder, decoder, encoder_optim, decoder_optim
def train(src_seqs, simi_seqs, desc_seqs, tgt_seqs, extra_review_seqs, src_lens, simi_lens, desc_lens, tgt_lens,
          extra_review_lens, encoder, decoder,
          encoder_optim, decoder_optim):
    # -------------------------------------
    # Prepare input and output placeholders
    # -------------------------------------
    # Last batch might not have the same size as we set to the `batch_size`
    batch_size = src_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))

    if runfig.tfidf_N == 1:
        batch_size = simi_seqs.size(1)
        assert (batch_size == tgt_seqs.size(1))
    else:
        for cur_simi_seq in simi_seqs:
            batch_size = cur_simi_seq.size(1)
            assert (batch_size == tgt_seqs.size(1))

    assert (batch_size == tgt_seqs.size(1))

    batch_size = desc_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))

    batch_size = extra_review_seqs.size(1)
    assert (batch_size == tgt_seqs.size(1))
    # Pack tensors to variables for neural network inputs (in order to autograd)
    # pay attention! simi_seqs and desc_seqs are not converted to variable
    src_seqs = Variable(src_seqs)
    tgt_seqs = Variable(tgt_seqs)
    src_lens = Variable(torch.LongTensor(src_lens))
    tgt_lens = Variable(torch.LongTensor(tgt_lens))

    if runfig.tfidf_N == 1:
        simi_lens = torch.LongTensor(simi_lens)
    else:
        for i in range(len(simi_lens)):
            simi_lens[i] = torch.LongTensor(simi_lens[i])

    desc_lens = torch.LongTensor(desc_lens)

    extra_review_lens = torch.LongTensor(extra_review_lens)

    # Decoder's input
    input_seq = Variable(torch.LongTensor([si.BOS] * batch_size))
    # Decoder's output sequence length = max target sequence length of current batch.
    max_tgt_len = tgt_lens.data.max()
    # Store all decoder's outputs.
    # **CRUTIAL**
    # Don't set:
    # >> decoder_outputs = Variable(torch.zeros(max_tgt_len, batch_size, decoder.vocab_size))
    # Varying tensor size could cause GPU allocate a new memory causing OOM,
    # so we intialize tensor with fixed size instead:
    # `opts.max_seq_len` is a fixed number, unlike `max_tgt_len` always varys.
    decoder_outputs = Variable(torch.zeros(decoder_config.max_seq_len, batch_size, decoder.vocab_size))

    if USE_CUDA:
        if runfig.tfidf_N == 1:
            simi_seqs = simi_seqs.cuda()
            simi_lens = simi_lens.cuda()
        else:
            for i in range(len(simi_seqs)):
                simi_seqs[i] = simi_seqs[i].cuda()
                simi_lens[i] = simi_lens[i].cuda()

        src_seqs = src_seqs.cuda()
        tgt_seqs = tgt_seqs.cuda()
        desc_seqs = desc_seqs.cuda()
        extra_review_seqs = extra_review_seqs.cuda()

        src_lens = src_lens.cuda()
        tgt_lens = tgt_lens.cuda()
        desc_lens = desc_lens.cuda()
        extra_review_lens = extra_review_lens.cuda()

        input_seq = input_seq.cuda()
        decoder_outputs = decoder_outputs.cuda()

    # -------------------------------------
    # Training mode (enable dropout)
    # -------------------------------------
    encoder.train()
    decoder.train()
    # -------------------------------------
    # Zero gradients, since optimizers will accumulate gradients for every backward.
    # -------------------------------------
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()
    # -------------------------------------
    # Forward encoder
    # -------------------------------------
    encoder_src_outputs, encoder_src_hidden = encoder(src_seqs, src_lens.data.tolist())
    if runfig.tfidf_N == 1:
        encoder_simi_outputs, encoder_simi_hidden = encoder(simi_seqs, simi_lens.data.tolist())
    else:
        encoder_simi_outputs = []
        encoder_simi_hidden = []
        for i in range(len(simi_seqs)):
            cur_encoder_simi_outputs, cur_encoder_simi_hidden = encoder(simi_seqs[i], simi_lens[i].data.tolist())
            encoder_simi_outputs.append(cur_encoder_simi_outputs)
            encoder_simi_hidden.append(cur_encoder_simi_hidden)

    encoder_desc_outputs, encoder_desc_hidden = encoder(desc_seqs, desc_lens.data.tolist())

    encoder_extra_review_outputs, encoder_extra_review_hidden = encoder(extra_review_seqs,
                                                                        extra_review_lens.data.tolist())

    # Initialize decoder's hidden state as encoder's last hidden state.
    decoder_hidden = encoder_src_hidden
    if USE_CUDA:
        decoder_hidden = decoder_hidden.cuda()

    # Run through decoder one time step at a time.
    for t in range(max_tgt_len):
        # decoder returns:
        # - decoder_output   : (batch_size, vocab_size)
        # - decoder_hidden   : (num_layers, batch_size, hidden_size)
        # - attention_weights: (batch_size, max_src_len)
        decoder_output, decoder_hidden, attention_src_weights, attention_simi_weights, attention_desc_weights = decoder(
            input_seq, decoder_hidden, encoder_src_outputs, encoder_simi_outputs,
            encoder_desc_outputs, encoder_extra_review_outputs, src_lens, simi_lens, desc_lens, extra_review_lens)

        # Store decoder outputs.
        decoder_outputs[t] = decoder_output

        # Next input is current target
        input_seq = tgt_seqs[t]

        # Detach hidden state:
        detach_hidden(decoder_hidden)

    # -------------------------------------
    # Compute loss
    # -------------------------------------
    loss, pred_seqs, num_corrects, num_words = masked_cross_entropy(
        decoder_outputs[:max_tgt_len].transpose(0, 1).contiguous(),
        tgt_seqs.transpose(0, 1).contiguous(),
        tgt_lens
    )

    pred_seqs = pred_seqs[:max_tgt_len]

    # -------------------------------------
    # Backward and optimize
    # -------------------------------------
    # Backward to get gradients w.r.t parameters in model.
    loss.backward()

    # Clip gradients
    encoder_grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), decoder_config.max_grad_norm)
    decoder_grad_norm = nn.utils.clip_grad_norm_(decoder.parameters(), decoder_config.max_grad_norm)
    clipped_encoder_grad_norm = compute_grad_norm(encoder.parameters())
    clipped_decoder_grad_norm = compute_grad_norm(decoder.parameters())

    # Update parameters with optimizers
    encoder_optim.step()
    decoder_optim.step()

    return loss.data.item(), pred_seqs, num_corrects, attention_src_weights, attention_simi_weights, attention_desc_weights, num_words, \
        encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm
