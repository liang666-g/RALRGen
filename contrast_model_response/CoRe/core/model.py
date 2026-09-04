import time
import torch
from configuration import runfig, encoder_config, decoder_config
from torch.utils.data import DataLoader
from configuration import file_space
from configuration import special_identifier as si
from data_factory import Dataset, collate_fn
from embedding import Embedding
from encoder import EncoderRNN
from torch import optim
from util import USE_CUDA, get_gpu_memory_usage
from trainer import train
from tqdm import tqdm
from decoder import LuongAttnDecoderRNN
import numpy as np
from checkpoint import save_checkpoint
import os
from tester import _load_test_data, _valid_test
from checkpoint import get_checkpoint
from datetime import datetime
from multiprocessing import freeze_support


def main():
    start_time = time.time()
    max_vocab_size = runfig.max_vocab_size  # 10000
    embedding_size = runfig.word_vec_size  # 25
    batch_size = runfig.batch_size  # 32

    train_dataset = Dataset(file_space.train_file, file_space.desc_file, file_space.train_simi_file, max_vocab_size)

    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            collate_fn=collate_fn)

    input_vocab_size = len(train_dataset.input_vocab.token2id)
    output_vocab_size = len(train_dataset.output_vocab.token2id)

    input_embedding = torch.nn.Embedding(input_vocab_size, embedding_size, padding_idx=si.PAD)
    output_embedding = torch.nn.Embedding(output_vocab_size, embedding_size, padding_idx=si.PAD)

    print("Initialize encoder decoder")

    encoder = EncoderRNN(embedding=input_embedding,
                         rnn_type=encoder_config.run_type,
                         hidden_size=encoder_config.hidden_size,
                         num_layers=encoder_config.num_layers,
                         dropout=encoder_config.dropout,
                         bidirectional=encoder_config.bidirectional)
    decoder = LuongAttnDecoderRNN(encoder, embedding=output_embedding,
                                  attention=decoder_config.attention,
                                  dropout=decoder_config.dropout,
                                  )

    print('emb start')
    if runfig.pretrained_embeddings:
        train_embedding = Embedding(filename=file_space.init_filename, embedding_size=embedding_size).load_embedding(
            train_dataset.input_vocab)
        target_embedding = Embedding(filename=file_space.init_filename, embedding_size=embedding_size).load_embedding(
            train_dataset.output_vocab)
        encoder.embedding.weight.data.copy_(train_embedding)
        decoder.embedding.weight.data.copy_(target_embedding)
        if runfig.fixed_embeddings:
            encoder.embedding.weight.requires_grad = False
            decoder.embedding.weight.requires_grad = False
        else:
            decoder.embedding.weight.requires_grad = True
    else:
        raise Exception('Please implement word embedding in a custom way')
    print('emb end')

    if runfig.LOAD_CHECKPOINT:
        target_path = 'wc_100_hs_200_ln_1_dp_0.0/2026-04-01_22-57-23_acc_95.88_loss_38.50_step_7000.pt'
        checkpoint = get_checkpoint(target_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    FINE_TUNE = True
    if FINE_TUNE:
        encoder.embedding.weight.requires_grad = True

    print('=' * 100)
    print('Model log:\n')
    print(encoder)
    print(decoder)
    print('- Encoder input embedding requires_grad={}'.format(encoder.embedding.weight.requires_grad))
    print('- Decoder input embedding requires_grad={}'.format(decoder.embedding.weight.requires_grad))
    print('- Decoder output embedding requires_grad={}'.format(decoder.W_s.weight.requires_grad))
    print('=' * 100 + '\n')
    # Initialize optimizers (we can experiment different learning rates)
    encoder_optim = optim.Adam([p for p in encoder.parameters() if p.requires_grad], lr=encoder_config.learning_rate,
                               weight_decay=encoder_config.weight_decay)
    decoder_optim = optim.Adam([p for p in decoder.parameters() if p.requires_grad], lr=decoder_config.learning_rate,
                               weight_decay=decoder_config.weight_decay)
    print("Successfully initialized encoder decoder")

    if not runfig.pred_test:

        # Start training

        # from tensorboardX import SummaryWriter
        # --------------------------
        # Configure tensorboard
        # --------------------------
        print("Start training......")
        model_name = 'deepcopy'
        experiment_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        tensorboard_log_dir = './tensorboard-logs/{}/'.format(experiment_name)
        # writer = SummaryWriter(tensorboard_log_dir)

        # --------------------------
        # Configure training
        # --------------------------
        num_epochs = runfig.num_epochs
        print_every_step = runfig.print_every_step
        save_every_step = runfig.save_every_step
        # For saving checkpoint and tensorboard
        global_step = 0

        # --------------------------
        # Load test data
        # --------------------------
        test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts = _load_test_data(
            file_space.test_file,
            file_space.desc_file,
            file_space.test_simi_file)

        # --------------------------
        # Start training
        # --------------------------
        total_loss = 0
        total_corrects = 0
        total_words = 0
        prev_gpu_memory_usage = 0
        max_accuracy = 0.0
        max_bleu = 0.0
        save_count = 0  # Save attention weights to file

        for epoch in range(num_epochs):
            print('mode:' + str(runfig.mode))
            for batch_id, batch_data in tqdm(enumerate(train_iter)):
                src_seqs, simi_seqs, desc_seqs, tgt_seqs, extra_review_seqs, src_lens, simi_lens, desc_lens, tgt_lens, extra_review_lens = batch_data
                if runfig.tfidf_N == 1:
                    max_seq_len = max(src_lens + simi_lens + tgt_lens + desc_lens)
                else:
                    cur = []
                    cur += src_lens
                    for cur_simi_len in simi_lens:
                        cur += cur_simi_len
                    cur += tgt_lens
                    cur += desc_lens
                    max_seq_len = max(cur)
                if max_seq_len > encoder_config.max_seq_len:
                    print('[!] Ignore batch: sequence length={} > max sequence length={}'.format(max_seq_len,
                                                                                                 encoder_config.max_seq_len))
                    continue
                loss, pred_seqs, num_corrects, attention_src_weights, attention_simi_weights, attention_desc_weights, num_words, \
                    encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm \
                    = train(src_seqs, simi_seqs, desc_seqs, tgt_seqs, extra_review_seqs, src_lens, simi_lens, desc_lens,
                            tgt_lens, extra_review_lens, encoder,
                            decoder, encoder_optim, decoder_optim)

                # Statistics.
                global_step += 1
                total_loss += loss
                total_corrects += num_corrects
                total_words += num_words
                total_accuracy = 100 * np.divide(total_corrects, float(total_words))

                # Save checkpoint.
                if global_step % save_every_step == 0:
                    if total_accuracy > max_accuracy:
                        max_accuracy = total_accuracy
                        checkpoint_path = save_checkpoint(experiment_name, encoder, decoder, encoder_optim,
                                                          decoder_optim,
                                                          total_accuracy, total_loss, global_step)

                        print('=' * 100)
                        print('Save checkpoint to "{}".'.format(checkpoint_path))
                        print('=' * 100 + '\n')

                if global_step % print_every_step == 0:
                    curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
                    diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
                    prev_gpu_memory_usage = curr_gpu_memory_usage

                    print('=' * 100)
                    print('Training log:')
                    print('- Epoch: {}/{}'.format(epoch, num_epochs))
                    print('- Global step: {}'.format(global_step))
                    print('- Total loss: {}'.format(total_loss))
                    print('- Total corrects: {}'.format(total_corrects))
                    print('- Total words: {}'.format(total_words))
                    print('- Total accuracy: {}'.format(total_accuracy))
                    print('- Current GPU memory usage: {}'.format(curr_gpu_memory_usage))
                    print('- Diff GPU memory usage: {}'.format(diff_gpu_memory_usage))
                    print('=' * 100 + '\n')

                    total_loss = 0
                    total_corrects = 0
                    total_words = 0

                    if total_accuracy > max_accuracy:
                        max_accuracy = total_accuracy
                        print("max accuracy is ", max_accuracy)

                del src_seqs, simi_seqs, desc_seqs, tgt_seqs, src_lens, simi_lens, desc_lens, tgt_lens, \
                    loss, pred_seqs, num_corrects, attention_src_weights, attention_simi_weights, attention_desc_weights, num_words, \
                    encoder_grad_norm, decoder_grad_norm, clipped_encoder_grad_norm, clipped_decoder_grad_norm

            test_start_time = time.time()
            bleu_score, pls, rouge_score, meteor_score, out_texts, attention_src_weights, attention_simi_weights, attention_desc_weights = _valid_test(
                test_src_texts, test_tgt_texts, test_desc_texts,
                test_simi_texts, test_extra_review_texts, train_dataset, encoder, decoder,
                decoder_config.max_seq_len)
            print("Test time cost is ", (time.time() - test_start_time) / 3600, "hrs.")
            max_bleu = bleu_score
            # '''
            # with open(os.path.join(runfig.dist_res_fp, 'attention_src_weights' + str(max_bleu) + '_' + str(save_count)),
            #           'wb') as f:
            #     pickle.dump(attention_src_weights, f)
            # if (runfig.mode == 1 or runfig.mode == 2):
            #     if (runfig.tfidf_N == 1):
            #         with open(
            #                 os.path.join(runfig.dist_res_fp,
            #                              'attention_simi_weights' + str(max_bleu) + '_' + str(save_count)),
            #                 'wb') as f:
            #             pickle.dump(attention_simi_weights, f)
            #     else:
            #         with open(os.path.join(runfig.dist_res_fp,
            #                                'attention_simi_weights$1$' + str(max_bleu) + '_' + str(save_count)), 'wb') as f:
            #             pickle.dump(attention_simi_weights[0], f)
            #         with open(os.path.join(runfig.dist_res_fp,
            #                                'attention_simi_weights$2$' + str(max_bleu) + '_' + str(save_count)), 'wb') as f:
            #             pickle.dump(attention_simi_weights[1], f)
            # if (runfig.mode == 1 or runfig.mode == 3):
            #     with open(
            #             os.path.join(runfig.dist_res_fp, 'attention_desc_weights' + str(max_bleu) + '_' + str(save_count)),
            #             'wb') as f:
            #         pickle.dump(attention_desc_weights, f)
            # '''
            # for id_att, weights in enumerate(attention_test_ws):
            #     pickle.dump(test_src_texts[id_att]+'-[split]-'+','.join([str(w) for w in weights]), att_fw))

            filename = (
                f"init_{runfig.init_mode}_"
                f"cos_{runfig.cos_mode}_"
                f"vocab_{runfig.max_vocab_size}_"
                f"embed_{runfig.word_vec_size}_"
                f"mode_{runfig.mode}_"
                f"hidden_{encoder_config.hidden_size}_"
                f"layers_{encoder_config.num_layers}_"
                f"input_len_{encoder_config.max_seq_len}_"
                f"output_len_{decoder_config.max_seq_len}"
            )
            filename = os.path.join(runfig.outtext_fp, filename)
            folder = os.path.exists(filename)
            if not folder:
                os.makedirs(filename)

            bleu_str = f"{max_bleu:.4f}"

            text_fw = open(os.path.join(filename, f"{bleu_str}_bleu_{save_count}"), 'w', encoding='utf-8')
            for id_text, text in enumerate(out_texts):
                text_fw.write(test_src_texts[id_text] + "-[split]-" + text + '\n')
            text_fw.close()
            # '''
            # with open(os.path.join(filename+'/', str(max_bleu) + '_pgen_' + str(save_count)), 'w') as f:
            #     for i in range(len(pgens)):
            #         f.write(str(pgens[i]) + '\n')
            # with open(os.path.join(filename+'/', str(max_bleu) + '_gama_' + str(save_count)), 'w') as f:
            #     for i in range(len(gamas)):
            #         f.write(str(gamas[i]) + '\n')
            # '''
            with open(os.path.join(filename, f"{bleu_str}_rouge_{save_count}"), 'w', encoding='utf-8') as f:
                f.write(str(rouge_score))
            with open(os.path.join(filename, f"{bleu_str}_meteor_{save_count}"), 'w', encoding='utf-8') as f:
                f.write(str(meteor_score))
            with open(os.path.join(filename, f"{bleu_str}_pls_{save_count}"), 'w', encoding='utf-8') as f:
                for p in pls:
                    f.write(str(p) + '\n')
            save_count += 1
            print("max blue score is ", max_bleu, "pls is ", pls)
            print("Current period is ", (time.time() - start_time) / 3600, "hrs.")
    else:
        # Start testing

        # from tensorboardX import SummaryWriter
        # --------------------------
        # Configure tensorboard
        # --------------------------
        print("Start testing......")
        test_src_texts, test_tgt_texts, test_desc_texts, test_simi_texts, test_extra_review_texts = _load_test_data(
            file_space.test_file,
            file_space.desc_file,
            file_space.test_simi_file)

        test_start_time = time.time()
        bleu_score, pls, rouge_score, meteor_score, out_texts, attention_src_weights, attention_simi_weights, attention_desc_weights = _valid_test(
            test_src_texts, test_tgt_texts, test_desc_texts,
            test_simi_texts, test_extra_review_texts, train_dataset, encoder, decoder,
            decoder_config.max_seq_len)
        print("Test time cost is ", (time.time() - test_start_time) / 3600, "hrs.")
        max_bleu = bleu_score

        filename = (
            'init_' + str(runfig.init_mode) + \
            'cos_' + str(runfig.cos_mode) + \
            'vocab_' + str(runfig.max_vocab_size) + \
            'embed_' + str(runfig.word_vec_size) + \
            'mode_' + str(runfig.mode) + \
            'hidden_' + str(encoder_config.hidden_size) + \
            'layers_' + str(encoder_config.num_layers)
        )
        filename = os.path.join(runfig.outtext_fp, filename)
        folder = os.path.exists(filename)
        if not folder:
            os.makedirs(filename)

        bleu_str = f"{max_bleu:.4f}"

        text_fw = open(os.path.join(filename, f"{bleu_str}_bleu"), 'w', encoding='utf-8')
        for id_text, text in enumerate(out_texts):
            text_fw.write(test_src_texts[id_text] + "-[split]-" + text + '\n')
        text_fw.close()
        # '''
        # with open(os.path.join(filename + '/', str(max_bleu) + '_pgen_' + str(save_count)), 'w') as f:
        #     for i in range(len(pgens)):
        #         f.write(str(pgens[i]) + '\n')
        # with open(os.path.join(filename + '/', str(max_bleu) + '_gama_' + str(save_count)), 'w') as f:
        #     for i in range(len(gamas)):
        #         f.write(str(gamas[i]) + '\n')
        # '''
        with open(os.path.join(filename, f"{bleu_str}_rouge"), 'w', encoding='utf-8') as f:
            f.write(str(rouge_score))
        with open(os.path.join(filename, f"{bleu_str}_meteor"), 'w', encoding='utf-8') as f:
            f.write(str(meteor_score))
        with open(os.path.join(filename, f"{bleu_str}_pls"), 'w', encoding='utf-8') as f:
            for p in pls:
                f.write(str(p) + '\n')
        print("max blue score is ", max_bleu, "pls is ", pls)
        print("Current period is ", (time.time() - start_time) / 3600, "hrs.")

    '''
    else:
        # --------------------------
        ### Start translation
        # --------------------------
        test_src_texts = []
        test_tgt_texts = []
        test_desc_texts = []
        test_simi = []
        test_simi_texts = []
        desc_map = {}
        with open(file_space.test_tfidf_file) as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split()
                test_simi.append(terms[1])
        with open(file_space.desc_file) as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split('-[split]-')
                desc_map[terms[0]] = [token for token in terms[1].split()]
        with open(file_space.test_file) as f:
            for sent in tqdm(f.readlines()):
                terms = sent.split('-[split]-')
                if len(terms) < 8:  # check term length
                    continue
                app_id = terms[0]
                test_desc_texts.append(desc_map[app_id])
                test_src_texts.append(terms[4])
                test_tgt_texts.append(terms[5])
        for n in test_simi:
            test_simi_texts.append(test_tgt_texts[n])
    
        # test_src_texts = [line.split('-[split]-')[4] for line in test_fr.readlines()]
        # test_tgt_texts = [ for line in test_fr.readlines()]
        print(len(test_src_texts), len(test_tgt_texts))
        out_texts = []
        for idx, src_text in tqdm(enumerate(test_src_texts)):
            _, out_text, _ = translate(src_text.strip(), test_desc_texts[idx], test_simi_texts[idx], train_dataset, encoder,
                                       decoder, max_seq_len=decoder_config.max_seq_len)
            out_texts.append(src_text.strip() + '-[split]-' + test_tgt_texts[idx].strip() + '-[split]-' + out_text + '\n')
            # if idx%100 == 0:
            #     print("already translate to %d th sentence." % idx)
            # print("> %s" % src_text)
            # print("= %s" % test_tgt_texts[idx])
            # print("< %s" % out_text)
            # if idx == 10:
            #     break
    
        from parameter import checkpoint_path
    
        fw_name = checkpoint_path.split('/')
        dir_name = os.path.join('/home/cygao/workspace/data', 'pred', fw_name[-2])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(os.path.join(dir_name, fw_name[-1].strip('.pt')), 'w') as f:
            f.writelines(out_texts)
    '''


if __name__ == '__main__':
    freeze_support()
    main()
