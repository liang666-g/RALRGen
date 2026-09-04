from argparse import Namespace
file_space = Namespace(
    test_file=r'./data/rrgen_test_data.txt',
    train_file=r'./data/rrgen_train_data.txt',
    valid_file=r'./data/rrgen_valid_data.txt',
    desc_file=r'./data/External_info.txt',
    desc_file_unprocessed=r'./data/External_info.xlsx',
    test_simi_file=r"./data/test_data_tfidf.txt",
    valid_simi_file=r"./data/valid_data_tfidf.txt",
    train_simi_file=r"./data/train_data_tfidf.txt",
    init_filename=r"./data/glove.twitter.27B.100d.txt"
        )
special_identifier = Namespace(
    PAD=0,
    BOS=1,
    EOS=2,
    UNK=3
        )
runfig = Namespace(
    mode=1,  # response and desc; #2 response; #3 desc; #4 review
    tfidf_N=3,
    cos_mode=1,  # tfidf; 2# glove;3#bert
    init_mode=1,  # glove, 2# bert
    additional_review=False,
    pred_test=False,
    max_vocab_size=10000,
    word_vec_size=100,
    batch_size=32,
    num_epochs=3,
    print_every_step=200,
    save_every_step=1000,
    pretrained_embeddings=True,
    fixed_embeddings=True,
    LOAD_CHECKPOINT=False,
    outtext_fp='./data/evaluation/',
    dist_res_fp='./data/dist/'
        )
encoder_config = Namespace(
    run_type='GRU',
    hidden_size=200,
    num_layers=1,
    dropout=0.0,
    bidirectional=True,
    learning_rate=0.001,
    weight_decay=1e-5,
    max_seq_len=100,
        )
decoder_config = Namespace(
    attention=True,
    dropout=0.1,
    learning_rate=0.001,
    weight_decay=1e-5,
    max_seq_len=100,
    max_grad_norm=2
        )
