class Config:
    data_name = 'nyt'
    num_epochs = 100
    batch_size = 500
    train_embeddings=0
    max_gradient_norm=-1
    hidden_state_size=100
    embedding_size=100
    learning_rate = 0.001
    data_dir="data/nyt_seq/"
    vocab_path=data_dir+"vocab.dat"
    embed_path=data_dir+"glove.trimmed.100.npz.npy"
    dropout_val=0.5
    
    # train_dir='pkl/models_nyt_multi_head'
    # train_dir='pkl/models_nyt_multi_head_dot'
    # train_dir='pkl/models_nyt_multi_head_hs'
    # train_dir='pkl/models_nyt_multi_head_hs2'

    # train_dir='pkl/models_nyt_multi_head_dot_simple'
    # train_dir='pkl/models_nyt_multi_head_add'





    max_length = 101
    log_file = 'log/log_nyt.txt'
    match_file = data_dir+'test_match_output.json'
    dev_match_file = data_dir+'dev_match_output.json'
    # result_file = 'new_result/result_nyt_origin_rc.txt'
    # result_file = 'new_result/result_nyt_match_gen.txt'
    # result_file = 'new_result/result_nyt_multi_head.txt'
    # result_file = 'new_result/result_nyt_multi_head_dot.txt'
    # result_file = 'new_result/result_nyt_multi_head_dot_simple.txt'
    result_file = 'new_result/result_nyt_multi_head_add.txt'





    words_id2vector_filename=data_dir+'words_id2vector_new.json'
    origin_train_data_file=data_dir+'train_new.json'
    origin_test_data_file=data_dir+'test_new.json'
    words_number= 90761
    embedding_dim = 100
    # batch_size = 200
    hidden_size = 128
    num_classes = 2
    # learning_rate = 0.001
    relation_num = 25

    max_decode_size = 16

    label_weight =  5
    head_num = 4
    

    def get_paths(data_dir,mode):
        question = data_dir+"%s.ids.question" %mode
        context = data_dir+"%s.ids.context" %mode
        answer = data_dir+"%s.span" %mode
        cnn_output=data_dir+"%s_cnn_output.span" %mode
        cnn_list=data_dir+"%s_cnn_output.json" %mode

        return question, context, answer ,cnn_output, cnn_list
    
    question_train, context_train, answer_train,cnn_output_train,cnn_list_train = get_paths(data_dir,"train")
    question_dev ,context_dev ,answer_dev ,cnn_output_dev,cnn_list_dev= get_paths(data_dir,"dev")
    question_test ,context_test ,answer_test ,cnn_output_test,cnn_list_test= get_paths(data_dir,"test")
                                           