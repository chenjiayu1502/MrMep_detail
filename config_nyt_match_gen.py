class Config:
    data_name = 'nyt'
    num_epochs = 100
    batch_size = 500
    train_embeddings=0
    hidden_state_size=100
    learning_rate = 0.001
    data_dir="data/nyt_seq/"
    dropout_val=0.5
    max_length = 101
    match_file = data_dir+'test_match_output.json'
    words_id2vector_filename=data_dir+'words_id2vector_new.json'
    words_number= 90761
    embedding_dim = 100
    num_classes = 2
    relation_num = 25
    max_decode_size = 16
    label_weight =  5

    train_dir='pkl/models_nyt_match_gen'
    log_file = 'log/log_nyt_match_gen.txt'
    result_file = 'result/result_nyt_match_gen.txt'

    use_match=True
    use_multi_head=False
    use_attention_type=0 #0 for linear attention 1 for dot attention


    def get_paths(data_dir,mode):
        question = data_dir+"%s.ids.question" %mode
        context = data_dir+"%s.ids.context" %mode
        answer = data_dir+"%s.span" %mode
        cnn_output=data_dir+"%s_cnn_output.span" %mode
        cnn_list=data_dir+"%s_cnn_output.json" %mode

        return question, context, answer ,cnn_output, cnn_list
    
    question_train, context_train, answer_train,cnn_output_train,cnn_list_train = get_paths(data_dir,"train")
    question_dev ,context_dev ,answer_dev ,cnn_output_dev,cnn_list_dev= get_paths(data_dir,"test")
                                           
