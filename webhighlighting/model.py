from keras.layers import Input, Dropout, Convolution1D, Flatten, Dense, MaxPooling1D, Concatenate, AveragePooling1D, \
    Embedding
from keras.layers.recurrent import LSTM
from keras.models import Model


def create_model_dscnn_document(embedd_mat, max_sentences_per_doc, max_sentence_len, kernel_sizes, filters=100,
                                dropout=0.5, hidden_dims=100):
    """

    :param hidden_dims:
    :param embedd_mat:
    :param max_sentences_per_doc:
    :param max_sentence_len:
    :param filters:
    :param kernel_sizes:
    :param dropout
    :return:
    """
    '''
    sentence modeling 
    '''
    # input (sentence-level)
    sentence_inputs = [Input(shape=(max_sentence_len, ), name="input_" + str(i)) for i in range(max_sentences_per_doc)]

    # embedding
    vocab_sz, embedd_dim = embedd_mat.shape
    shared_embedding = Embedding(vocab_sz, embedd_dim, input_length=max_sentence_len, weights=[embedd_mat],
                                 trainable=True)
    sentence_embeddings = [shared_embedding(sentence_inputs[i]) for i in range(max_sentences_per_doc)]

    # LSTMs and Average Pooling (sentence-level)
    shared_sentence_lstm = LSTM(units=embedd_dim, return_sequences=True, activation='tanh')
    shared_average_pooling = AveragePooling1D(pool_size=max_sentence_len)
    sentence_modeling = [shared_sentence_lstm(sentence_embeddings[i]) for i in range(max_sentences_per_doc)]
    sentence_modeling = [shared_average_pooling(sentence_modeling[i]) for i in range(max_sentences_per_doc)]

    '''
    document modeling
    '''
    doc_modeling = Concatenate(axis=1)(sentence_modeling)
    doc_modeling = LSTM(units=embedd_dim, activation='tanh', return_sequences=True)(doc_modeling)

    conv_blocks = []
    for sz in kernel_sizes:
        conv = Convolution1D(filters=filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1)(doc_modeling)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    doc_modeling = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    doc_modeling = Dropout(dropout)(doc_modeling)
    doc_modeling = Dense(hidden_dims, activation="relu")(doc_modeling)
    # doc_modeling = Activation('softmax')(doc_modeling)

    # model_output = Dense(1, activation="sigmoid")(doc_modeling)
    model_output = Dense(2, activation="softmax", name="prediction")(doc_modeling)

    return Model(inputs=sentence_inputs, outputs=model_output)
