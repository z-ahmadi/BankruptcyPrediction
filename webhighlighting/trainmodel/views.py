import numpy as np
import pickle
import sklearn.metrics as sklm
import sys
import time
from os import path, makedirs, environ
from os.path import join, dirname, abspath
from highlight.helper_sentence_filtering import filter_multiproc
from highlight.helper_correlated_pattern_mining import calc_pattern_freqs, keyword_counter_to_chi_square
from preprocessing_helper import load_sentences, sentence_tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import save_model
from keras.preprocessing.text import Tokenizer
from metrics import Metrics
from model import create_model_dscnn_document
from sklearn.utils import class_weight
from django.shortcuts import render
from w2v import get_embeddings, get_embedding_matrix


def index(request):
    """
    hier kommt das form rein

    :param request:
    :return:
    """
    context = {
        'con_var': "",
        'page_name': "trainmodel"
    }
    return render(request, 'trainmodel/index.html', context)


def doit(request):
    # deactivate tensorflow warnings
    environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # to avoid pickle errors
    sys.setrecursionlimit(10000)

    '''
    configuration
    '''
    language = request.POST['language']

    if language is "english":
        from preprocessing_helper import clean_str_engl as clean_str, read_file
        w2v_file = "./data/english_300dim.model"
    else:
        from preprocessing_helper import clean_str_ger as clean_str, read_file
        w2v_file = "./data/german_300dim.model"

    w2v_file = join(dirname(abspath(__file__)), w2v_file)

    timestamp = int(time.time())

    outdir = './models/'
    save_base_name = 'model_' + str(timestamp)
    sentence_len = int(request.POST['sentence_len'])
    batch_size = 50
    epochs = int(request.POST['epochs'])
    patience = 0
    max_vocab_words = None
    mx_doc_len = int(request.POST['mx_doc_len'])
    filtersz = [3, 4, 5]
    w2v_dim = 60
    train_file = request.POST['trainfile']
    test_file = request.POST['testfile']
    valid_file = request.POST['validfile']
    optim = request.POST['optimizer']
    filters = 100
    balanced = True

    '''
    outdir
    '''
    if path.exists(outdir) is False:
        print('Creating path: %s' % outdir)
        makedirs(outdir)

    outname_model = join(outdir, save_base_name + ".model")
    outname_model_last = join(outdir, save_base_name + "_last.model")
    outname_data = join(outdir, save_base_name + ".data")
    outname_patterns = join(outdir, save_base_name + ".patterns")

    '''
    load data
    '''
    print("loading data")
    processing_time = time.time()
    train_txt, train_labels = read_file(train_file)
    test_txt, test_labels = read_file(test_file)
    val_txt, val_labels = read_file(valid_file)
    print("Took", str(time.time() - processing_time), "seconds")

    '''
    filter sentences by correlated pattern mining
    '''
    # tokenize sentences
    print("tokenize sentences")
    processing_time = time.time()
    tokenized_train_instances = [sentence_tokenizer(txt) for txt in train_txt]
    tokenized_valid_instances = [sentence_tokenizer(txt) for txt in val_txt]
    tokenized_test_instances = [sentence_tokenizer(txt) for txt in val_txt]

    print("Took", str(time.time() - processing_time), "seconds")

    if any([len(inst) > mx_doc_len for inst in tokenized_train_instances]):
        print("generate patternsets")
        processing_time = time.time()

        # generate patternsets
        ngram_lengths = [1, 2, 3, 4]
        pattern_sets = calc_pattern_freqs(train_txt, train_labels, ngram_lengths, stopword_removing=True, stemming=True,
                                          language=language)
        doc_lengths = {klass: len([idx for idx in train_labels if idx == klass]) for klass in set(train_labels)}
        print("Took", str(time.time() - processing_time), "seconds")

        print("calculate chi square values")
        processing_time = time.time()
        pattern_sets = keyword_counter_to_chi_square(pattern_sets, doc_len=doc_lengths)
        print("Took", str(time.time() - processing_time), "seconds")

        # filter via multiprocessing
        print("filter train sentences")
        processing_time = time.time()
        tokenized_train_instances = filter_multiproc(tokenized_train_instances, mx_doc_len, pattern_sets)
        print("Took", str(time.time() - processing_time), "seconds")

        # filter test und validation sets
        print("filter test and validationsets")
        processing_time = time.time()
        tokenized_valid_instances = filter_multiproc(tokenized_valid_instances, mx_doc_len, pattern_sets)
        tokenized_test_instances = filter_multiproc(tokenized_test_instances, mx_doc_len, pattern_sets)
        print("Took", str(time.time() - processing_time), "seconds")

        pickle.dump(pattern_sets, open(outname_patterns, 'wb'), protocol=4)
        del pattern_sets

    # concatenate tokenized sentences again
    train_txt = [" ".join(inst) for inst in tokenized_train_instances]
    val_txt = [" ".join(inst) for inst in tokenized_valid_instances]
    test_txt = [" ".join(inst) for inst in tokenized_test_instances]

    '''
    build vocab and train tokenizer
    '''
    print("build vocab and embedding matrix")
    tokenizer = Tokenizer(num_words=max_vocab_words, lower=True, split=" ")
    tokenizer.fit_on_texts(train_txt + val_txt + test_txt)
    embed, embed_dim = get_embeddings(w2v_file, vocab=tokenizer.word_index, reduce_dim=w2v_dim)
    embed_mat = get_embedding_matrix(embed, embed_dim, tokenizer.word_index)

    '''
    build datasets
    '''
    print("build datasets")
    train_inst, labels = load_sentences(train_txt, train_labels, {}, tokenizer, sentence_len, mx_doc_len, clean_str)
    val_inst, labels = load_sentences(val_txt, val_labels, labels, tokenizer, sentence_len, mx_doc_len, clean_str)
    test_inst, labels = load_sentences(test_txt, test_labels, labels, tokenizer, sentence_len, mx_doc_len, clean_str)

    '''
    del large vars that are not needed anymore
    '''
    del train_txt, train_labels, test_txt, test_labels, val_txt, val_labels
    del tokenized_test_instances, tokenized_valid_instances, tokenized_train_instances

    '''
    train the model
    '''
    print("build the model")
    # create the model
    model = create_model_dscnn_document(embed_mat, mx_doc_len, sentence_len, kernel_sizes=filtersz, filters=filters)

    # prepare some functions
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(outname_model, monitor='val_loss', verbose=1, save_best_only=False, period=5)

    # balancing data
    if balanced:
        class_weights = class_weight.compute_class_weight('balanced', np.unique(train_inst.y_1d), train_inst.y_1d)
    else:
        class_weights = class_weight.compute_class_weight(None, np.unique(train_inst.y_1d), train_inst.y_1d)

    # run training
    print("start training. this part may take a long time...")
    train_time = time.time()
    model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['acc'])
    metrics = Metrics()
    history = model.fit(
        train_inst.x,
        train_inst.y,
        batch_size,
        epochs,
        verbose=2,
        callbacks=[checkpoint, early_stopping, metrics],
        validation_data=(val_inst.x, val_inst.y),
        shuffle=True,
        class_weight=class_weights
    )
    print("History:", history)
    train_time = time.time() - train_time
    save_model(model, outname_model_last)

    '''
    evaluation
    '''
    print("start evaluation of this model...")
    score = model.evaluate(test_inst.x, test_inst.y, batch_size, verbose=1)
    print("Score:", score)

    y_prediction_probs = model.predict(test_inst.x, batch_size=batch_size, verbose=2)
    y_prediction = np.argmax(y_prediction_probs, axis=1)

    # model = load_model(outname_model)
    # score = model.evaluate(test_inst.x, test_inst.y, batch_size, verbose=1)

    export_data = {
        'tokenizer': tokenizer,
        'embed': embed,
        'embed_dim': embed_dim,
        'mx_doc_len': mx_doc_len,
        'sentence_len': sentence_len,
        'labels': labels,
        'clean_str': clean_str,
        'load_sentences': load_sentences,
    }
    pickle.dump(export_data, open(outname_data, 'wb'), protocol=4)

    confusion_matrix = sklm.confusion_matrix(test_inst.y_1d, y_prediction)
    precision_score = sklm.precision_score(test_inst.y_1d, y_prediction)
    recall_score = sklm.recall_score(test_inst.y_1d, y_prediction)
    f1_score = sklm.f1_score(test_inst.y_1d, y_prediction)
    cohen_kappa_score = sklm.cohen_kappa_score(test_inst.y_1d, y_prediction)

    '''
    render template
    '''
    context = {
        'con_var': """
        <h5>Training successful</h5>
        <h6>Confusion Matrix</h6>
        <table>
        <tr>
            <th></th>
            <th>actual positive</th>
            <th>actual negative</th>
        </tr>
        <tr>
            <th>predicted positive</th>
            <td>"""+str(confusion_matrix[0][0])+"""</td>
            <td>"""+str(confusion_matrix[0][1])+"""</td>
        </tr>
        <tr>
            <th>predicted negative</th>
            <td>"""+str(confusion_matrix[1][0])+"""</td>
            <td>"""+str(confusion_matrix[1][1])+"""</td>
        </tr>
        </table>
        <h6>Scores</h6>
        <p>
        <b>Precision Score:</b> """+str(precision_score)+"""<br>
        <b>Recall Score:</b> """+str(recall_score)+"""<br>
        <b>F1 Score:</b> """+str(f1_score)+"""<br>
        <b>Cohen Kappa Score:</b> """+str(cohen_kappa_score)+"""
        </p>
        <h6>Training Information</h6>
        <p> 
        <!--<b>Training time:</b> """ + "%.2f" % (train_time/60) + """ minutes <br>-->
        <b>Safe path:</b> """ + str(outname_data) + """
        </p>
        """,

        "test_confusion_matrix:": confusion_matrix,
        "test_precision_score:": precision_score,
        "test_recall_score:": recall_score,
        "test_f1_score:": f1_score,
        "test_cohen_kappa_score:": cohen_kappa_score,

        "train_time": train_time,
        "download_link": outname_data,
        'page_name': "doit"
    }
    return render(request, 'trainmodel/index.html', context)
