import multiprocessing
from nltk.util import ngrams
from collections import Counter
from nltk import word_tokenize, SnowballStemmer
from preprocessing_helper import clean_str_ger as clean_str, stopword_cleaner_string  # TODO: Change this for multi-lang


def generate_ngram(text, ngram_lengths, stopword_removing=True, stemming=True, language="german", return_type="ngram"):
    """
    generates ngrams.

    :param text: string which should be used to generate the ngram
    :param ngram_lengths: integer or list of integers
    :param stopword_removing: boolean, should the stopwords be removed
    :param stemming: stem all words
    :param language: language for nltk use
    :param return_type: "counter", "set" or "ngram"
    :return: map of ngrams (depending on @return_type, if its gonna be a set or counter). the key of the map is the
             ngram-length.

    """
    if ngram_lengths == 0:
        return
    if type(ngram_lengths) == int:
        ngram_lengths = [ngram_lengths]

    language = language.lower()

    # text preprocessing
    text = clean_str(text)

    # stopword removing
    if stopword_removing:
        text = stopword_cleaner_string(text, language=language)

    # tokenize
    tokens = word_tokenize(text, language=language)

    # stemming
    if stemming:
        stemmer = SnowballStemmer(language=language)
        tokens = [stemmer.stem(w) for w in tokens]

    # return
    return_type = return_type.lower()
    if return_type == "counter":
        return {i: Counter(ngrams(tokens, i)) for i in ngram_lengths}
    elif return_type == "set":
        return {i: Counter(set(ngrams(tokens, i))) for i in ngram_lengths}
    else:
        return {i: ngrams(tokens, i) for i in ngram_lengths}


def generate_ngrams_by_textlist(textlist, ngram_lengths, stopword_removing=True, stemming=True, language="german",
                                return_type="ngram"):
    """
    generates ngram out of textlist

    :param textlist: list of texts which should be used to generate the ngram
    :param ngram_lengths: integer or list of integers
    :param stopword_removing: boolean, should the stopwords be removed
    :param stemming: stem all words
    :param language: language for nltk use
    :param return_type: "counter", "set" or "ngram"
    :return: list of maps of ngrams if type(n)=list (it is a deep list)
             first level: documents, second level: maps where n is the key, and the value are ngrams
    """
    if return_type.lower() is not "ngram":
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        tasks = [(text, ngram_lengths, stopword_removing, stemming, language, return_type, ) for text in textlist]
        results = [pool.apply_async(generate_ngram, t) for t in tasks]
        return_var = [result.get() for result in results]
        pool.close()
        pool.join()
        return return_var
    else:
        return [generate_ngram(text, ngram_lengths=ngram_lengths, stopword_removing=stopword_removing,
                               stemming=stemming, language=language, return_type=return_type) for text in textlist]


def calculate_document_frequency(ngram_list):
    """
    Counts the document frequency. That means it counts in how many ngrams each phrase occurs.

    :param ngram_list: list of ngrams (nltk) ( preprocessing needed, type: Counter(set(ngram)) )
    :return: Counter, with the document frequency
    """
    def helper_chunk_list(seq, number_of_parts):
        avg = len(seq) / float(number_of_parts)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    def helper_add_counters(counter_obj):
        total_counter = Counter()
        for idx in range(len(counter_obj)):
            total_counter += counter_obj[idx]
            counter_obj[idx] = 0

        return total_counter

    # chunk lists
    chunk_list = helper_chunk_list(ngram_list, multiprocessing.cpu_count())

    # multiproc pool
    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    # define tasks for the pool (one task is reading one file and returning the document)
    tasks = [(chunk,) for chunk in chunk_list]

    # run the pool of workers
    results = [pool.apply_async(helper_add_counters, t) for t in tasks]

    # getting the results
    chunk_added_counters = [result.get() for result in results]

    pool.close()
    pool.join()

    df_counter = Counter()
    for c in chunk_added_counters:
        df_counter += c

    return df_counter


def get_document_frequencies(list_docs, ngram_lengths, stopword_removing, stemming, language):
    """
    run the whole pipeline to get document frequencies of one dataset.
    :param list_docs: list of documents
    :param ngram_lengths: list of n-gram types
    :param stopword_removing:
    :param stemming:
    :param language:
    :return: document_frequencies as a dictionary, where key: k-gram and value: df-counter (collections.Counter())
    """
    if type(ngram_lengths) == int:
        ngram_lengths = [ngram_lengths]

    # ngrams is a list of ngrams if type(n)=int resp. list of maps of ngrams if type(n)=list (it is a deep list)
    ngrame = generate_ngrams_by_textlist(list_docs, ngram_lengths=ngram_lengths, stopword_removing=stopword_removing,
                                         stemming=stemming, language=language, return_type="set")
    del list_docs

    # order ngrams, so put 1-grams in one list, 2-grams in another list and so on...
    ngrams_ordered_list = {k: list() for k in ngram_lengths}
    for ngram_dict in enumerate(ngrame):
        for k, ngram in ngram_dict[1].items():
            ngrams_ordered_list[k].append(ngram)
    del ngrame

    # calculate document frequency of ngrams,
    # document_frequencies is a dictionary
    # k: k-gram, v: df-counter (collections.Counter())
    document_frequencies = {k: calculate_document_frequency(v) for k, v in ngrams_ordered_list.items()}

    return document_frequencies
