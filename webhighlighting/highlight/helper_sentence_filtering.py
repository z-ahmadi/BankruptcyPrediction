import multiprocessing
from nltk import word_tokenize
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer
from preprocessing_helper import stopword_cleaner_string, stem_string


def sentence_tokenizer(text):
    """
    Tokenizes sentences.

    :param text:
    :return: list of sentences (a sentence is a string)
    """
    text = text.replace("<br />", ". ")

    punkt_param = PunktParameters()
    punkt_param.abbrev_types = {'zzgl', 'prof', 'ca', 'vj', 't', 'mio', 'sro', 'lv', 'io', 'ihv', 'bzw', 'usw', 'inkl',
                                'zt', 'vh', 'dr', 'entspr', 'dem', 'fort', 'co', 'kg', 'zb', 'bspw', 'ua', 'rd', 'abs',
                                'etc', 'tsd', 'z.b', 'evtl', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                                '25', '26', '27', '28', '29', '30', '31'}
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    return sentence_splitter.tokenize(text)


def filter_instance(instance_text, patterns, language="german", tokenized=True):
    """

    :param instance_text:
    :param patterns:
    :param language:
    :param tokenized:
    :return:
    """
    if tokenized:
        sentences = instance_text
    else:
        sentences = sentence_tokenizer(instance_text.strip())

    new_doc_content = []
    for sentence in sentences:
        sentence_wo_stopwords = word_tokenize(stopword_cleaner_string(stem_string(str(sentence), language), language),
                                              language=language)
        if any(keyword in sentence_wo_stopwords for keyword in patterns):
            new_doc_content.append(sentence)

    return new_doc_content


def flat_list(l):
    return [item for sublist in l for item in sublist]


def get_top_pattern(_pattern_set, number=100, chi=None):
    if chi is not None:
        return [pattern for pattern, value in _pattern_set.items() if value > chi]
    else:
        return sorted(_pattern_set, key=lambda x: _pattern_set[x], reverse=True)[:number]


def filter_instance_worker(tokenized_instance, mx_doc_len, pattern_sets):
    """

    :param tokenized_instance: List of sentences that belong to one document
    :param mx_doc_len: max document length
    :param pattern_sets: list of two pattern sets
    :return: list of maximum mx_doc_len documents
    """
    inst_filtered = tokenized_instance

    if len(inst_filtered) > mx_doc_len:
        n_of_patterns = int(mx_doc_len / len(inst_filtered) * sum([len(p_set) for p_set in pattern_sets]) / 2)
        tmp_pattern_sets = [get_top_pattern(p_set, number=n_of_patterns) for p_set in pattern_sets]
        inst_filtered = filter_instance(inst_filtered, flat_list(tmp_pattern_sets), tokenized=True)

        while len(inst_filtered) < mx_doc_len:
            n_of_patterns += 1
            tmp_pattern_sets = [get_top_pattern(p_set, number=n_of_patterns) for p_set in pattern_sets]
            inst_filtered = filter_instance(tokenized_instance, flat_list(tmp_pattern_sets), tokenized=True)

        while len(inst_filtered) > mx_doc_len:
            tmp_pattern_sets = [l[:-1] for l in tmp_pattern_sets]
            inst_filtered = filter_instance(inst_filtered, flat_list(tmp_pattern_sets), tokenized=True)

    return inst_filtered


def filter_multiproc(tokenized_instances, mx_doc_len, pattern_sets):
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    tasks = [(tokenized_instance, mx_doc_len, pattern_sets,) for tokenized_instance in tokenized_instances]
    results = [pool.apply_async(filter_instance_worker, t) for t in tasks]
    tokenized_instances = [result.get() for result in results]
    pool.close()
    pool.join()
    return tokenized_instances
