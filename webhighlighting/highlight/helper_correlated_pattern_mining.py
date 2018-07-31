from collections import Counter
from highlight.helper_ngram import get_document_frequencies
import numpy as np


def get_document_frequencies_as_single_counter(txt_list, ngram_lengths, stopword_removing, stemming, language):
    doc_freqs = get_document_frequencies(txt_list, ngram_lengths, stopword_removing, stemming, language)

    df_complete_counter = Counter()
    for k, val in doc_freqs.items():
        df_complete_counter += val

    return df_complete_counter


def calc_pattern_freqs(txt_list, label_list, ngram_lengths, stopword_removing=True, stemming=True, language="german"):
    """

    :param txt_list:
    :param ngram_lengths:
    :param stopword_removing:
    :param stemming:
    :param language:
    :return:
    """
    text_to_class = {klass: [idx for idx in label_list if idx == klass] for klass in set(label_list)}
    return {klass: get_document_frequencies_as_single_counter(
        [txt_list[idx] for idx in text_to_class[klass]],
        ngram_lengths,
        stopword_removing,
        stemming,
        language
    ) for klass in set(label_list)}


def keyword_counter_to_chi_square(keyword_counter, doc_len):
    """
    transform to [keyword: chi-square for each klass]

    :param keyword_counter: should be [keyword_counter_ins, keyword_counter_sol]
    :param doc_len: [length_of_doc_ins, length_of_doc_sol]
    :return: 2 lists, one for solvent, one for insolvent: index 0: insolvent, index 1: solvent
    """
    def calc_chi_square_value(a, n):
        """
        @NEW!

        :param a: [a_i for i in range(n_classes)] as in chapter 5 of multi-class correlated pattern mining by
                    siegrfried nijssen et al.
                    a_i is the relative frequency of the considered pattern in the docs of the first document.
        :param n: [n_i for i in range(n_classes)] number of docs in class i
        :return: the chi-square value
        """
        if len(a) is not len(n):
            raise Exception("Number of elements in arrays a and n has to fit.")

        if any(abs(val) > 1 for val in a):
            raise Exception("All values in a should be in [0,1].")

        def chisquare(f_obs, f_exp):
            f_exp = f_exp if f_exp > 0 else 1
            return (f_obs - f_exp) * (f_obs - f_exp) / f_exp

        n_classes = len(a)

        a = np.array(a)
        n = list(n.values())
        n = np.array(n)
        N = sum(n)

        # preparation
        scalar_an = np.dot(a, n)
        scalar_an_div_N = scalar_an / N
        scalar_an_counter = N - scalar_an
        scalar_an_counter_div_N = scalar_an_counter / N
        # scalar_an = np.dot(a,n)

        # expected vals
        E_i1 = [scalar_an_div_N * n[i] for i in range(n_classes)]
        E_i2 = [scalar_an_counter_div_N * n[i] for i in range(n_classes)]

        # observed vals
        O_i1 = [a[i] * n[i] for i in range(n_classes)]
        O_i2 = [(1 - a[i]) * n[i] for i in range(n_classes)]

        # chi-square-vals
        chis = [chisquare(O_i1[i], E_i1[i]) + chisquare(O_i2[i], E_i2[i]) for i in range(n_classes)]

        return sum(chis)

    # transform to [keyword: chi-square for each klass]
    return [
        {
            " ".join(pattern): calc_chi_square_value(
                a=[support / doc_len[j], keyword_counter[1-j][pattern] / doc_len[1-j]],
                n=doc_len)
            for pattern, support in keyword_counter[j].items()
        } for j in range(len(doc_len))
    ]
