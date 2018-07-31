import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import SnowballStemmer, word_tokenize, PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters


class SentenceLabelExamples(object):
    def __init__(self, x, y):
        self.x = x
        self.y_1d = np.array(y)
        self.y = to_categorical(y, num_classes=2)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def width(self):
        return self.x[0].shape[0]


def load_sentences(inst_texts, inst_labels, labels, tokenizer, max_sen_len, max_sen_per_doc, clean_str):
    """
    This method will return a list (x) that contains a list for each instance. One instance contains a list of
    sentences. Each sentence is one np.array of length mxlen, with one index for each word.

    :param clean_str:
    :param inst_texts:
    :param inst_labels:
    :param labels:
    :param tokenizer:
    :param max_sen_len:
    :param max_sen_per_doc:
    :return: x is list (each item is one instance) of list (each item is a sentence) of np-arrays (sequence of word ids)
    """
    number_instances = len(inst_texts)
    inputs = [np.zeros((number_instances, max_sen_len)) for _ in range(max_sen_per_doc)]
    y = np.zeros(number_instances, dtype=int)

    for inst_idx in range(number_instances):
        '''
        label
        '''
        if not inst_labels[inst_idx] in labels:
            labels[inst_labels[inst_idx]] = len(labels)

        y[inst_idx] = labels[inst_labels[inst_idx]]

        '''
        text
        '''
        # get list of all sentences and clean them
        sentences = sentence_tokenizer(inst_texts[inst_idx])
        sentences = [clean_str(text) for text in sentences]

        # sequence and pad each sentence
        sentences = tokenizer.texts_to_sequences(sentences)
        sentences = list(pad_sequences(sentences, maxlen=max_sen_len, padding="post", truncating="post"))

        # pad sentence length
        for input_idx in range(min(max_sen_per_doc, len(sentences))):
            inputs[input_idx][inst_idx] = sentences[input_idx]

    return SentenceLabelExamples(inputs, y), labels


def read_file(file):
    texts = []  # list of text samples
    labels = []  # list of labels

    with open(file, 'r') as f:
        for l in f:
            line = l.split()

            # labeling
            label_id = int(line[0])

            # texts
            texts.append(" ".join(line[1:]))
            labels.append(label_id)

    return texts, labels


def clean_str_engl(string):
    string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# def clean_str_ger(string):
#     string = re.sub(r"[^A-Za-z0-9(),!?'`]", " ", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()


def clean_str_ger(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if len(string) < 10:
        return ""

    # remove numbers at the beginning of strings
    while string[0].isdigit():
        if len(string) < 10:
            return ""
        string = string[1:]

    string = string.replace("§§", "§")

    # remove dates
    string = re.sub("[0-9]{1,2}[.]? [a-zA-Z]{3,} (19|20)\d{2}", " DATUMAUSDRUCK ", string) # e.g. 24. dec 1991
    string = re.sub("^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2])\2))(?:(?:1[6-9]"
                    "|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)0?2\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|"
                    "(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9])|(?:1[0-2]))"
                    "\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$", " DATUMAUSDRUCK ", string)
    string = re.sub("(19|20)\d{2}", " JAHRESAUSDRUCK ", string)

    # numbers (comma and dot numbers)
    string = re.sub("[+]?\d+[.,]{1}\d+", " POSITIVZAHLAUSDRUCK ", string)
    string = re.sub("[-]\d+[.,]{1}\d+", " NEGATIVZAHLAUSDRUCK ", string)

    # all other numbers
    string = re.sub("[-]\d+(?:\.\d+)?", " NEGATIVZAHLAUSDRUCK ", string)
    string = re.sub("[-]\d+(?:\,\d+)?", " NEGATIVZAHLAUSDRUCK ", string)
    string = re.sub("[+]?\d+(?:\.\d+)?", " POSITIVZAHLAUSDRUCK ", string)
    string = re.sub("[+]?\d+(?:\,\d+)?", " POSITIVZAHLAUSDRUCK ", string)

    # space around special characters
    for char in ['%', '&', '§', '.', '!', '?', '(', ')']:
        string = string.replace(char, " "+str(char)+" ")

    # remove bad characters
    oldstring = string
    string = ""
    for c in oldstring:
        string += character_replacement(c)

    for char in ['(', ')', '\n']:
        string = string.replace(char, " ")

    string = string.replace("%", " prozent ")
    string = string.replace("&", " und ")

    # remove double spaces
    string = re.sub(' +', ' ', string)
    return string.strip().lower()


def stem_string(string, language="german"):
    """
    stems each word of a string

    :param string: text
    :param language
    :return: string, where each word is stemmed
    """
    stemmer = SnowballStemmer(language=language)
    tokens = word_tokenize(string, language=language)
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)


def sentence_tokenizer(text):
    """
    Tokenizes sentences.

    :param text:
    :return: list of sentences (a sentence is a string)
    """
    punkt_param = PunktParameters()
    punkt_param.abbrev_types = {'zzgl', 'prof', 'ca', 'vj', 't', 'mio', 'sro', 'lv', 'io', 'ihv', 'bzw', 'usw', 'inkl',
                                'zt', 'vh', 'dr', 'entspr', 'dem', 'fort', 'co', 'kg', 'zb', 'bspw', 'ua', 'rd', 'abs',
                                'etc', 'tsd', 'z.b', 'evtl', '1', '2', '3', '4', '5', '6', '7', '8', '9', '19', '20',
                                '21'}
    sentence_splitter = PunktSentenceTokenizer(punkt_param)
    return sentence_splitter.tokenize(text)


def stopword_cleaner_string(string, language="german"):
    """
    removes all stopwords from a string (without stemming)

    :param string: Text
    :param language: in nltk style
    :return: string without stopwords
    """
    stop = ['aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere',
            'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf',
            'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die', 'das',
            'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe', 'dieselben', 'dasselbe',
            'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn', 'derer', 'dessen', 'dich', 'dir',
            'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses', 'doch', 'dort', 'durch', 'ein', 'eine',
            'einem', 'einen', 'einer', 'eines', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einiges',
            'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'fuer',
            'gegen', 'gewesen', 'hab', 'habe', 'haben', 'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich',
            'mich', 'mir', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins',
            'ist', 'jede', 'jedem', 'jeden', 'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt',
            'kann', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keines', 'koennen', 'koennte', 'machen', 'man',
            'manche', 'manchem', 'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner',
            'meines', 'mit', 'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne',
            'sehr', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind',
            'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst', 'ueber',
            'um', 'und', 'uns', 'unsere', 'unserem', 'unseren', 'unser', 'unseres', 'unter', 'viel', 'vom', 'von',
            'vor', 'waehrend', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche', 'welchem', 'welchen',
            'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will', 'wir', 'wird', 'wirst', 'wo',
            'wollen', 'wollte', 'wuerde', 'wuerden', 'zu', 'zum', 'zur', 'zwar', 'zwischen']
    tokens = word_tokenize(string, language=language)
    tokens = [w for w in tokens if w.lower() not in stop]
    return " ".join(tokens)


def character_replacement(c):
    if ord(c) < 128:
        return c
    elif ord(c) in [224, 225, 226, 227, 229, 230]:
        return "a"
    elif ord(c) in [192, 193, 194, 195, 197, 198]:
        return "a"
    elif ord(c) == 231:
        return "c"
    elif ord(c) in [232, 233, 234, 235]:
        return "e"
    elif ord(c) in [242, 243, 244, 245]:
        return "o"
    elif ord(c.lower()) == 128:
        return " euro "
    elif ord(c) == 36:
        return " dollar "
    elif ord(c) == 37:
        return " prozent "
    elif ord(c.lower()) == 167:
        return " paragraph "
    elif ord(c.lower()) == 223:
        return "ss"
    elif ord(c.lower()) == 228:
        return "ae"
    elif ord(c.lower()) == 246:
        return "oe"
    elif ord(c.lower()) == 252:
        return "ue"
    elif ord(c) == 163:
        return " pfund "
    elif ord(c) in [129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                    149, 150, 151, 152, 153]:
        return " "
    elif ord(c) == 173:
        return " - "
    elif ord(c) == 183:
        return " . "
    elif ord(c) in [40, 41, 91, 92, 93, 123, 124, 125]:
        return " "
    else:
        return " "
