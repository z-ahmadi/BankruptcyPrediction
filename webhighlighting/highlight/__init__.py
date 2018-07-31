from colour import Color
from keras.models import load_model
import pickle
import numpy as np
from yattag import Doc
from highlight.helper_sentence_filtering import sentence_tokenizer
from highlight.helper_sentence_filtering import filter_instance, filter_instance_worker
from preprocessing_helper import load_sentences, clean_str_ger as clean_str


mp_ = "model_highlight_example/20170210_dscnn_azp-dataset_100epochs"
model_ = load_model(mp_ + ".model")


def perform_prediction(instance_text, model=model_, model_path=mp_, batch_size=50, filtering=True):
    """
    load data and model
    """
    data_container = pickle.load(open(model_path + '.data', 'rb'))
    tokenizer = data_container['tokenizer']
    sentence_len = data_container['sentence_len']
    mx_doc_len = data_container['mx_doc_len']
    del data_container

    # load filter patterns
    if filtering:
        patterns = pickle.load(open(model_path + '.patterns', 'rb'))
        filtered_sentences = filter_instance_worker(sentence_tokenizer(instance_text), mx_doc_len, patterns)
        del patterns
    else:
        filtered_sentences = sentence_tokenizer(instance_text)

    # preprocess data
    # the elements in the lists 'texts' are the instances to be predicted with instance_text, each left a sentence out
    texts = [" ".join([s for s in filtered_sentences if filtered_sentences.index(s) != idx])
             for idx in range(min(mx_doc_len, len(filtered_sentences)))] + [" ".join(filtered_sentences)]
    labels = [0 for _ in range(len(texts))]
    classify_instances = load_sentences(texts, labels, {}, tokenizer, sentence_len, mx_doc_len, clean_str)[0]

    # prediction
    prob_softmax = model.predict(classify_instances.x, batch_size=batch_size, verbose=2)

    # compare instance text and sentences and get index mapping
    sentence_is_significant_map = [True if sentence in filtered_sentences else False
                                   for sentence in sentence_tokenizer(instance_text)]

    # return sentences, prob_softmax
    return sentence_tokenizer(instance_text), prob_softmax, sentence_is_significant_map


def get_impacts(prob_softmax):
    probs = np.max(prob_softmax, axis=1)

    # evaluation: calculate saliencies / impacts
    impacts = [np.divide(probs[-1] - probs[idx], probs[-1]) for idx in range(len(probs) - 1)]

    return impacts


def perform_highlighting(sentences, prob_softmax, sentence_is_significant_map,
                         color_min="red", color_max="green", color_steps=10):
    # convert softmax
    impacts = get_impacts(prob_softmax)
    probs = np.max(prob_softmax, axis=1)
    print("Prob", probs[-1])
    preds = np.argmax(prob_softmax, axis=1)

    # generate color gradient
    gradient = list(Color(color_min).range_to(Color(color_max), color_steps))
    impact_stepwidth = 2/color_steps  # 2 = impact_max-impact_min
    sentence_color_idx = [int((impact + 1)/impact_stepwidth) for impact in impacts]  # 1 = - impact_min

    # generate html
    doc, tag, text = Doc().tagtext()

    # print prediction
    with tag('strong'):
        text("The report is classified as {} with a probability of {:.2f} %.".format(
            ("distressed" if preds[-1] == 0 else "safe"),
            probs[-1]*100
        ))

    # doc.stag('br')
    with tag("h5"):
        text("Sentence Filtering")

    with tag("p"):
        text("You can let the system display only sentences that have an impact greater / less than a given value on "
             "the classification result:")

    # show filter slider
    with tag("div", klass="mdl-grid"):
        with tag("div", klass="mdl-cell mdl-cell--3-col"):
            with tag('label',
                     ('class', 'mdl-radio mdl-js-radio mdl-js-ripple-effect'),
                     ('for', 'radio-greater')):
                doc.input(name='radio-greater-less', id="radio-greater", type="radio", klass="mdl-radio__button",
                          value="greater", checked="checked")
                with tag('span', klass="mdl-radio__label"):
                    text("impact greater than")

            with tag('label',
                     ('class', 'mdl-radio mdl-js-radio mdl-js-ripple-effect'),
                     ('for', 'radio-less')):
                doc.input(name='radio-greater-less', id="radio-less", type="radio", klass="mdl-radio__button",
                          value="less")
                with tag('span', klass="mdl-radio__label"):
                    text("impact less than")

        with tag("div", klass="mdl-cell mdl-cell--6-col", id="impact-slider-div"):
            doc.input(name="impact-slider", id="impact-slider", klass="mdl-slider mdl-js-slider", type="range", min="0",
                      max="1", step="0.05", value="0", tabindex="0")

        with tag("div", klass="mdl-cell mdl-cell--1-col"):
            with tag("button", ("type", "button"), ("class", "mdl-chip")):
                with tag("span", klass="mdl-chip__text", id="impact-slider-value"):
                    text("0.00")

        with tag("div", klass="mdl-cell mdl-cell--2-col", style="text-align:right;"):
            with tag("button", ("type", "button"),
                     ("class", "mdl-button mdl-js-button mdl-button--raised mdl-button--accent"), ("id", "filter-go")):
                text("filter")

    # doc.stag('br')
    with tag("h5"):
        text("Report")

    # print management report sentences
    for idx in range(len(impacts)):
        if sentence_is_significant_map[idx]:
            _css = 'background-color: ' + str(gradient[sentence_color_idx[idx]])
            with tag('span', ('style', _css), ("data-impact", str(impacts[idx]))):
                text(sentences[idx] + " ")
        else:
            _css = ''
            with tag('span', ('style', _css), ("data-impact", '0.0')):
                text(sentences[idx] + " ")

    # print colors
    with tag('h5'):
        text("Color Legend:")

    with tag('table', ('style', 'width: 100%; height: 3em;')):
        with tag('tr'):
            for idx in range(len(gradient)):
                color_css = "padding: .3em; text-align: center; background-color: " + gradient[idx].get_web()
                with tag('td', ('style', color_css)):
                    text("{:.2f}".format(-1 + idx*impact_stepwidth))  # -1 = impact_min
                    doc.stag('br')
                    text("to")
                    doc.stag('br')
                    text("{:.2f}".format(-1 + (idx+1)*impact_stepwidth))  # -1 = impact_min

    return doc.getvalue()
