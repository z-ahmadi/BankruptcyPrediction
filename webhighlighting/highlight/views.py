from django.shortcuts import render
from highlight import perform_highlighting, perform_prediction


def index(request):
    """
    hier kommt das form rein

    :param request:
    :return:
    """
    context = {
        'con_var': "",
        'page_name': "index"
    }
    return render(request, 'highlight/index.html', context)


def doit(request):
    sentences, prob_softmax, sentence_is_significant_map = perform_prediction(request.POST['mrinput'])
    highlighted_html = perform_highlighting(sentences, prob_softmax, sentence_is_significant_map)

    context = {
        'con_var': highlighted_html,
        'page_name': "doit"
    }
    return render(request, 'highlight/index.html', context)
