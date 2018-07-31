from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'/trainmodel/doit/', views.doit, name='doit'),
    url(r'^doit/$', views.doit)
]
