from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyse/', views.analyse_fichier, name='analyse_fichier'),
]