from django.urls import path
from . import views

app_name = 'myapps'

urlpatterns = [
    path('', views.index, name='index'),
    path('strategies/', views.strategies, name='strategies'),
    path('analyze/', views.analyze, name='analyze'),
    path('improve/', views.improve, name='improve'),


]