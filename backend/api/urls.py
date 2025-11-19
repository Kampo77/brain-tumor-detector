from django.urls import path
from .views import PingView, AnalyzeView, PredictView, AppointmentView
from .brats_views import BraTSPredictView, BraTSHealthView

urlpatterns = [
    path('ping/', PingView.as_view(), name='ping'),
    path('analyze/', AnalyzeView.as_view(), name='analyze'),
    path('predict/', PredictView.as_view(), name='predict'),
    path('brats/predict/', BraTSPredictView.as_view(), name='brats-predict'),
    path('brats/health/', BraTSHealthView.as_view(), name='brats-health'),
    path('appointments/', AppointmentView.as_view(), name='appointments'),
]
