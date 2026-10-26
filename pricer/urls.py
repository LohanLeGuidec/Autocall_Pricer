from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("market-data/", views.market_data, name="market_data"),
    path("correlation/", views.correlation, name="correlation"),
    path("calibration/", views.calibration, name="calibration"),
    path("plotly.js", views.plotly_js, name="plotly_js"),
]
