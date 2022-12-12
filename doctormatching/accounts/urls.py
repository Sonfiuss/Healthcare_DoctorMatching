from django.urls import path
from . import views

urlpatterns = [

    path('logout', views.logout , name='logout'),
    path('sign_in_admin', views.sign_in_admin , name='sign_in_admin'),

    path('signup_patient', views.signup_patient, name="signup_patient"),
    path('sign_in_patient', views.sign_in_patient , name='sign_in_patient'),
    
    path('savepdata/<str:patientusername>', views.savepdata , name='savepdata'),
    
]