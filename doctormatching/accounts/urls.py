from django.urls import path
from . import view

urlpatterns = [


    path('sign_in_admin', view.sign_in_admin , name='sign_in_admin'),

    path('signup_patient', view.signup_patient, name="signup_patient"),
    path('sign_in_patient', view.sign_in_patient , name='sign_in_patient'),
    path('savepdata/<str:patientusername>', view.savepdata , name='savepdata'),
    

    path('signup_doctor', view.signup_doctor , name="signup_doctor"),
    path('sign_in_doctor', view.sign_in_doctor , name='sign_in_doctor'),
    path('saveddata/<str:doctorusername>', view.saveddata , name='saveddata'),

    path('logout', view.logout , name='logout'),
]