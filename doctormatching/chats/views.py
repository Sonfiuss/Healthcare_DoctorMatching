from django.shortcuts import render , redirect
from django.http import HttpResponse, HttpResponseRedirect, JsonResponse
from .models import Chat , Feedback
from main_app.views import patient_ui, doctor_ui
from main_app.models import patient , doctor

# Create your views here.


def get_feedback(request):
    
    if request.method == "GET":

      obj = Feedback.objects.all()
      
      return redirect(request, 'consultation/chat_body.html',{"obj":obj})

