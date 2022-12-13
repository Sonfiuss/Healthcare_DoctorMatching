from django.shortcuts import render, ren
import joblib as jb
from django.http import HttpResponse
from django.http import JsonResponse
from doctormatching.chats.models import Feedback, Chat
# Create your views here.
model = jb.load('trained_model')

def home(request):
  if request.method == 'GET':   
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')
      else :
        return render(request,'homepage/index.html')

def admin_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        auser = request.user
        Feedbackobj = Feedback.objects.all()

        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')
