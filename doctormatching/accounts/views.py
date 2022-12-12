from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import HttpResponseRedirect, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from django.contrib import messages
from django.contrib.auth.models import User , auth
from main_app.models import patient , doctor
from datetime import datetime

# Create your views here.


   
def logout(request):
    auth.logout(request)
    request.session.pop('patientid', None)
    request.session.pop('doctorid', None)
    request.session.pop('adminid', None)
    return render(request,'homepage/index.html')
