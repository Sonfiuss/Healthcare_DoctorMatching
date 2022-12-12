from django.shortcuts import render
import joblib as jb
# Create your views here.
model = jb.load('trained_model')

def home(request):

  if request.method == 'GET':   
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')
      else :
        return render(request,'homepage/index.html')
