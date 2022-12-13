from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date

from django.contrib import messages
from django.contrib.auth.models import User , auth
from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback

# Create your views here.


#loading trained_model
import joblib as jb
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





def patient_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)

        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')

       


def pviewprofile(request, patientusername):

    if request.method == 'GET':

          puser = User.objects.get(username=patientusername)

          return render(request,'patient/view_profile/view_profile.html', {"puser":puser})




def checkdisease(request):

  diseaselist=['Nhiễm nấm','Dị ứng',' Trào ngược dạ dày thực quản ','Ứ mật mãn tính','Phản ứng thuốc','Loét dạ dày tá tràng','AIDS',
               'Tiểu đường ', 'Viêm dạ dày ruột','Hen phế quản','Tăng huyết áp ','Chứng đau nửa đầu','Đốt sống cổ','Bại liệt (xuất huyết não)', 
               'Viêm gan','Sốt rét','Thủy đậu','Sốt xuất huyết','Thương hàn','Viêm gan A', 'Viêm gan B', 'Viêm gan C', 'Viêm gan D', 'Viêm gan E', 
               'Viêm gan do rượu','Bệnh lao', 'Cảm lạnh thông thường', 'Viêm phổi', 'Bệnh trĩ lưỡng hình', 'Đau tim', 'Giãn tĩnh mạch','Suy giáp',
               'Cường giáp', 'Hạ đường huyết', 'Bệnh thoái hóa khớp', 'Viêm khớp', '(chóng mặt) Chóng mặt Lành tính do Tư thế','Mụn trứng cá', 
               'Nhiễm trùng đường tiết niệu', 'Vảy nến', 'Chốc lở']


  symptomslist=['ngứa','phát ban da','phát ban nốt trên da','hắt hơi liên tục','run rẩy','ớn lạnh','đau khớp',
   'đau dạ dày','có tính axit','loét trên lưỡi','gầy mòn cơ bắp','nôn mửa','đi tiểu nóng rát','đi tiểu ra máu',
   'mệt mỏi','tăng cân','lo lắng','tay chân lạnh','tâm trạng thất thường','giảm cân','bồn chồn','thờ ơ',
   'cổ họng','mức đường không đều','ho','sốt cao','mắt trũng sâu','khó thở','đổ mồ hôi',
   'mất nước','khó tiêu','nhức đầu','da hơi vàng','nước tiểu sẫm màu','buồn nôn','chán ăn','đau sau mắt',
   'đau lưng','táo bón','đau bụng','tiêu chảy','sốt nhẹ','nước tiểu vàng',
   'vàng mắt', 'suy gan cấp tính', 'quá tải chất lỏng', 'sưng bụng',
   'sưng hạch bạch huyết','khó chịu','mắt mờ và méo mó','đờm','ngứa cổ họng',
   'đỏ mắt','áp lực xoang','chảy nước mũi','nghẹt mũi','đau ngực','yếu tay chân',
   'nhịp tim nhanh','đau khi đi tiêu','đau vùng hậu môn','phân có máu',
   'khó chịu ở hậu môn','đau cổ','chóng mặt','chuột rút','bầm tím','béo phì','sưng chân',
   'mạch máu sưng lên','mặt và mắt sưng húp','tuyến giáp to','móng tay giòn',
   'sưng lên cực độ','đói quá mức','quan hệ vợ chồng nhiều hơn','môi khô và ngứa ran',
   'nói lắp','đau đầu gối','đau khớp háng','yếu cơ','cứng cổ','sưng khớp',
   'cứng cử động','chuyển động xoay tròn','mất thăng bằng','không vững',
   'yếu một bên cơ thể','mất khứu giác','khó chịu ở bàng quang','nước tiểu có mùi',
   'cảm giác đi tiểu liên tục','khí thải ra ngoài','ngứa bên trong','nhìn có độc (typhos)',
   'trầm cảm','cáu kỉnh','đau cơ','cảm giác thay đổi','đốm đỏ trên cơ thể','đau bụng',
   'kinh nguyệt bất thường','mảng đổi màu','chảy nước mắt','thèm ăn hơn','đa niệu','tiền sử gia đình','đờm nhầy',
   'đờm gỉ','thiếu tập trung','rối loạn thị giác','được truyền máu',
   'tiêm thuốc không vô trùng','hôn mê','chảy máu dạ dày','bụng phình to',
   'tiền sử uống rượu', 'quá tải chất lỏng', 'máu trong đờm', 'tĩnh mạch nổi rõ trên bắp chân',
   'đánh trống ngực','đi lại đau đớn','mụn nhọt đầy mủ','mụn đầu đen','sẹo lõm','lột da',
   'bạc như bụi','vết lõm nhỏ trên móng tay','móng tay bị viêm','mụn nước','mẩn đỏ quanh mũi',
   'lớp vỏ màu vàng chảy ra']

  alphabaticsymptomslist = sorted(symptomslist)

  


  if request.method == 'GET':
    
     return render(request,'patient/checkdisease/checkdisease.html', {"list2":alphabaticsymptomslist})




  elif request.method == 'POST':
       
      ## access you data by playing around with the request.POST object
      
      inputno = int(request.POST["noofsym"])
      print(inputno)
      if (inputno == 0 ) :
          return JsonResponse({'predicteddisease': "none",'confidencescore': 0 })
  
      else :

        psymptoms = []
        psymptoms = request.POST.getlist("symptoms[]")
       
        print(psymptoms)

      
        """      #main code start from here...
        """
      

      
        testingsymptoms = []
        #append zero in all coloumn fields...
        for x in range(0, len(symptomslist)):
          testingsymptoms.append(0)


        #update 1 where symptoms gets matched...
        for k in range(0, len(symptomslist)):

          for z in psymptoms:
              if (z == symptomslist[k]):
                  testingsymptoms[k] = 1


        inputtest = [testingsymptoms]

        print(inputtest)
      

        predicted = model.predict(inputtest)
        print("bệnh được dự đoán là : ")
        print(predicted)

        y_pred_2 = model.predict_proba(inputtest)
        confidencescore=y_pred_2.max() * 100
        print(" độ tin cậy : = {0} ".format(confidencescore))

        confidencescore = format(confidencescore, '.0f')
        predicted_disease = predicted[0]

        

        #consult_doctor codes----------

        #   doctor_specialization = ["Rheumatologist","Cardiologist","ENT specialist","Orthopedist","Neurologist",
        #                             "Allergist/Immunologist","Urologist","Dermatologist","Gastroenterologist"]
        

        Rheumatologist = [ 'Viêm xương khớp','Viêm khớp']
       
        Cardiologist = [ 'Đau tim','Hen phế quản','Tăng huyết áp ']
       
        ENT_specialist = ['(chóng mặt) Chóng mặt Lành tính do Tư thế','Suy giáp' ]

        Orthopedist = []

        Neurologist = ['Giãn tĩnh mạch','Bại liệt (xuất huyết não)','Chứng đau nửa đầu','Thoái hóa đốt sống cổ']

        Allergist_Immunologist = ['Dị ứng','Viêm phổi',
         'AIDS','Cảm lạnh thông thường','Lao','Sốt rét','Sốt xuất huyết','Thương hàn']

        Urologist = [ 'Nhiễm trùng đường tiết niệu',
          'Trĩ lưỡng hình']

        Dermatologist = [  'Mụn trứng cá','Thủy đậu','Nhiễm nấm','Bệnh vẩy nến','Chốc lở']

        Gastroenterologist = ['Loét dạ dày tá tràng', 'GERD', 'Ứ mật mãn tính', 'Phản ứng thuốc', 'Viêm dạ dày ruột', 'Viêm gan E',
         'Viêm gan do rượu','Vàng da','Viêm gan A',
          'Viêm gan B', 'Viêm gan C', 'Viêm gan D','Tiểu đường','Hạ đường huyết']
        print("=================")
        print(predicted_disease+"\n================")
        if(predicted_disease == "Rheumatologist"):
               predicted_disease = "Thấp khớp"
        if(predicted_disease == "Fungal infection"):
               predicted_disease = "Nhiễm nấm"
        if(predicted_disease == "Allergy"):
               predicted_disease = "Dị ứng"
        if(predicted_disease == "GERD"):
               predicted_disease = "Trào ngược dạ dày thực quản"
        if(predicted_disease == "Chronic cholestasis"):
               predicted_disease = "Ứ mật mãn tính"
        if(predicted_disease == "Drug Reaction"):
               predicted_disease = "Phản ứng thuốc"
        if(predicted_disease == "Diabetes"):
               predicted_disease = "Bệnh tiểu đường"
        if(predicted_disease == "Gastroenteritis"):
               predicted_disease = "Viêm dạ dày ruột"
        if(predicted_disease == "Bronchial Asthma"):
               predicted_disease = "Hen phế quản"
        if(predicted_disease == "Hypertension"):
               predicted_disease = "Tăng huyết áp"
               
        if(predicted_disease == "Cervical spondylosis"):
               predicted_disease = "Thoái hóa đốt sống cổ"
        if(predicted_disease == "Paralysis (brain hemorrhage)"):
               predicted_disease = "Tê liệt (xuất huyết não)"
        if(predicted_disease == "Jaundice"):
               predicted_disease = "Vàng da"
        if(predicted_disease == "Malaria"):
               predicted_disease = "Bệnh sốt rét"
        if(predicted_disease == "Dengue"):
               predicted_disease = "Thủy đậu"
        if(predicted_disease == "Typhoid"):
               predicted_disease = "Sốt xuất huyết"
        if(predicted_disease == "hepatitis A"):
               predicted_disease = "Viêm gan A"
               
        if(predicted_disease == "hepatitis B"):
               predicted_disease = "Viêm gan B"
        if(predicted_disease == "hepatitis C"):
               predicted_disease = "Viêm gan C"
        if(predicted_disease == "hepatitis D"):
               predicted_disease = "Viêm gan D"                   
        if(predicted_disease == "hepatitis E"):
               predicted_disease = "Viêm gan E"
        if(predicted_disease == "Alcoholic hepatitis"):
               predicted_disease = "Viêm gan do rượu"
        if(predicted_disease == "Tuberculosis"):
               predicted_disease = "tăng huyết áp"
        if(predicted_disease == "Common Cold"):
               predicted_disease = "Cảm lạnh thông thường"               
        if(predicted_disease == "Pneumonia"):
               predicted_disease = "Viêm phổi"
        if(predicted_disease == "Dimorphic hemmorhoids(piles)"):
               predicted_disease = "Trĩ lưỡng tính"
        if(predicted_disease == "Heart attack"):
               predicted_disease = "Đau tim"
        if(predicted_disease == "Varicose veins"):
               predicted_disease = "Suy tĩnh mạch"
        if(predicted_disease == "Hypothyroidism"):
               predicted_disease = "Suy giáp"
        if(predicted_disease == "Hyperthyroidism"):
               predicted_disease = "Cường giáp"
        if(predicted_disease == "Hypoglycemia"):
               predicted_disease = "Hạ đường huyết"
        if(predicted_disease == "Osteoarthristis"):
               predicted_disease = "Thoái hóa khớp"
        if(predicted_disease == "Arthritis"):
               predicted_disease = "Viêm khớp"
        if(predicted_disease == "(vertigo) Paroymsal  Positional Vertigo"):
               predicted_disease = "Chóng mặt Lành tính do Tư thế"
        if(predicted_disease == "Acne"):
               predicted_disease = "Mụn"
        if(predicted_disease == "Urinary tract infection"):
               predicted_disease = "Nhiễm trùng đường tiết niệu"
        if(predicted_disease == "Psoriasis"):
               predicted_disease = "Bệnh vẩy nến"
        if(predicted_disease == "Impetigo"):
               predicted_disease = "Chốc lở"
                
               
               
                                            
        if predicted_disease in Rheumatologist :
           consultdoctor = "Thấp khớp"
           
        if predicted_disease in Cardiologist :
           consultdoctor = "Tim mạch"
           

        elif predicted_disease in ENT_specialist :
           consultdoctor = "Chuyên khoa tai mũi họng"
     
        elif predicted_disease in Orthopedist :
           consultdoctor = "Chỉnh hình"
     
        elif predicted_disease in Neurologist :
           consultdoctor = "Thần kinh"
     
        elif predicted_disease in Allergist_Immunologist :
           consultdoctor = "Dị ứng/miễn dịch học"
     
        elif predicted_disease in Urologist :
           consultdoctor = "Tiết niệu"
     
        elif predicted_disease in Dermatologist :
           consultdoctor = "Da liễu"
     
        elif predicted_disease in Gastroenterologist :
           consultdoctor = "Chuyên khoa tiêu hóa"
     
        else :
           consultdoctor = "Khác"


        request.session['doctortype'] = consultdoctor 

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
     

        #saving to database.....................

        patient = puser.patient
        diseasename = predicted_disease
        no_of_symp = inputno
        symptomsname = psymptoms
        confidence = confidencescore

        diseaseinfo_new = diseaseinfo(patient=patient,diseasename=diseasename,no_of_symp=no_of_symp,symptomsname=symptomsname,confidence=confidence,consultdoctor=consultdoctor)
        diseaseinfo_new.save()
        

        request.session['diseaseinfo_id'] = diseaseinfo_new.id

        print("lưu bệnh án thành công............................")

        return JsonResponse({'predicteddisease': predicted_disease ,'confidencescore':confidencescore , "consultdoctor": consultdoctor})
   


   
    



   





def pconsultation_history(request):

    if request.method == 'GET':

      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      patient_obj = puser.patient
        
      consultationnew = consultation.objects.filter(patient = patient_obj)
      
    
      return render(request,'patient/consultation_history/consultation_history.html',{"consultation":consultationnew})


def dconsultation_history(request):

    if request.method == 'GET':

      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
        
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
      
    
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})



def doctor_ui(request):

    if request.method == 'GET':

      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)

    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})



      


def dviewprofile(request, doctorusername):

    if request.method == 'GET':

         
         duser = User.objects.get(username=doctorusername)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "rate":r} )








       
def  consult_a_doctor(request):


    if request.method == 'GET':

        
        doctortype = request.session['doctortype']
        print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)


        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj})

   


def  make_consultation(request, doctorusername):

    if request.method == 'POST':
       

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername


        diseaseinfo_id = request.session['diseaseinfo_id']
        diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, diseaseinfo=diseaseinfo_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")

         
        return redirect('consultationview',consultation_new.id)



def  consultationview(request,consultation_id):
   
    if request.method == 'GET':

   
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj })

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )





def rate_review(request,consultation_id):
   if request.method == "POST":
         
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         

         return redirect('consultationview',consultation_id)





def close_consultation(request,consultation_id):
   if request.method == "POST":
         
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         
         return redirect('home')






#-----------------------------chatting system ---------------------------------------------------


def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)

        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')



def chat_messages(request):
   if request.method == "GET":

         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})
