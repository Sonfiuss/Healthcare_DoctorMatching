from django.contrib import admin
from models import patient, doctor, diseaseinfo,
# Register your models here.
admin.site.register(patient)
admin.site.register(doctor)
admin.site.register(diseaseinfo)
admin.site.register(consultation)
admin.site.register(rating_review)