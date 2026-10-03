from django.db import models
from django.contrib import admin

class Analyser(models.Model):
    # Define your fields here
    title = models.CharField(max_length=100)