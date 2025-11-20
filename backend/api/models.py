from django.db import models

# Create your models here.

class Appointment(models.Model):
    """Model for storing patient appointments with doctors"""
    doctor_id = models.IntegerField()
    doctor_name = models.CharField(max_length=255)
    patient_name = models.CharField(max_length=255)
    email = models.EmailField()
    phone = models.CharField(max_length=50)
    appointment_date = models.DateField()
    appointment_time = models.CharField(max_length=10)
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('confirmed', 'Confirmed'),
            ('cancelled', 'Cancelled'),
            ('completed', 'Completed')
        ],
        default='pending'
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.patient_name} - {self.doctor_name} - {self.appointment_date} {self.appointment_time}"
