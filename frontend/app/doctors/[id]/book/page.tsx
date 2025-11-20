'use client';

import { useState, useEffect } from 'react';
import { useRouter, useParams } from 'next/navigation';

interface Doctor {
  id: number;
  name: string;
  specialization: string;
  experience: number;
  rating: number;
  education: string;
  available_days: string[];
  image?: string;
}

const doctors: Doctor[] = [
  {
    id: 1,
    name: "Dr. Sarah Johnson",
    specialization: "Neurosurgeon",
    experience: 15,
    rating: 4.9,
    education: "Harvard Medical School",
    available_days: ["Monday", "Wednesday", "Friday"],
    image: "👩‍⚕️"
  },
  {
    id: 2,
    name: "Dr. Michael Chen",
    specialization: "Neuro-Oncologist",
    experience: 12,
    rating: 4.8,
    education: "Johns Hopkins University",
    available_days: ["Tuesday", "Thursday", "Saturday"],
    image: "👨‍⚕️"
  },
  {
    id: 3,
    name: "Dr. Emily Rodriguez",
    specialization: "Radiologist",
    experience: 10,
    rating: 4.7,
    education: "Stanford University",
    available_days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    image: "👩‍⚕️"
  },
  {
    id: 4,
    name: "Dr. James Wilson",
    specialization: "Neurologist",
    experience: 18,
    rating: 4.9,
    education: "Mayo Clinic",
    available_days: ["Monday", "Wednesday", "Friday"],
    image: "👨‍⚕️"
  },
  {
    id: 5,
    name: "Dr. Lisa Anderson",
    specialization: "Neurosurgeon",
    experience: 14,
    rating: 4.8,
    education: "Yale School of Medicine",
    available_days: ["Tuesday", "Thursday", "Saturday"],
    image: "👩‍⚕️"
  },
  {
    id: 6,
    name: "Dr. David Kim",
    specialization: "Neuro-Oncologist",
    experience: 11,
    rating: 4.7,
    education: "University of California, San Francisco",
    available_days: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    image: "👨‍⚕️"
  }
];

const timeSlots = [
  "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
  "14:00", "14:30", "15:00", "15:30", "16:00", "16:30", "17:00"
];

export default function BookAppointmentPage() {
  const router = useRouter();
  const params = useParams();
  const doctorId = parseInt(params.id as string);
  
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [formData, setFormData] = useState({
    patientName: '',
    email: '',
    phone: '',
    date: '',
    time: '',
    notes: ''
  });
  const [submitting, setSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    const foundDoctor = doctors.find(d => d.id === doctorId);
    setDoctor(foundDoctor || null);
  }, [doctorId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    try {
      const response = await fetch('http://127.0.0.1:8000/api/appointments/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          doctor_id: doctorId,
          doctor_name: doctor?.name,
          patient_name: formData.patientName,
          email: formData.email,
          phone: formData.phone,
          appointment_date: formData.date,
          appointment_time: formData.time,
          notes: formData.notes,
        }),
      });

      if (response.ok) {
        setSuccess(true);
        setTimeout(() => router.push('/'), 3000);
      }
    } catch (error) {
      console.error('Error booking appointment:', error);
    } finally {
      setSubmitting(false);
    }
  };

  if (!doctor) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white flex items-center justify-center">
        <p className="text-xl">Doctor not found</p>
      </div>
    );
  }

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">✅</div>
          <h2 className="text-3xl font-bold mb-2">Appointment Confirmed!</h2>
          <p className="text-gray-400 mb-4">
            Your appointment with {doctor.name} has been scheduled.
          </p>
          <p className="text-sm text-gray-500">Redirecting to home page...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.push('/doctors')}
            className="mb-4 text-blue-400 hover:text-blue-300 flex items-center gap-2 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Doctors
          </button>
          
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
            Book Appointment
          </h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Doctor Info Card */}
          <div className="lg:col-span-1">
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 sticky top-8">
              <div className="text-6xl mb-4 text-center">{doctor.image}</div>
              <h3 className="text-xl font-bold text-blue-300 mb-1 text-center">{doctor.name}</h3>
              <p className="text-purple-400 font-medium mb-4 text-center">{doctor.specialization}</p>
              
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Experience:</span>
                  <span className="text-white font-semibold">{doctor.experience} years</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Rating:</span>
                  <span className="text-yellow-400 font-semibold">⭐ {doctor.rating}</span>
                </div>
              </div>

              <div>
                <p className="text-xs text-gray-400 mb-2">Available Days:</p>
                <div className="flex flex-wrap gap-1">
                  {doctor.available_days.map(day => (
                    <span
                      key={day}
                      className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded border border-green-900/50"
                    >
                      {day}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Booking Form */}
          <div className="lg:col-span-2">
            <form onSubmit={handleSubmit} className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50">
              <h2 className="text-2xl font-bold text-blue-300 mb-6">Patient Information</h2>

              {/* Patient Name */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Full Name *
                </label>
                <input
                  type="text"
                  required
                  value={formData.patientName}
                  onChange={(e) => setFormData({...formData, patientName: e.target.value})}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="John Doe"
                />
              </div>

              {/* Email */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Email *
                </label>
                <input
                  type="email"
                  required
                  value={formData.email}
                  onChange={(e) => setFormData({...formData, email: e.target.value})}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="john@example.com"
                />
              </div>

              {/* Phone */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Phone Number *
                </label>
                <input
                  type="tel"
                  required
                  value={formData.phone}
                  onChange={(e) => setFormData({...formData, phone: e.target.value})}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="+7 777 123 4567"
                />
              </div>

              {/* Date and Time */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Date *
                  </label>
                  <input
                    type="date"
                    required
                    value={formData.date}
                    onChange={(e) => setFormData({...formData, date: e.target.value})}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Time *
                  </label>
                  <select
                    required
                    value={formData.time}
                    onChange={(e) => setFormData({...formData, time: e.target.value})}
                    className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors"
                  >
                    <option value="">Select time</option>
                    {timeSlots.map(slot => (
                      <option key={slot} value={slot}>{slot}</option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Notes */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Additional Notes (Optional)
                </label>
                <textarea
                  value={formData.notes}
                  onChange={(e) => setFormData({...formData, notes: e.target.value})}
                  rows={4}
                  className="w-full px-4 py-3 bg-slate-900/50 border border-slate-700 rounded-lg text-white focus:outline-none focus:border-blue-500 transition-colors resize-none"
                  placeholder="Any specific concerns or information the doctor should know..."
                />
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={submitting}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-xl transition-all shadow-lg hover:shadow-blue-500/50 flex items-center justify-center gap-2"
              >
                {submitting ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Booking...</span>
                  </>
                ) : (
                  <>
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span>Confirm Appointment</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
