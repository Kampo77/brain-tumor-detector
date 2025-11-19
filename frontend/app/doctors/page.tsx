'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

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

export default function DoctorsPage() {
  const router = useRouter();
  const [selectedSpecialization, setSelectedSpecialization] = useState<string>('All');

  const specializations = ['All', ...Array.from(new Set(doctors.map(d => d.specialization)))];

  const filteredDoctors = selectedSpecialization === 'All' 
    ? doctors 
    : doctors.filter(d => d.specialization === selectedSpecialization);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => router.push('/')}
            className="mb-4 text-blue-400 hover:text-blue-300 flex items-center gap-2 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Analysis
          </button>
          
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
            Our Specialists
          </h1>
          <p className="text-gray-400">Choose a doctor to schedule your appointment</p>
        </div>

        {/* Filter */}
        <div className="mb-6 flex flex-wrap gap-2">
          {specializations.map(spec => (
            <button
              key={spec}
              onClick={() => setSelectedSpecialization(spec)}
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                selectedSpecialization === spec
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-slate-800/50 text-gray-400 hover:bg-slate-700/50'
              }`}
            >
              {spec}
            </button>
          ))}
        </div>

        {/* Doctors Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredDoctors.map(doctor => (
            <div
              key={doctor.id}
              className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-6 border border-slate-700/50 hover:border-blue-500/50 transition-all hover:shadow-xl hover:shadow-blue-500/10"
            >
              {/* Doctor Image */}
              <div className="text-6xl mb-4 text-center">{doctor.image}</div>

              {/* Doctor Info */}
              <h3 className="text-xl font-bold text-blue-300 mb-1">{doctor.name}</h3>
              <p className="text-purple-400 font-medium mb-3">{doctor.specialization}</p>

              {/* Stats */}
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Experience:</span>
                  <span className="text-white font-semibold">{doctor.experience} years</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">Rating:</span>
                  <span className="text-yellow-400 font-semibold flex items-center gap-1">
                    ⭐ {doctor.rating}
                  </span>
                </div>
                <div className="text-sm">
                  <span className="text-gray-400">Education:</span>
                  <p className="text-white text-xs mt-1">{doctor.education}</p>
                </div>
              </div>

              {/* Available Days */}
              <div className="mb-4">
                <p className="text-xs text-gray-400 mb-2">Available:</p>
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

              {/* Book Button */}
              <button
                onClick={() => router.push(`/doctors/${doctor.id}/book`)}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-bold py-3 px-4 rounded-lg transition-all shadow-lg hover:shadow-blue-500/50"
              >
                Schedule Appointment
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
