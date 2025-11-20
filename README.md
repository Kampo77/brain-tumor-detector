# 🧠 Brain Tumor Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.8-green.svg)](https://www.djangoproject.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Advanced web-based system for brain tumor detection and medical appointment scheduling using deep learning models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements a comprehensive brain tumor detection system that combines:
- **2D CNN (ResNet-18)** for rapid MRI scan classification
- **3D CNN (U-Net)** for precise tumor segmentation on BraTS dataset
- **Medical appointment system** for patient-doctor scheduling
- **Modern web interface** built with Next.js and Django REST API

The system provides both quick screening (2D analysis) and detailed diagnostic insights (3D segmentation with anatomical localization), making it suitable for clinical workflows.

---

## ✨ Features

### 🔬 Medical Analysis

#### 2D Analysis (Quick Screening)
- Upload JPG/PNG MRI scans
- Binary classification: Tumor / No Tumor
- ResNet-18 with ImageNet transfer learning
- ~98% accuracy
- Results in < 1 second

#### 3D Analysis (Detailed Diagnostics)
- Upload ZIP archives with 4 NIfTI modalities (FLAIR, T1, T1CE, T2)
- 3D U-Net volumetric segmentation
- **Results:**
  - Tumor fraction (% of brain volume affected)
  - Confidence score
  - Brain region localization (hemisphere, lobe, level)
  - **3 orthogonal views**: Axial, Coronal, Sagittal
  - High-resolution visualization (1024×1024 px)
  - Red tumor mask with yellow contour overlay

#### Advanced Visualization
- Smart slice selection (displays slice with maximum tumor)
- Percentile-based contrast enhancement
- Click-to-zoom fullscreen modal
- Three-plane anatomical views (like 3D Slicer)

### 📅 Appointment System
- Triggered when tumor detected
- 6 medical specialists:
  - Neurosurgeons
  - Neuro-Oncologists
  - Radiologists
  - Neurologists
- Booking form with date/time selection
- Database storage with Django ORM
- Admin panel for appointment management

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ ImageUpload  │  │BraTS3DUpload │  │   Doctors    │      │
│  │   (2D CNN)   │  │   (3D CNN)   │  │    Booking   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           ↓ HTTP/JSON ↓
┌─────────────────────────────────────────────────────────────┐
│                   Backend (Django REST API)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  PredictView │  │ BraTSPredict │  │ Appointment  │      │
│  │   ResNet-18  │  │   3D U-Net   │  │     View     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                           ↓ ORM ↓
┌─────────────────────────────────────────────────────────────┐
│                      SQLite Database                         │
│                  (Appointment records)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies

### Backend
- **Python 3.10+**
- **Django 5.2.8** - Web framework
- **Django REST Framework** - API
- **PyTorch 2.5.1** - Deep learning
- **nibabel 5.1.0** - NIfTI file processing
- **scipy 1.11.4** - 3D image processing
- **Pillow** - Image manipulation

### Frontend
- **Next.js 14+** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client

### Models
- **ResNet-18** (2D) - 11M parameters, ImageNet pretrained
- **3D U-Net** (3D) - 5M parameters, trained on BraTS 2018

### Dataset
- **2D:** Brain MRI Images (Kaggle) - 3000 images
- **3D:** BraTS 2018 - 285 subjects (210 HGG + 75 LGG)

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- GPU recommended (4GB+ VRAM) for 3D model training

### Backend Setup

```bash
# Clone repository
git clone https://github.com/Kampo77/rmt.git
cd rmt/backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser for admin panel
python manage.py createsuperuser

# Download trained models (if not included)
# Place models in backend/model/:
#   - model_weights.pth (ResNet-18)
#   - brats2020_unet3d.pth (3D U-Net)

# Start server
python manage.py runserver
```

Backend will be available at: `http://127.0.0.1:8000`

### Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: `http://localhost:3000`

---

## 🚀 Usage

### 1. Quick 2D Screening

1. Open `http://localhost:3000`
2. Upload JPG/PNG brain MRI scan
3. Get instant Tumor/No Tumor classification
4. View confidence score

### 2. Detailed 3D Analysis

1. Click "Upload 3D BraTS Volume"
2. Upload ZIP with 4 NIfTI files:
   - `*_flair.nii.gz`
   - `*_t1.nii.gz`
   - `*_t1ce.nii.gz`
   - `*_t2.nii.gz`
3. Wait ~1-2 minutes for analysis
4. View results:
   - Tumor fraction percentage
   - Confidence score
   - Anatomical location (e.g., "Left Frontal (Middle)")
   - 3 orthogonal segmentation views
   - Click images to zoom

### 3. Book Appointment

1. If tumor detected, click "Schedule Appointment"
2. Browse available doctors
3. Select specialist
4. Fill booking form (name, email, phone, date/time)
5. Submit appointment

### 4. Admin Panel

Access Django admin at `http://127.0.0.1:8000/admin/`
- View all appointments
- Manage bookings
- Export data

---

## 📡 API Documentation

### Base URL
```
http://127.0.0.1:8000/api/
```

### Endpoints

#### Health Check
```http
GET /api/ping/
```

#### 2D Prediction
```http
POST /api/predict/
Content-Type: multipart/form-data

Parameters:
  - image: JPG/PNG file

Response:
{
  "success": true,
  "prediction": "tumor",
  "confidence": 0.873
}
```

#### 3D BraTS Prediction
```http
POST /api/brats/predict/
Content-Type: multipart/form-data

Parameters:
  - file: ZIP with NIfTI files

Response:
{
  "success": true,
  "has_tumor": true,
  "tumor_fraction": 0.0206,
  "confidence": 0.979,
  "subject_id": "Brats18_2013_3_1",
  "overlay_axial": "data:image/png;base64,...",
  "overlay_coronal": "data:image/png;base64,...",
  "overlay_sagittal": "data:image/png;base64,...",
  "brain_region": "Right Parietal/Central (Middle)"
}
```

#### Create Appointment
```http
POST /api/appointments/
Content-Type: application/json

Body:
{
  "doctor_id": 1,
  "doctor_name": "Dr. Sarah Johnson",
  "patient_name": "John Doe",
  "email": "john@example.com",
  "phone": "+1234567890",
  "appointment_date": "2025-12-01",
  "appointment_time": "14:30",
  "notes": "Headaches for 2 weeks"
}

Response:
{
  "success": true,
  "appointment_id": 1,
  "status": "confirmed"
}
```

#### Get Appointments
```http
GET /api/appointments/
```

---

## 📊 Model Performance

### 2D ResNet-18 (Classification)
- **Accuracy:** 98.2%
- **Sensitivity:** 99%
- **Specificity:** 98%
- **Precision:** 97%
- **Dataset:** 3000 MRI slices

### 3D U-Net (Segmentation)
- **Dice Score:** 0.8313 (83.13%)
- **IoU:** 0.7244 (72.44%)
- **Precision:** 0.7853 (78.53%)
- **Recall:** 0.9169 (91.69%)
- **Accuracy:** 0.9960 (99.60%)
- **Dataset:** BraTS 2018 (285 patients)
- **Training:** 25 epochs, ~12 hours on GTX 1650

**Interpretation:**
- Dice 0.83 = Excellent segmentation quality
- High Recall (91.69%) = Finds almost all tumor tissue (critical for medical use)
- Comparable to clinical tools like 3D Slicer

---

## 📂 Project Structure

```
rmt/
├── backend/
│   ├── api/
│   │   ├── views.py              # 2D classification endpoint
│   │   ├── brats_views.py        # 3D segmentation endpoint
│   │   ├── brats_utils.py        # 3-view overlay generation
│   │   ├── models.py             # Appointment database model
│   │   ├── admin.py              # Django admin config
│   │   └── urls.py               # API routes
│   ├── model/
│   │   ├── model_weights.pth     # ResNet-18 weights (2D)
│   │   ├── brats2020_unet3d.pth  # 3D U-Net weights
│   │   └── model.py              # Model architectures
│   ├── tumor_detector/
│   │   ├── settings.py           # Django settings
│   │   └── urls.py               # Main URL config
│   ├── db.sqlite3                # SQLite database
│   ├── manage.py                 # Django management
│   └── requirements.txt          # Python dependencies
├── frontend/
│   ├── app/
│   │   ├── page.tsx              # Home page
│   │   ├── layout.tsx            # Root layout
│   │   └── doctors/
│   │       ├── page.tsx          # Doctors list
│   │       └── [id]/book/
│   │           └── page.tsx      # Booking form
│   ├── components/
│   │   ├── ImageUpload.tsx       # 2D upload component
│   │   └── BraTS3DUpload.tsx     # 3D upload component
│   ├── package.json              # Node dependencies
│   └── tsconfig.json             # TypeScript config
├── brats_pipeline.py             # 3D U-Net inference
├── train_brats_3d.py             # 3D model training script
├── evaluate_model.py             # Model evaluation metrics
└── README.md                     # This file
```

---

## 👥 Team

This project was developed by a team of students from **Astana IT University** under the supervision of **Seitenov Altynbek, Senior-lecturer**.

### Team Members

| Name | GitHub | Role |
|------|--------|------|
| **Yerassyl Salimgerey** | [@Kampo77](https://github.com/Kampo77) | Team Lead, 3D Model Development |
| **Amankeldi Zhanatov** | [@Zhandos001w](https://github.com/Zhandos001w) | Backend Development, API Integration |
| **Lada Mulkulanova** | [@mestriw](https://github.com/mestriw) | Frontend Development, UI/UX Design |
| **Aruzhan Zhuanysh** | [@Aruzhan Zhuanysh](https://github.com/Aruzhan-Zhuanysh) | Data Processing, Model Training |
| **Berdiyar Akbergen** | [@AkBexGod](https://github.com/AkBexGod) | Documentation, Testing |

**Supervisor:** Seitenov Altynbek, Senior-lecturer, Astana IT University

---

## 📄 Paper Reference

This project implements concepts from:

**Paper Title:** "Brain Tumor Detection System Using 2D and 3D Deep Learning Models"

**Abstract:** We propose a comprehensive brain tumor detection system that combines a 2D convolutional neural network (ResNet-18) for image-level tumor classification with a 3D CNN (U-Net) for volumetric tumor segmentation and localization. The system achieves ~98% accuracy in 2D classification and Dice score of 0.83 in 3D segmentation.

---

## 🔮 Future Improvements

- [ ] Multi-modal ensemble (combine all 4 MRI sequences more effectively)
- [ ] Attention mechanisms for U-Net (Attention U-Net)
- [ ] Additional training epochs (25 → 50+) to reach Dice 0.87+
- [ ] Real-time segmentation updates
- [ ] Export segmentation masks (NIfTI format)
- [ ] Integration with PACS systems
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Patient history tracking

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **BraTS Challenge** for providing the dataset
- **Kaggle** for Brain MRI Images dataset
- **3D Slicer** team for medical imaging insights
- **Astana IT University** for support and resources

---

## 📧 Contact

For questions or collaboration:
- **Project Repository:** [https://github.com/Kampo77/rmt](https://github.com/Kampo77/rmt)
- **Issues:** [https://github.com/Kampo77/rmt/issues](https://github.com/Kampo77/rmt/issues)

---

## 📸 Screenshots

### Home Page - 3D Analysis
![3D Analysis](docs/screenshots/3d_analysis.png)

### Segmentation Results (3 Views)
![Segmentation Views](docs/screenshots/segmentation_views.png)

### Doctors Booking
![Doctors Page](docs/screenshots/doctors_page.png)

### Admin Panel
![Admin Panel](docs/screenshots/admin_panel.png)

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Kampo77/rmt&type=Date)](https://star-history.com/#Kampo77/rmt&Date)

---

**Built with ❤️ by Astana IT University Team**

**© 2025 Brain Tumor Detection System. All rights reserved.**
