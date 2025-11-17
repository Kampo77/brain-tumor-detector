# 🧬 Tumor Detector Backend API

## Overview

A Django REST Framework API for medical image analysis. Currently provides placeholder predictions—real ML model integration coming later.

## ✅ Current Features

- ✅ **GET /api/ping/** - Health check endpoint
- ✅ **POST /api/analyze/** - Image upload and analysis (placeholder)
- ✅ Django REST Framework integration
- ✅ Configured for file uploads
- ✅ Error handling for missing files
- ✅ CORS-ready (configuration included)

## 🏗️ Backend Structure

```
backend/
├── manage.py                    # Django management script
├── venv/                        # Python virtual environment (Python 3.11)
├── db.sqlite3                   # SQLite database
├── tumor_detector/              # Django project folder
│   ├── settings.py              # ✅ Configured with DRF & media settings
│   ├── urls.py                  # ✅ Includes api.urls
│   ├── asgi.py
│   ├── wsgi.py
│   └── __init__.py
├── api/                         # Django app (REST endpoints)
│   ├── views.py                 # ✅ PingView & AnalyzeView implemented
│   ├── urls.py                  # ✅ Route definitions
│   ├── models.py                # (empty, ready for future models)
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   └── __init__.py
└── test_api_simple.py           # ✅ Automated test suite
```

## 🛠️ Setup on macOS

### 1. Activate Virtual Environment

```bash
cd /Users/kampo77/Desktop/rmt/backend
source venv/bin/activate
```

### 2. Run Migrations

```bash
python manage.py migrate
```

### 3. Start Development Server

```bash
python manage.py runserver 127.0.0.1:8000
```

You should see:
```
Starting development server at http://127.0.0.1:8000/
```

## 📡 API Endpoints

### Endpoint 1: Health Check

**Request:**
```http
GET /api/ping/
```

**Response (200 OK):**
```json
{
  "message": "API is working"
}
```

---

### Endpoint 2: Image Analysis

**Request:**
```http
POST /api/analyze/
Content-Type: multipart/form-data

file: <image_file>
```

**File types supported:**
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp` (standard images)
- `.dcm` (DICOM medical images)

**Success Response (200 OK):**
```json
{
  "result": "clean",
  "confidence": 0.99,
  "message": "Placeholder prediction. Real ML model will be integrated later."
}
```

**Error Response - No file (400 Bad Request):**
```json
{
  "error": "No file provided. Please upload an image with field name 'file'."
}
```

**Error Response - Invalid file type (400 Bad Request):**
```json
{
  "error": "Invalid file type. Please upload an image (jpg, png, gif, bmp) or DICOM file (dcm)."
}
```

---

## 🧪 Testing

### Option 1: Automated Tests (Recommended)

Run all tests at once:

```bash
cd /Users/kampo77/Desktop/rmt/backend
source venv/bin/activate
python test_api_simple.py
```

**Output:**
```
============================================================
🧪 TESTING TUMOR DETECTOR API ENDPOINTS
============================================================

1️⃣  Testing GET /api/ping/
------------------------------------------------------------
Status Code: 200
Response: {
  "message": "API is working"
}
✅ PING TEST PASSED!

2️⃣  Testing POST /api/analyze/ (without file - should fail)
------------------------------------------------------------
Status Code: 400
Response: {
  "error": "No file provided. Please upload an image with field name 'file'."
}
✅ ERROR HANDLING TEST PASSED!

3️⃣  Testing POST /api/analyze/ (with PNG image file)
------------------------------------------------------------
Status Code: 200
Response: {
  "result": "clean",
  "confidence": 0.99,
  "message": "Placeholder prediction. Real ML model will be integrated later."
}
✅ ANALYZE TEST PASSED!

============================================================
✅ ALL TESTS COMPLETED SUCCESSFULLY!
============================================================
```

### Option 2: cURL Commands (Manual Testing)

**Ping endpoint:**
```bash
curl -X GET http://127.0.0.1:8000/api/ping/
```

**Analyze endpoint (no file error):**
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/
```

**Analyze endpoint (with image):**
```bash
# Using any image file from your system
curl -X POST \
  -F "file=@/path/to/your/image.png" \
  http://127.0.0.1:8000/api/analyze/
```

### Option 3: Using httpie (More Readable)

Install httpie (if not already):
```bash
brew install httpie
```

**Ping:**
```bash
http GET http://127.0.0.1:8000/api/ping/
```

**Analyze (no file):**
```bash
http POST http://127.0.0.1:8000/api/analyze/
```

**Analyze (with file):**
```bash
http -f POST http://127.0.0.1:8000/api/analyze/ \
  file@/path/to/your/image.png
```

### Option 4: Using Python Requests

```python
import requests

# Ping
response = requests.get('http://127.0.0.1:8000/api/ping/')
print(response.json())

# Analyze
with open('/path/to/image.png', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://127.0.0.1:8000/api/analyze/', files=files)
    print(response.json())
```

---

## 🔧 Configuration Notes

### `settings.py` Changes Made:

1. **ALLOWED_HOSTS** - Set to `['*']` for development (restrict in production!)
   ```python
   ALLOWED_HOSTS = ['*']
   ```

2. **REST_FRAMEWORK** - Configured pagination
   ```python
   REST_FRAMEWORK = {
       'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
       'PAGE_SIZE': 100,
   }
   ```

3. **MEDIA Files** - Added for handling uploads
   ```python
   MEDIA_URL = 'media/'
   MEDIA_ROOT = BASE_DIR / 'media'
   ```

4. **INSTALLED_APPS** - Already includes:
   - `'rest_framework'` - DRF
   - `'api'` - Our custom app

### `urls.py` Changes Made:

Added include for API routes:
```python
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # ← This routes /api/* to our views
]
```

---

## 📝 Next Steps (Roadmap)

1. **Database Models** - Store analysis history
   - `AnalysisRecord` - timestamp, filename, uploaded_file, result, confidence
   
2. **Real ML Model Integration** - Call the trained model
   - Import from `model/` folder
   - Process DICOM/image files
   - Return real predictions

3. **User Authentication**
   - Add Django User model
   - Implement token-based auth (JWT)
   - Role-based access (doctors, admins, patients)

4. **Logging & Monitoring**
   - Structured logging
   - Error tracking
   - Performance metrics

5. **Frontend Integration**
   - CORS headers
   - WebSocket support (optional)

6. **Production Deployment**
   - Switch from SQLite to PostgreSQL
   - Use Gunicorn/uWSGI instead of `runserver`
   - Set secure headers
   - Configure HTTPS

---

## ⚙️ Django & DRF Versions

- Django: 5.2.8
- Django REST Framework: 3.16.1
- Python: 3.11

---

## 🐛 Common Issues & Solutions

### Issue: `ALLOWED_HOSTS` Error

**Error:**
```
Invalid HTTP_HOST header: 'testserver'. You may need to add 'testserver' to ALLOWED_HOSTS.
```

**Solution:**
Edit `tumor_detector/settings.py`:
```python
ALLOWED_HOSTS = ['*']  # Or specific hosts: ['localhost', '127.0.0.1', 'yourdomain.com']
```

---

### Issue: Port 8000 Already in Use

**Error:**
```
Error: That port is already in use.
```

**Solution:**
Use a different port:
```bash
python manage.py runserver 127.0.0.1:8001
```

Or kill the process:
```bash
pkill -f "manage.py runserver"
```

---

### Issue: `No module named 'rest_framework'`

**Error:**
```
ModuleNotFoundError: No module named 'rest_framework'
```

**Solution:**
Make sure venv is activated and DRF is installed:
```bash
source venv/bin/activate
pip list | grep djangorestframework
```

If not installed:
```bash
pip install djangorestframework
```

---

## 📚 Documentation Links

- [Django REST Framework Documentation](https://www.django-rest-framework.org/)
- [Django File Upload Documentation](https://docs.djangoproject.com/en/5.2/topics/files/uploads/)
- [macOS Python & Virtual Environments](https://docs.python.org/3/tutorial/venv.html)

---

## 👨‍💼 Questions?

This backend is ready for:
- ✅ Integration with your ML model
- ✅ Adding database models and authentication
- ✅ Frontend integration via the `/api/` routes
- ✅ Deployment to production

Next step: Integrate the trained model from the `model/` folder!

