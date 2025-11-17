# Medical Image Analyzer - Frontend Integration

## 🎯 Project Overview

This is a full-stack medical imaging application built with:
- **Frontend:** Next.js 16 + React 19 + TypeScript + Tailwind CSS
- **Backend:** Django 5.2 + Django REST Framework
- **Purpose:** AI-powered tumor detection from CT/MRI images

The frontend provides a clean, user-friendly interface for uploading medical images and receiving AI analysis results.

---

## 📁 Project Structure

```
rmt/
│
├── 📖 QUICK_SETUP.md              ← Start here! Quick checklist
├── 📖 INTEGRATION_GUIDE.md         ← Detailed integration guide
│
├── backend/                        ← Django REST API
│   ├── tumor_detector/
│   │   ├── settings.py             ✨ CORS configured
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── api/
│   │   ├── views.py                (AnalyzeView, PingView)
│   │   ├── urls.py
│   │   └── models.py
│   ├── manage.py
│   ├── db.sqlite3
│   └── venv/
│
└── frontend/                       ← Next.js Application
    ├── app/
    │   ├── page.tsx                ✨ Updated with ImageUpload
    │   ├── layout.tsx
    │   ├── globals.css
    │   └── favicon.ico
    ├── components/
    │   ├── ImageUpload.tsx         ✨ NEW - Main upload component
    │   └── ImageUploadAxios.tsx    ✨ NEW - Alternative (requires axios)
    ├── public/
    ├── api-test.js                 ✨ NEW - Browser testing utilities
    ├── .env.example                ✨ NEW - Environment template
    ├── package.json
    ├── tsconfig.json
    ├── tailwind.config.ts
    ├── next.config.ts
    └── eslint.config.mjs
```

---

## ✨ What's New

### Frontend Components

#### 1. **`components/ImageUpload.tsx`** (Primary Component)
A production-ready image upload component with:
- ✅ Drag & drop support
- ✅ File validation (type, size)
- ✅ Image preview
- ✅ Loading states with spinner
- ✅ Error handling
- ✅ Results display with confidence bar
- ✅ Callback function for parent integration
- ✅ Supports: JPG, PNG, GIF, BMP, DICOM (max 50MB)

**Key Props:**
```tsx
<ImageUpload
  onAnalysisComplete={(result) => console.log(result)}
  apiBaseUrl="http://127.0.0.1:8000"
/>
```

#### 2. **`components/ImageUploadAxios.tsx`** (Alternative)
Same functionality as above, but using Axios instead of native fetch.
Requires: `npm install axios`

#### 3. **`app/page.tsx`** (Main Page)
Updated landing page featuring:
- Header with branding
- ImageUpload component integration
- Info card explaining the workflow
- Recent analyses history
- Footer with version info
- Responsive grid layout (mobile + desktop)

### Supporting Files

#### 4. **`api-test.js`**
Browser console utilities for testing:
- `testPing()` - Test `/ping/` endpoint
- `testAnalyze(file)` - Test `/analyze/` with file
- `testCors()` - Check CORS headers
- `runAllTests()` - Run all tests

Usage in browser console:
```javascript
// Load file
<script src="api-test.js"></script>

// Run tests
testPing()
testAnalyzeInteractive()
```

#### 5. **`.env.example`**
Environment variables template:
```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### Backend Updates

#### 6. **CORS Configuration**
`tumor_detector/settings.py` now includes:
- `django-cors-headers` package support
- Allowed origins for `localhost:3000`
- Proper CORS middleware placement
- Allowed methods and headers configuration

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (backend)
- Node.js 18+ (frontend)
- macOS/Linux/Windows with zsh or bash

### Step 1: Backend Setup
```bash
cd backend

# Install CORS package
pip install django-cors-headers

# Start server
python manage.py runserver
```
✅ Backend running at `http://127.0.0.1:8000`

### Step 2: Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
✅ Frontend running at `http://localhost:3000`

### Step 3: Test Integration
1. Open browser to `http://localhost:3000`
2. Upload an image
3. Click "Analyze Image"
4. See results within seconds

---

## 🔌 API Integration Details

### Frontend Request Flow
```
User Selects File
    ↓
Component Validates (type, size)
    ↓
User Clicks "Analyze"
    ↓
POST /analyze/ with FormData
    ↓
Backend Processes (ML model placeholder)
    ↓
Returns: {result, confidence, message}
    ↓
Display Result with Confidence Bar
```

### HTTP Request
```typescript
const formData = new FormData();
formData.append('file', file);

const response = await fetch('http://127.0.0.1:8000/analyze/', {
  method: 'POST',
  body: formData,
  // Note: Don't set Content-Type; browser handles it
});

const data = await response.json();
// data = { result: "clean", confidence: 0.99, message: "..." }
```

### Error Handling
The component gracefully handles:
- ❌ CORS errors
- ❌ Network unavailable
- ❌ Invalid file types
- ❌ File too large
- ❌ Backend errors (400, 404, 500)
- ❌ Request timeouts

All errors display user-friendly messages.

---

## 📦 Dependencies

### Backend
```
Django==5.2.8
djangorestframework==3.15.0
django-cors-headers==4.3.1  ← ADDED
```

### Frontend
```
Next.js 16.0.3
React 19.2.0
React DOM 19.2.0
TypeScript 5
Tailwind CSS 4
```

### Optional Frontend
```
axios (for ImageUploadAxios component)
```

---

## 🧪 Testing

### Test 1: Backend Health Check
```bash
curl http://127.0.0.1:8000/ping/
# Response: {"message": "API is working"}
```

### Test 2: Frontend Health
Visit `http://localhost:3000` in browser

### Test 3: File Upload (Browser Console)
```javascript
// After loading api-test.js
testPing()                    // Test backend connection
testAnalyzeInteractive()      // Interactive file upload
```

### Test 4: cURL Upload
```bash
curl -X POST -F "file=@image.jpg" \
  http://127.0.0.1:8000/analyze/
```

---

## 🎨 UI Features

### Image Upload Component
- Clean, modern design with Tailwind CSS
- Large drop zone with visual feedback
- File type and size validation with helpful error messages
- Image preview (before upload)
- Loading spinner during analysis
- Success display with:
  - Status badge (CLEAN/TUMOR)
  - Confidence percentage
  - Visual progress bar
  - Optional message from backend

### Main Page Layout
- Responsive 3-column grid (1 column on mobile)
- Header with branding
- Main upload section (2 columns)
- Sidebar with info + history (1 column)
- Recent analyses history (scrollable)
- Professional color scheme (blue, gray, green, red)

---

## 🔒 Security Notes

### Current Configuration (Development)
- ✅ CORS enabled for localhost only
- ✅ DEBUG = True (Django development)
- ✅ ALLOWED_HOSTS = '*' (OK for development)
- ⚠️ SECRET_KEY visible in settings (don't commit to production!)

### Before Production
- 🔐 Set DEBUG = False
- 🔐 Generate new SECRET_KEY
- 🔐 Restrict ALLOWED_HOSTS to your domain
- 🔐 Update CORS_ALLOWED_ORIGINS to production domain
- 🔐 Use environment variables for secrets
- 🔐 Add authentication (JWT, OAuth2)
- 🔐 Validate file uploads more strictly
- 🔐 Add rate limiting

---

## 🎯 Usage Examples

### Basic Integration
```tsx
'use client';
import ImageUpload from '@/components/ImageUpload';

export default function UploadPage() {
  return (
    <ImageUpload
      onAnalysisComplete={(result) => {
        console.log('Analysis complete:', result);
      }}
      apiBaseUrl="http://127.0.0.1:8000"
    />
  );
}
```

### With Custom Error Handling
```tsx
const [error, setError] = useState<string | null>(null);

const handleAnalysis = (result: AnalysisResult) => {
  if (result.result === 'tumor') {
    setError('⚠️ Tumor detected. Please consult a medical professional.');
  } else {
    setError(null);
  }
};

<ImageUpload onAnalysisComplete={handleAnalysis} />
```

### Using Axios Alternative
```bash
npm install axios
```

```tsx
import ImageUploadAxios from '@/components/ImageUploadAxios';

// Use same way as ImageUpload component
<ImageUploadAxios onAnalysisComplete={handleAnalysis} />
```

---

## 🚨 Troubleshooting

### CORS Error
```
Access to XMLHttpRequest at 'http://127.0.0.1:8000/analyze/'
from origin 'http://localhost:3000' has been blocked by CORS policy
```

**Fix:**
1. Verify `django-cors-headers` installed: `pip list | grep django-cors`
2. Verify `corsheaders` in `INSTALLED_APPS`
3. Verify middleware configuration
4. Restart Django: `python manage.py runserver`
5. Clear browser cache (Cmd+Shift+R on Mac)

### Backend Not Responding
**Error:** "Failed to analyze image: ... Make sure the backend is running"

**Fix:**
```bash
# Check if running on port 8000
lsof -i :8000

# If occupied, run on different port
python manage.py runserver 8001
```

### File Upload Fails
**Error:** "Invalid file type" or similar

**Check:**
- File has valid extension (.jpg, .png, .dcm, etc.)
- File size < 50MB
- Backend is running and responding

---

## 📈 Next Steps & Enhancements

### Immediate (Week 1)
- [ ] Test with real medical images
- [ ] Integrate real ML model in `api/views.py`
- [ ] Add unit tests
- [ ] Set up CI/CD pipeline

### Short-term (Week 2-3)
- [ ] Add user authentication (JWT tokens)
- [ ] Save analysis history to database
- [ ] Create analysis detail page
- [ ] Add export functionality (PDF, CSV)

### Medium-term (Month 1-2)
- [ ] Display heatmaps from model predictions
- [ ] Batch upload functionality
- [ ] Advanced filtering in history
- [ ] Performance optimizations

### Long-term (Month 3+)
- [ ] Mobile app (React Native)
- [ ] Real-time collaboration
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

---

## 📚 Documentation Files

- **QUICK_SETUP.md** - Step-by-step setup checklist
- **INTEGRATION_GUIDE.md** - Detailed integration & troubleshooting
- **api-test.js** - Browser testing utilities
- **This file** - Project overview and reference

---

## 🤝 Contributing

When making changes:
1. Keep components reusable and well-documented
2. Follow TypeScript best practices
3. Test across desktop and mobile
4. Update documentation accordingly
5. Test CORS configuration after backend changes

---

## 📞 Support

For issues or questions:
1. Check `QUICK_SETUP.md` - most common issues covered
2. Review `INTEGRATION_GUIDE.md` for detailed explanations
3. Check browser console for error messages
4. Test backend directly with `curl` or browser
5. Use `api-test.js` utilities for quick testing

---

## 🎉 You're All Set!

Your Next.js frontend is ready to integrate with the Django backend. Follow the **QUICK_SETUP.md** for immediate start, or read **INTEGRATION_GUIDE.md** for detailed understanding.

**Happy coding!** 🚀
