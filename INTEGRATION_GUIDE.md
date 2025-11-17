# Frontend-Backend Integration Guide

## Overview
This guide walks you through the complete setup of the Next.js frontend integrated with the Django REST backend for the Medical Image Analyzer.

---

## Part 1: Backend Setup (Django)

### 1.1 Install CORS Support
The backend now uses `django-cors-headers` to allow cross-origin requests from the Next.js frontend.

```bash
cd backend
pip install django-cors-headers
```

### 1.2 Verify Django Settings
The `settings.py` has been updated with:
- `corsheaders` added to `INSTALLED_APPS`
- `CorsMiddleware` added to `MIDDLEWARE` (must be near the top)
- CORS allowed origins configured for `localhost:3000` and `127.0.0.1:3000`

The configuration allows:
- POST, GET, OPTIONS, PATCH, PUT, DELETE methods
- Common headers like `content-type`, `authorization`, `x-csrftoken`, etc.
- Credentials to be sent with requests

### 1.3 Run the Backend
```bash
cd backend
python manage.py runserver
```

The backend will run at `http://127.0.0.1:8000`.

**Check health:** Visit `http://127.0.0.1:8000/ping/` in your browser. You should see:
```json
{"message": "API is working"}
```

---

## Part 2: Frontend Setup (Next.js)

### 2.1 Install Dependencies
If you haven't already, install axios (optional, the code uses native `fetch`):

```bash
cd frontend
npm install
# Optional: npm install axios
```

### 2.2 Project Structure
New/updated files:

```
frontend/
├── app/
│   ├── page.tsx          (Main page - updated with ImageUpload integration)
│   ├── globals.css
│   └── layout.tsx
├── components/
│   └── ImageUpload.tsx   (NEW - Reusable upload component)
├── package.json
├── tsconfig.json
└── tailwind.config.ts
```

### 2.3 Component: `ImageUpload.tsx`
This is the core component handling:
- File upload via click or drag-and-drop
- File validation (type, size)
- Image preview
- POST request to `/analyze/`
- Loading states
- Error handling
- Result display with confidence score

**Key Features:**
- Supports: JPG, PNG, GIF, BMP, DICOM (max 50MB)
- Error messages for invalid files
- Spinning loader during analysis
- Visual confidence bar
- Callback function for parent component

### 2.4 Page: `app/page.tsx`
The main page component includes:
- Header with branding
- `ImageUpload` component integration
- Info card ("How it works")
- Analysis history (recent analyses logged)
- Footer with version info
- Responsive grid layout

### 2.5 Run the Frontend
```bash
cd frontend
npm run dev
```

The frontend will run at `http://localhost:3000`.

---

## Part 3: Testing the Integration

### 3.1 Check Both Servers Are Running

**Backend:**
```bash
curl http://127.0.0.1:8000/ping/
# Expected response: {"message": "API is working"}
```

**Frontend:**
- Visit `http://localhost:3000` in your browser

### 3.2 Test File Upload
1. Go to `http://localhost:3000`
2. Click on the upload area or drag & drop an image
3. Select a CT/MRI image (JPG, PNG, GIF, BMP, or DICOM)
4. Click "Analyze Image"
5. The component will:
   - Show a loading spinner
   - Send the file to `POST http://127.0.0.1:8000/analyze/`
   - Display the result (e.g., `"clean"` or `"tumor"` with confidence)

### 3.3 Test with cURL (from backend directory)
```bash
curl -X POST -F "file=@/path/to/image.jpg" http://127.0.0.1:8000/analyze/
```

Expected response:
```json
{
  "result": "clean",
  "confidence": 0.99,
  "message": "Placeholder prediction. Real ML model will be integrated later."
}
```

---

## Part 4: API Endpoint Reference

### Endpoint: `/analyze/`
- **Method:** POST
- **URL:** `http://127.0.0.1:8000/analyze/`
- **Request:**
  - Content-Type: `multipart/form-data`
  - Field name: `file`
  - File types: JPG, PNG, GIF, BMP, DICOM
  - Max size: No hard limit in backend (adjust if needed)

- **Response (Success - 200):**
  ```json
  {
    "result": "clean",
    "confidence": 0.99,
    "message": "Placeholder prediction. Real ML model will be integrated later."
  }
  ```
  OR (with tumor)
  ```json
  {
    "result": "tumor",
    "confidence": 0.87
  }
  ```

- **Response (Error - 400):**
  ```json
  {
    "error": "No file provided. Please upload an image with field name 'file'."
  }
  ```
  OR
  ```json
  {
    "error": "Invalid file type. Please upload an image (jpg, png, gif, bmp) or DICOM file (dcm)."
  }
  ```

### Endpoint: `/ping/`
- **Method:** GET
- **URL:** `http://127.0.0.1:8000/ping/`
- **Response:**
  ```json
  {"message": "API is working"}
  ```

---

## Part 5: Frontend Code Examples

### Using the ImageUpload Component

```tsx
import ImageUpload from '@/components/ImageUpload';

export default function MyPage() {
  const handleResult = (result: AnalysisResult) => {
    console.log('Analysis complete:', result);
  };

  return (
    <ImageUpload
      onAnalysisComplete={handleResult}
      apiBaseUrl="http://127.0.0.1:8000"
    />
  );
}
```

### Using Axios (Alternative)

If you prefer Axios over fetch:

```tsx
import axios from 'axios';

const handleAnalyze = async (file: File) => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await axios.post(
      'http://127.0.0.1:8000/analyze/',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' }
      }
    );
    console.log('Result:', response.data);
  } catch (error) {
    console.error('Upload failed:', error.message);
  }
};
```

---

## Part 6: Troubleshooting

### Issue: CORS Error in Browser Console
**Error:** `Access to XMLHttpRequest at 'http://127.0.0.1:8000/analyze/' from origin 'http://localhost:3000' has been blocked by CORS policy`

**Fix:**
1. Ensure `django-cors-headers` is installed: `pip install django-cors-headers`
2. Verify `corsheaders` is in `INSTALLED_APPS`
3. Verify `CorsMiddleware` is in `MIDDLEWARE` (high in the list)
4. Verify `CORS_ALLOWED_ORIGINS` includes `http://localhost:3000`
5. Restart Django: `python manage.py runserver`

### Issue: Backend Not Responding
**Error:** `Failed to analyze image: ... Make sure the backend is running at http://127.0.0.1:8000`

**Fix:**
1. Check if Django is running: `python manage.py runserver`
2. Test the endpoint: `curl http://127.0.0.1:8000/ping/`
3. Verify the port (default: 8000)
4. Check firewall settings

### Issue: File Upload Returns 400 Error
**Error:** `{"error": "Invalid file type. Please upload an image..."}`

**Fix:**
1. Ensure your file has a valid extension: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, or `.dcm`
2. Check the file size (max 50MB in frontend validation)
3. Verify the field name is `file` (exact match)

### Issue: DICOM Files Not Previewing
**Expected Behavior:** DICOM files (.dcm) won't show a preview because browsers can't natively display them. This is normal. The file will still upload successfully.

---

## Part 7: Future Enhancements

### 1. **Add Authentication**
```tsx
// Add JWT token to requests
const response = await fetch(`${apiBaseUrl}/analyze/`, {
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData,
});
```

### 2. **Display Heatmap**
Once your ML model returns heatmap data:
```tsx
interface AnalysisResult {
  result: 'clean' | 'tumor';
  confidence: number;
  heatmap_url?: string; // Add this
}

// In component:
{result.heatmap_url && (
  <img src={result.heatmap_url} alt="Analysis heatmap" />
)}
```

### 3. **Store Analysis History in Database**
Add a backend endpoint to save analysis results:
```python
# backend/api/models.py
class Analysis(models.Model):
  user = models.ForeignKey(User, on_delete=models.CASCADE)
  file = models.FileField(upload_to='analyses/')
  result = models.CharField(max_length=20)
  confidence = models.FloatField()
  created_at = models.DateTimeField(auto_now_add=True)
```

### 4. **Bulk Upload**
Modify `ImageUpload` to accept multiple files.

### 5. **Progress Bar for Large Files**
```tsx
const handleAnalyze = async () => {
  const xhr = new XMLHttpRequest();
  xhr.upload.addEventListener('progress', (e) => {
    const percentComplete = (e.loaded / e.total) * 100;
    setUploadProgress(percentComplete);
  });
  // ... rest of upload logic
};
```

### 6. **Add Export Results**
Export analysis history as CSV or PDF.

---

## Part 8: Production Deployment

### Backend (Django)
1. Set `DEBUG = False` in `settings.py`
2. Update `ALLOWED_HOSTS` with your domain
3. Update `CORS_ALLOWED_ORIGINS` with production domain
4. Use a production database (PostgreSQL recommended)
5. Use Gunicorn or similar WSGI server
6. Add environment variables for secrets

### Frontend (Next.js)
1. Build: `npm run build`
2. Start: `npm run start`
3. Deploy to Vercel, Netlify, or your own server
4. Update `apiBaseUrl` to production backend URL
5. Add environment variable: `NEXT_PUBLIC_API_URL`

**Example `.env.local`:**
```
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

**Update component:**
```tsx
const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
```

---

## Summary

✅ **Backend:** Django REST API with CORS configured  
✅ **Frontend:** Next.js with reusable `ImageUpload` component  
✅ **Integration:** Fetch-based POST requests with error handling  
✅ **Styling:** Tailwind CSS for clean, responsive UI  
✅ **Testing:** Ready for integration with real ML model  

You're ready to start analyzing medical images! 🚀
