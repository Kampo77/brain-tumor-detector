# Code Walkthrough - ImageUpload Component

This document provides a line-by-line walkthrough of the `ImageUpload.tsx` component to help you understand how it works and how to modify it for your needs.

---

## Component Overview

The `ImageUpload` component handles the complete image upload and analysis workflow:
1. File selection (click or drag-drop)
2. Validation
3. Preview
4. Upload to backend
5. Result display
6. History management

---

## Part 1: Imports & TypeScript Interfaces

```typescript
'use client';

import { useState, useRef } from 'react';
```

- `'use client'` - Marks this as a Client Component (Next.js 13+ App Router)
- `useState` - React hook for state management
- `useRef` - React hook for direct DOM access (file input)

### Type Definitions

```typescript
interface AnalysisResult {
  result: 'clean' | 'tumor' | 'no_tumor';
  confidence: number;
  message?: string;
}
```

**Explanation:**
- `result` - Can only be one of three strings (tuple type)
- `confidence` - Number between 0 and 1 (representing 0-100%)
- `message` - Optional field (?) for backend messages
- This matches the Django API response

```typescript
interface ImageUploadProps {
  onAnalysisComplete?: (result: AnalysisResult) => void;
  apiBaseUrl?: string;
}
```

**Explanation:**
- `onAnalysisComplete` - Callback function (optional ?) that parent component can use
- `apiBaseUrl` - URL of backend (default: http://127.0.0.1:8000)
- Both are optional, providing sensible defaults

---

## Part 2: Component Function & State

```typescript
export default function ImageUpload({
  onAnalysisComplete,
  apiBaseUrl = 'http://127.0.0.1:8000',
}: ImageUploadProps) {
```

**Explanation:**
- Destructures props with default value for `apiBaseUrl`
- If parent doesn't pass `apiBaseUrl`, defaults to localhost:8000

### State Variables

```typescript
const [file, setFile] = useState<File | null>(null);
```
- Holds the selected File object or null

```typescript
const [preview, setPreview] = useState<string | null>(null);
```
- Holds the image preview as a data URL string (base64)
- Only for images, not DICOM files

```typescript
const [loading, setLoading] = useState(false);
```
- Controls loading spinner visibility during upload

```typescript
const [result, setResult] = useState<AnalysisResult | null>(null);
```
- Stores the response from backend API

```typescript
const [error, setError] = useState<string | null>(null);
```
- Stores any error messages to display to user

```typescript
const fileInputRef = useRef<HTMLInputElement>(null);
```
- Reference to hidden file input element
- Allows us to trigger it from the click handler

### Constants

```typescript
const VALID_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'dcm'];
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50 MB
```

- Hardcoded validation rules
- Change these to add/remove file types or adjust size limit

---

## Part 3: File Validation

```typescript
const validateFile = (file: File): string | null => {
```

Returns error message string if invalid, null if valid.

### Size Check
```typescript
if (file.size > MAX_FILE_SIZE) {
  return `File size must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB...`;
}
```

- `file.size` - Browser File API property (in bytes)
- Converts to MB for display
- Returns early if file too large

### Extension Check
```typescript
const fileName = file.name.toLowerCase();
const hasValidExtension = VALID_EXTENSIONS.some((ext) =>
  fileName.endsWith(`.${ext}`)
);

if (!hasValidExtension) {
  return `Invalid file type...`;
}
```

- Converts filename to lowercase (case-insensitive check)
- `Array.some()` returns true if ANY extension matches
- Example: "image.JPG" → "image.jpg" → matches "jpg"

### Return
```typescript
return null;
```
- Returns null if validation passes (no error)

---

## Part 4: File Selection Handler

```typescript
const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
```

Called when user clicks to select a file.

```typescript
const selectedFile = e.target.files?.[0];
if (!selectedFile) return;
```

- `e.target.files` - Array-like of selected files
- `?.[0]` - Optional chaining: safely get first file or undefined
- `if (!selectedFile) return` - Early exit if no file selected

```typescript
const validationError = validateFile(selectedFile);
if (validationError) {
  setError(validationError);
  setFile(null);
  setPreview(null);
  return;
}
```

- Validate file
- If error, display it and reset file state
- Clear preview
- Exit early

```typescript
setError(null);
setFile(selectedFile);
setResult(null);
```

- Clear previous errors
- Store new file
- Clear previous result

### Create Preview

```typescript
if (!selectedFile.name.toLowerCase().endsWith('.dcm')) {
  const reader = new FileReader();
  reader.onload = (e) => {
    setPreview(e.target?.result as string);
  };
  reader.readAsDataURL(selectedFile);
} else {
  setPreview(null);
}
```

**Explanation:**
- Skip preview for DICOM files (browsers can't display them)
- `FileReader` - Browser API to read file contents
- `readAsDataURL()` - Converts file to base64 string
- `onload` callback - Called when reading complete
- `e.target?.result` - The base64 data URL
- Store as preview for `<img>` tag

---

## Part 5: Drag & Drop Handlers

```typescript
const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
  e.preventDefault();
  e.stopPropagation();
};
```

**Why prevent default?**
- By default, browsers would open the file (navigate to it)
- `preventDefault()` stops this behavior
- Allows us to handle the drop ourselves

```typescript
const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
  e.preventDefault();
  e.stopPropagation();

  const droppedFile = e.dataTransfer.files?.[0];
```

- `e.dataTransfer.files` - Files from drag operation
- Rest is identical to file selection handler

---

## Part 6: Upload Handler (Core Logic)

```typescript
const handleAnalyze = async () => {
```

Called when user clicks "Analyze Image" button.

### Pre-flight Check
```typescript
if (!file) {
  setError('Please select a file first');
  return;
}
```

- Safeguard: verify file selected before sending
- Should be impossible due to button disabled state, but good practice

### Prepare Request
```typescript
setLoading(true);
setError(null);
```

- Show spinner
- Clear previous errors

### Create FormData
```typescript
const formData = new FormData();
formData.append('file', file);
```

**Why FormData?**
- Proper way to send files in multipart/form-data
- Django expects field named 'file'

### Send Request

```typescript
const response = await fetch(`${apiBaseUrl}/analyze/`, {
  method: 'POST',
  body: formData,
});
```

**Key points:**
- `async/await` - Modern promise syntax
- `fetch()` - Browser's HTTP client
- `method: 'POST'` - HTTP method
- Don't set `Content-Type` header - browser handles it!
- `body: formData` - The file data

### Check Response

```typescript
if (!response.ok) {
  const errorData = await response.json().catch(() => ({}));
  throw new Error(
    errorData.error ||
      `Server error: ${response.status} ${response.statusText}`
  );
}
```

**Explanation:**
- `!response.ok` - Checks if status >= 400
- Try to parse error JSON from backend
- `.catch(() => ({}))` - If JSON parsing fails, use empty object
- Throw custom error with backend message or generic message

### Parse Success Response

```typescript
const data: AnalysisResult = await response.json();
setResult(data);
onAnalysisComplete?.(data);
```

- Parse response JSON
- Store in state
- Call parent callback with `?.` (optional chaining - only call if provided)

### Error Handling

```typescript
} catch (err) {
  const errorMessage =
    err instanceof Error ? err.message : 'Unknown error occurred';
  setError(
    `Failed to analyze image: ${errorMessage}. Make sure the backend is running at ${apiBaseUrl}`
  );
}
```

- Catch network errors, JSON parse errors, thrown errors
- Check if error is Error instance (has `.message`)
- Display helpful error with backend URL

### Cleanup

```typescript
} finally {
  setLoading(false);
}
```

- Always called, whether success or error
- Hide loading spinner

---

## Part 7: Reset Handler

```typescript
const handleReset = () => {
  setFile(null);
  setPreview(null);
  setResult(null);
  setError(null);
  if (fileInputRef.current) {
    fileInputRef.current.value = '';
  }
};
```

- Clears all state
- Clears hidden file input (so same file can be re-selected)
- Called when user clicks "Clear" or "Analyze Another Image"

---

## Part 8: JSX Render

### Upload Area

```typescript
<div
  onDragOver={handleDragOver}
  onDragLeave={handleDragLeave}
  onDrop={handleDrop}
  className="border-2 border-dashed border-blue-300 rounded-lg p-8..."
  onClick={() => fileInputRef.current?.click()}
>
```

**Interactive regions:**
- `onDragOver` - User dragging file over area
- `onDrop` - User dropped file
- `onClick` - User clicked area

### Hidden File Input

```typescript
<input
  ref={fileInputRef}
  type="file"
  accept=".jpg,.jpeg,.png,.gif,.bmp,.dcm"
  onChange={handleFileChange}
  disabled={loading}
  className="hidden"
/>
```

- `ref={fileInputRef}` - Link to state variable
- `accept` - Browser file picker filter (user can still override)
- `onChange` - Triggered when user selects file
- `disabled={loading}` - Prevent multiple uploads
- `className="hidden"` - Tailwind CSS to hide input

### File Info Display

```typescript
{file && (
  <div className="mt-6 p-4 bg-gray-100 rounded-lg">
    <p className="text-sm font-medium text-gray-900">Selected file:</p>
    <p className="text-sm text-gray-600 mt-1">{file.name}</p>
    <p className="text-xs text-gray-500 mt-1">
      {(file.size / 1024).toFixed(2)} KB
    </p>
  </div>
)}
```

- `{file && ...}` - Only render if file selected
- `file.name` - Filename
- `file.size` - Size in bytes, convert to KB

### Image Preview

```typescript
{preview && (
  <div className="mt-6">
    <p className="text-sm font-medium text-gray-900 mb-3">Preview:</p>
    <img
      src={preview}
      alt="Selected image preview"
      className="w-full max-h-96 object-contain rounded-lg border border-gray-300"
    />
  </div>
)}
```

- `{preview && ...}` - Only render if preview generated
- `src={preview}` - Use base64 data URL from state
- Tailwind classes for sizing and styling

### Error Display

```typescript
{error && (
  <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
    <p className="text-sm font-medium text-red-800">Error:</p>
    <p className="text-sm text-red-700 mt-1">{error}</p>
  </div>
)}
```

- Red background for visual warning
- Shows error message text

### Result Display

```typescript
{result && (
  <div className="mt-6 p-6 bg-green-50 border border-green-200 rounded-lg">
    <p className="text-lg font-semibold text-green-900 mb-3">
      Analysis Result
    </p>
    
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <span className="text-gray-700">Status:</span>
        <span
          className={`font-semibold px-3 py-1 rounded ${
            result.result === 'clean' || result.result === 'no_tumor'
              ? 'bg-green-200 text-green-900'
              : 'bg-red-200 text-red-900'
          }`}
        >
          {result.result.toUpperCase()}
        </span>
      </div>
```

**Badge coloring:**
```typescript
? 'bg-green-200 text-green-900'  // "clean" or "no_tumor"
: 'bg-red-200 text-red-900'      // "tumor"
```

### Confidence Display

```typescript
<div className="flex justify-between items-center">
  <span className="text-gray-700">Confidence:</span>
  <div className="flex items-center gap-3">
    <div className="w-32 bg-gray-200 rounded-full h-2">
      <div
        className={`h-2 rounded-full transition-all ${
          result.confidence > 0.8
            ? 'bg-green-500'
            : 'bg-yellow-500'
        }`}
        style={{ width: `${result.confidence * 100}%` }}
      />
    </div>
    <span className="font-semibold text-gray-900 min-w-fit">
      {(result.confidence * 100).toFixed(1)}%
    </span>
  </div>
</div>
```

**Key parts:**
- Background bar (gray) showing potential 100%
- Foreground bar showing actual confidence
- Color changes at 80% threshold (green/yellow)
- Width calculated: `confidence * 100` (0.95 → 95%)
- `.toFixed(1)` - Display as percentage with 1 decimal

### Action Buttons

```typescript
{!result ? (
  <>
    <button
      onClick={handleAnalyze}
      disabled={!file || loading}
      className="flex-1 px-6 py-3 bg-blue-600 text-white..."
    >
```

**Button states:**
- Enabled: Has file selected AND not loading
- Shows spinner when loading
- `flex-1` - Takes remaining space in flex container

---

## Customization Guide

### Change File Types

```typescript
// Line: const VALID_EXTENSIONS = ...
const VALID_EXTENSIONS = ['jpg', 'jpeg', 'png']; // Remove bmp, dcm, gif

// Line: accept=".jpg,.jpeg,..."
accept=".jpg,.jpeg,.png"  // Update input element
```

### Change Max File Size

```typescript
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB instead of 50 MB
```

### Change API Endpoint

```typescript
// Default (in parent component):
<ImageUpload apiBaseUrl="http://192.168.1.100:8000" />

// Or in production:
<ImageUpload apiBaseUrl="https://api.yourdomain.com" />
```

### Add Custom Error Messages

```typescript
if (!response.ok) {
  let userMessage = 'Unknown error';
  
  if (response.status === 413) {
    userMessage = 'File too large. Max 50MB.';
  } else if (response.status === 415) {
    userMessage = 'Unsupported file type.';
  }
  
  throw new Error(userMessage);
}
```

### Add Loading Progress

```typescript
// For percentage-based upload:
const xhr = new XMLHttpRequest();
xhr.upload.addEventListener('progress', (e) => {
  if (e.lengthComputable) {
    const percentComplete = (e.loaded / e.total) * 100;
    setUploadProgress(percentComplete);
  }
});

// Note: fetch API doesn't support progress natively
// Use Axios or XMLHttpRequest for this
```

### Add Request Timeout

```typescript
const controller = new AbortController();
const timeout = setTimeout(() => controller.abort(), 30000); // 30 sec

const response = await fetch(`${apiBaseUrl}/analyze/`, {
  signal: controller.signal,
  // ...
});

clearTimeout(timeout);
```

---

## Common Issues & Solutions

### Issue: Upload works but button stays disabled
**Cause:** `result` state not being set properly
**Fix:** Check backend is returning valid JSON with `result` and `confidence` fields

### Issue: Preview not showing
**Cause:** File is DICOM or browser can't display format
**Fix:** This is expected for DICOM. For images, check browser console for errors.

### Issue: Component keeps showing "Analyzing..."
**Cause:** Error in response handling, `loading` not being set to false
**Fix:** Check browser console for errors, verify backend response format

### Issue: File input won't trigger on second upload
**Cause:** File input value wasn't cleared
**Fix:** Ensure `handleReset()` is called, which clears the input

---

## TypeScript Tips

### Narrowing Error Types
```typescript
if (err instanceof Error) {
  console.log(err.message);  // Error has .message
} else {
  console.log(String(err));  // Unknown type
}
```

### Optional Chaining
```typescript
onAnalysisComplete?.(data);     // Only call if defined
file?.name;                      // Only access if file exists
```

### Type Assertions
```typescript
const preview = e.target?.result as string;  // We know it's a string
```

---

## Performance Considerations

1. **File Preview:** Only read first MB of file if it's large
2. **Request Timeout:** Add timeout for slow connections
3. **Debounce:** If handling drag events multiple times
4. **Memoization:** Wrap component with `memo()` if used multiple times

---

## Next Steps

- Read `INTEGRATION_GUIDE.md` for full integration details
- Check `app/page.tsx` for how to use this component
- Modify `api/views.py` in Django to integrate real ML model
- Add authentication when needed
- Add more validation/error handling as requirements change

Happy coding! 🚀
