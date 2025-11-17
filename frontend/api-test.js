/**
 * API Test Utility
 * Use this to manually test the backend endpoints from the browser console
 * Copy and paste functions into your browser DevTools Console
 */

// Configuration
const API_BASE_URL = 'http://127.0.0.1:8000';

/**
 * Test the /ping/ endpoint
 */
async function testPing() {
  console.log('🔍 Testing /ping/ endpoint...');
  try {
    const response = await fetch(`${API_BASE_URL}/ping/`);
    const data = await response.json();
    console.log('✅ Ping successful:', data);
    return data;
  } catch (error) {
    console.error('❌ Ping failed:', error.message);
  }
}

/**
 * Test the /analyze/ endpoint with a file
 * @param {File} file - The file to upload
 */
async function testAnalyze(file) {
  console.log('🔍 Testing /analyze/ endpoint...');
  console.log('📄 File:', file.name, `(${(file.size / 1024).toFixed(2)}KB)`);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/analyze/`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      console.log('✅ Analysis successful:', data);
      return data;
    } else {
      console.error('❌ Analysis failed:', data);
      return null;
    }
  } catch (error) {
    console.error('❌ Request failed:', error.message);
  }
}

/**
 * Helper to upload file from URL (for testing)
 * @param {string} url - URL to fetch file from
 */
async function testAnalyzeFromUrl(url) {
  try {
    const response = await fetch(url);
    const blob = await response.blob();
    const file = new File([blob], 'test-image', { type: blob.type });
    return testAnalyze(file);
  } catch (error) {
    console.error('❌ Failed to fetch file from URL:', error.message);
  }
}

/**
 * Interactive file upload test
 * This creates an input element and lets you select a file
 */
function testAnalyzeInteractive() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*,.dcm';
  input.onchange = (e) => {
    const file = e.target.files[0];
    if (file) {
      testAnalyze(file);
    }
  };
  input.click();
}

/**
 * Check CORS headers
 */
async function testCors() {
  console.log('🔍 Testing CORS configuration...');
  try {
    const response = await fetch(`${API_BASE_URL}/ping/`, {
      method: 'OPTIONS',
    });
    console.log('Response headers:');
    response.headers.forEach((value, key) => {
      if (key.toLowerCase().includes('cors') || key.toLowerCase().includes('access')) {
        console.log(`  ${key}: ${value}`);
      }
    });
  } catch (error) {
    console.error('❌ CORS test failed:', error.message);
  }
}

/**
 * Run all tests
 */
async function runAllTests() {
  console.log('🚀 Running all API tests...\n');
  
  console.log('=== Test 1: Ping ===');
  await testPing();
  
  console.log('\n=== Test 2: CORS ===');
  await testCors();
  
  console.log('\n=== Test 3: Interactive Analysis ===');
  console.log('A file dialog will open. Select an image to test the /analyze/ endpoint.');
  testAnalyzeInteractive();
}

// Export for use in tests
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    testPing,
    testAnalyze,
    testAnalyzeFromUrl,
    testAnalyzeInteractive,
    testCors,
    runAllTests,
  };
}

console.log(`
📡 API Testing Utilities Loaded

Available functions:
  • testPing()                    - Test /ping/ endpoint
  • testAnalyze(file)             - Test /analyze/ with a file
  • testAnalyzeFromUrl(url)       - Test /analyze/ with URL
  • testAnalyzeInteractive()      - Interactive file upload test
  • testCors()                    - Check CORS headers
  • runAllTests()                 - Run all tests at once

API Base URL: ${API_BASE_URL}

Examples:
  testPing()
  testAnalyzeInteractive()
  runAllTests()
`);
