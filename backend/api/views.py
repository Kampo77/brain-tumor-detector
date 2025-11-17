from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser


class PingView(APIView):
    """
    Health check endpoint.
    GET /ping/ → returns {"message": "API is working"}
    """
    def get(self, request):
        return Response(
            {"message": "API is working"},
            status=status.HTTP_200_OK
        )


class AnalyzeView(APIView):
    """
    Image analysis endpoint.
    POST /analyze/ accepts an uploaded image file and returns a placeholder prediction.
    
    Request: multipart/form-data with "file" field
    Response: {"result": "clean" | "tumor", "confidence": 0.0-1.0}
    """
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        # Check if file was uploaded
        if 'file' not in request.FILES:
            return Response(
                {"error": "No file provided. Please upload an image with field name 'file'."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        uploaded_file = request.FILES['file']
        
        # Validate file is an image (basic check)
        if not uploaded_file.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.dcm')):
            return Response(
                {"error": "Invalid file type. Please upload an image (jpg, png, gif, bmp) or DICOM file (dcm)."},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # TODO: Here we will later call the real ML model from model/ folder
        # For now, return a placeholder prediction
        placeholder_result = {
            "result": "clean",
            "confidence": 0.99,
            "message": "Placeholder prediction. Real ML model will be integrated later."
        }
        
        return Response(placeholder_result, status=status.HTTP_200_OK)
