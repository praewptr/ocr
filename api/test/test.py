import pytest
import sys
import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from api.main import app

client = TestClient(app)


def get_test_image():
    """Get the real test image (DRAFT image) if it exists"""
    current_dir = Path(__file__).parent
    search_paths = [
        current_dir,
        current_dir.parent,
        current_dir.parent.parent,
    ]

    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    for search_path in search_paths:
        if search_path.exists():
            for ext in image_extensions:
                image_files = list(search_path.glob(f"*{ext}"))
                if image_files:
                    return image_files[0]
    
    return None

def test_api_health():
    response = client.post("/ocr", files={"file": ("test.txt", b"fake", "text/plain")})
    assert response.status_code == 400


def test_ocr_valid_file():
    img = Image.new('RGB', (200, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    test_image = img_byte_arr.getvalue()
    
    response = client.post(
        "/ocr",
        files={"file": ("test.jpg", test_image, "image/jpeg")}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_ocr_invalid_file():
    response = client.post(
        "/ocr",
        files={"file": ("test.txt", b"not an image", "text/plain")}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "Invalid file type" in data["detail"]



def test_ocr_performance():
    import time
    
    real_image_path = get_test_image()
    
    if real_image_path:
        with open(real_image_path, "rb") as f:
            image_data = f.read()
        
        start_time = time.time()
        response = client.post(
            "/ocr",
            files={"file": (real_image_path.name, image_data, "image/jpeg")}
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        file_size = len(image_data)
        
        assert response.status_code == 200
        assert response_time < 15.0 
       

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])