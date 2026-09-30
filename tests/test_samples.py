import os
import pytest
from PIL import Image
from sample_generator import generate_preset_images

def test_preset_image_generation(tmp_path):
    output_dir = str(tmp_path / "presets")
    image_paths = generate_preset_images(output_dir)
    
    assert len(image_paths) == 4
    for case_id, path in image_paths.items():
        assert os.path.exists(path)
        with Image.open(path) as img:
            assert img.size == (640, 640)
            assert img.format in ["JPEG", "PNG"]
