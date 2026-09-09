import os

import numpy as np
import torch

from app.preprocessing.eo_preprocessor import EOPreprocessor


def test_eo_preprocessor_synthetic_image(tmp_path):
    # Create temporary RGB PNG image
    from PIL import Image
    img_data = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
    img_path = os.path.join(tmp_path, "test_eo.png")
    Image.fromarray(img_data).save(img_path)

    preprocessor = EOPreprocessor(target_size=(128, 128), normalize=True)
    img_norm, tensor, meta = preprocessor.preprocess_eo(img_path)

    assert isinstance(img_norm, np.ndarray)
    assert img_norm.min() >= 0.0 and img_norm.max() <= 1.0
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 128, 128)
    assert meta["file_path"] == img_path
