import os

from inference.predict import run_inference
from training.generate_synthetic_data import generate_benchmark_dataset


def test_full_pipeline_end_to_end(tmp_path):
    dataset_dir = os.path.join(tmp_path, "data")
    generate_benchmark_dataset(base_dir=dataset_dir)

    sample_img = os.path.join(dataset_dir, "sample", "sample_sentinel1.tif")
    out_dir = os.path.join(tmp_path, "outputs")

    spill_event = run_inference(
        input_image_path=sample_img,
        threshold=0.4,
        config_path="config.yaml",
        output_dir=out_dir
    )

    assert spill_event.event_id.startswith("SPILL_EVENT_")
    assert os.path.exists(os.path.join(out_dir, "probability_map.png"))
    assert os.path.exists(os.path.join(out_dir, "segmentation_mask.png"))
    assert os.path.exists(os.path.join(out_dir, "spill_overlay.png"))
    assert os.path.exists(os.path.join(out_dir, "spill_event.json"))
    assert os.path.exists(os.path.join(out_dir, "spill.geojson"))
