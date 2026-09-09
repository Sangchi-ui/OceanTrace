# PROJECT TECHNICAL DOCUMENTATION

## 1. Project Overview

- **Project Name:** OceanTrace - Agent 1 SAR Oil Spill Detection
- **Project Purpose:** To automatically detect, segment, and characterize marine oil spills using remote sensing imagery.
- **Problem being solved:** Marine oil spills cause massive ecological damage. Rapid and automated identification of slicks from satellite imagery enables faster response and mitigation.
- **Why the problem is important:** Manual analysis of satellite imagery is slow and subjective. Automating detection scales environmental monitoring globally.
- **Proposed Solution:** A deep learning pipeline built on a U-Net architecture that ingests Earth Observation (EO) optical satellite imagery, predicts pixel-level oil spill probabilities, applies morphological filtering, extracts geographic polygon data, validates the detections against physical constraints, and outputs a highly structured data contract.
- **What the system ultimately produces:** A `SpillEvent` JSON payload, a GeoJSON vector file containing precise slick boundaries, and comparative visualization overlays.
- **High-level workflow:** Image Input → Preprocessing (RGB Resizing & Normalization) → U-Net Inference → Post-processing (Thresholding, Morphology, Connected Components) → Geospatial Extraction (Polygons, Area) → Quality Control Validation → API/JSON Output.
- **Major components:** EO Preprocessor, PyTorch U-Net Model, Mask Processor, Geospatial Processor, Spill Validator, FastAPI Web Service.

---

## 2. Complete Technology Stack

| Category | Technology | Version | Purpose | Where/How Used |
| :--- | :--- | :--- | :--- | :--- |
| **Language** | Python | 3.10+ | Core programming language | Entire backend and ML logic |
| **ML Framework** | PyTorch (`torch`) | >=2.0.0 | Deep Learning framework | Model definition, training, inference |
| **Computer Vision** | OpenCV (`opencv-python-headless`) | >=4.7.0 | Image manipulation & morphology | Preprocessing / dataset generation |
| **Computer Vision** | scikit-image | >=0.20.0 | Advanced morphology & connected components | Post-processing (`mask_processor.py`) |
| **Data Processing** | NumPy | >=1.24.0 | N-dimensional array manipulation | Tensor manipulation, image handling |
| **Data Processing** | SciPy | >=1.10.0 | Scientific computing | Backend operations for scikit-image |
| **Data Processing** | Pillow (`pillow`) | >=9.5.0 | Image I/O | Reading/writing images and masks |
| **Geospatial** | Rasterio | >=1.3.0 | Raster georeferencing & vectorization | Extracting shapes and affine transforms |
| **Geospatial** | Shapely | >=2.0.0 | Planar geometry manipulation | Creating Polygons and Centroids |
| **Geospatial** | GeoPandas | >=0.13.0 | Geospatial data manipulation | Handling coordinate reference systems |
| **Backend/API** | FastAPI | >=0.95.0 | High-performance async web framework | Serving the model inference endpoints |
| **Backend/API** | Uvicorn | >=0.22.0 | ASGI web server | Running the FastAPI application |
| **Backend/API** | Pydantic | >=2.0.0 | Data validation & serialization | Defining the `SpillEvent` output schema |
| **Testing** | PyTest | >=7.3.0 | Unit testing framework | Executing automated end-to-end tests |
| **Hardware** | CUDA / GPU | N/A | Hardware acceleration | GPU training and fast inference via PyTorch |

---

## 3. Complete Library & Dependency Inventory

- **torch & torchvision:** Core deep learning engine. Used during training, validation, and inference to build the U-Net, load state dictionaries, and execute tensor operations. Required for model execution.
- **numpy:** Used throughout preprocessing, inference, postprocessing, and evaluation to handle image arrays before converting to/from PyTorch tensors.
- **scikit-image (`skimage`):** Used strictly during post-processing. Required for `opening`, `closing`, `disk` structuring elements, and `label`/`regionprops` to extract connected oil spill regions from the raw probability map.
- **rasterio:** Used during geospatial processing and dataset generation to read GeoTIFFs, extract spatial transforms (Affine), and convert raster masks into vector boundaries.
- **shapely:** Used during geospatial processing to construct `Polygon` and `MultiPolygon` objects, calculate geometric centroids, and project bounding boxes.
- **pyproj:** Used to project coordinates from standard geographic CRS (e.g., EPSG:4326) into equal-area projections (like ESRI:54009) to accurately compute the surface area in square kilometers.
- **fastapi & uvicorn:** Used for deployment. Forms the backend API allowing users to POST images to `/api/v1/detect` and receive JSON responses.
- **pydantic:** Used throughout the pipeline output stage to strictly enforce data types and nested schemas for the `SpillEvent` data contract.
- **pyyaml:** Used to parse `config.yaml` during training and inference initialization.
- **cv2 (OpenCV):** Used primarily in dataset preparation (e.g., `prepare_kaggle_dataset.py`) for Gaussian blurring and Otsu thresholding to generate pseudo-ground-truth masks.

---

## 4. AI/ML Models Used

### UNet
- **Exact model name:** UNet
- **Model type:** Fully Convolutional Network for Semantic Segmentation
- **Architecture:** Standard U-Net (Encoder-Bottleneck-Decoder with Skip Connections)
- **Framework:** PyTorch (`torch.nn.Module`)
- **Pretrained or trained from scratch:** Trained from scratch (checkpoints saved to `models/unet_oil_spill.pth`)
- **Number of parameters:** Varies based on base feature count, but utilizes an expansive [64, 128, 256, 512] feature hierarchy.
- **Input requirements:** Tensor of shape `[Batch, 3, Height, Width]` containing normalized RGB data.
- **Output:** Tensor of shape `[Batch, 1, Height, Width]` containing raw logits (which are passed through a Sigmoid activation to produce probabilities).
- **Main purpose:** To predict the per-pixel probability that a specific pixel contains an oil slick.
- **Where it appears:** It sits exactly in the middle of the pipeline (Inference Phase), directly consuming the preprocessed tensor and yielding a probability map for the post-processor.

---

## 5. Model Architecture — Deep Technical Explanation

The custom PyTorch `UNet` (`app/models/unet.py`) consists of the following components:

- **DoubleConv Block:** A reusable sequential block consisting of: `Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU -> Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU`.
- **Input Layer:** Accepts `[B, 3, 256, 256]` tensors.
- **Encoder (Down-sampling path):**
  - Iterates through the feature list: 64, 128, 256, 512.
  - At each step, applies a `DoubleConv` block, saves the output tensor to a `skip_connections` list, and then applies a `MaxPool2d(kernel_size=2, stride=2)` to halve the spatial dimensions.
- **Bottleneck:**
  - A single `DoubleConv` block expanding features from 512 to 1024.
- **Decoder (Up-sampling path):**
  - Uses `ConvTranspose2d(kernel_size=2, stride=2)` to double the spatial dimensions.
  - Retrieves the corresponding tensor from the reversed `skip_connections` list.
  - Applies Bilinear Interpolation (`F.interpolate`) if spatial dimensions mismatch due to odd input sizes.
  - Concatenates the up-sampled tensor with the skip connection tensor (`torch.cat(dim=1)`).
  - Applies a `DoubleConv` block to halve the feature depth back down the hierarchy.
- **Classification Head (Output Layer):**
  - A final `Conv2d(kernel_size=1)` mapping the 64 features to `num_classes` (1).
- **Inference Wrapper (`predict` method):**
  - Wraps the forward pass in `torch.no_grad()` and applies `torch.sigmoid()` to bound the logits between `[0.0, 1.0]`.

---

## 6. Input Data

- **Input type:** Satellite Raster Imagery.
- **File format:** `.tif`, `.tiff`, `.png`, `.jpg`.
- **Resolution/shape:** Dynamically resized to a standard target size (defined in `config.yaml` as `256 x 256`).
- **Data dimensions:** `[Height, Width, 3]`.
- **Number of classes:** 1 (Binary Segmentation: Oil vs. Ocean/Background).
- **Input tensor datatype:** `torch.float32`.
- **Normalization range:** Linearly scaled from `[0, 255]` to `[0.0, 1.0]`.

---

## 7. INPUT CHANNELS — VERY IMPORTANT

**How many input channels does the model use?**
- **3 channels** (Multi-spectral Earth Observation optical data).

**What does each channel represent?**
- Channel 1 → **Red** Optical Band
- Channel 2 → **Green** Optical Band
- Channel 3 → **Blue** Optical Band
*(Inferred from the implementation of `EOPreprocessor` which explicitly calls `Image.open().convert("RGB")`)*

**WHY are these channels used?**
The current iteration of the project has been updated to process multi-spectral optical Earth Observation (EO) data (like the MADOS dataset), deviating from the original single-channel SAR setup. The 3 channels (RGB) capture visible spectrum reflectance. Oil slicks on the ocean surface reflect light differently than surrounding water due to changes in surface tension and refractive index, creating visual contrast that the convolutional filters learn to detect across the three color dimensions.

---

## 8. Dataset

- **Dataset names:** MADOS (Optical EO dataset) and synthetic benchmark datasets. There are also scripts present for a CSIRO Kaggle dataset.
- **Dataset size:** Inferred from system logs to be at least 1,433 training samples and 642 validation samples (MADOS dataset).
- **Number of classes:** 2 (Background vs. Slick). Formulated as a single-class binary mask (1 = slick, 0 = background).
- **Data format:** Images stored as TIFF/PNG in `data/raw/train/images/` and binary masks stored as PNGs in `data/raw/train/masks/`.
- **Annotation method:** Ground truth masks are standard 2D binary grids where `255` or `1` denotes the slick.
- **Synthetic Data:** The project contains `generate_synthetic_data.py`, which algorithmically generates simulated slicks using random ellipses, wind-drift lines, speckle noise (Gamma distribution), and bounding boxes.

---

## 9. Complete Data Pipeline

`Raw Image (.tif)`
↓
`Data Loading`: Pillow (`Image.open`) loads the raster from disk.
↓
`Channel Conversion`: Pillow converts the image to strictly 3-channel RGB (`convert("RGB")`).
↓
`Resizing`: Pillow applies Bilinear interpolation to scale the image to 256x256.
↓
`Normalization`: NumPy divides the pixel values by 255.0 to scale them to `[0, 1]`.
↓
`Tensor Conversion`: PyTorch permutes the dimensions from HWC to CHW and adds a batch dimension (`[1, 3, 256, 256]`).
↓
`Model Inference`: The U-Net performs a forward pass, outputting logits.
↓
`Activation`: Sigmoid scales logits to probability scores.
↓
`Post-processing`: Morphological cleaning and connected component labeling to extract distinct regions.
↓
`Geospatial Extraction`: Rasterio and Shapely map pixel regions to real-world latitude/longitude and compute physical area.
↓
`Validation`: The `SpillValidator` checks the area and confidence against configured thresholds.
↓
`Final Output`: A serialized `SpillEvent` JSON response is returned to the user via FastAPI.

---

## 10. Preprocessing Pipeline

Defined entirely in `app/preprocessing/eo_preprocessor.py`.

1. **Reading & Conversion**: Safely reads the file path and uses `convert("RGB")` to enforce 3-channel data regardless of source bit-depth.
2. **Resizing**: Down-samples or up-samples to `(256, 256)` using `Image.BILINEAR`.
3. **Normalization**: `arr_resized / 255.0` ensures the data fits nicely into the active domain of the neural network's initial convolutional layers, speeding up gradient convergence.
4. **Dimension Permutation**: PyTorch requires Channel-First data. The array is shifted from `(H, W, C)` to `(C, H, W)`.

---

## 11. Training Pipeline

Defined in `training/train.py` and `training/dataset.py`.

- **Data loader:** PyTorch `DataLoader` with dynamic batching (batch_size=16) and shuffling enabled.
- **Epochs:** 10 (defined in `config.yaml`).
- **Optimizer:** Adam.
- **Learning rate:** 0.001.
- **Weight decay:** 0.0001 (L2 regularization to prevent overfitting).
- **Loss function:** Combined Binary Cross Entropy with Logits + Dice Loss.
- **Validation:** Validation occurs at the end of every epoch without gradient tracking (`torch.no_grad()`).
- **Checkpointing:** The model saves a checkpoint (`best_model_path`) exclusively if the current epoch's validation loss is lower than the previous best (Best-Model saving strategy).
- **GPU Usage:** Dynamically detects CUDA (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`) and mounts tensors to the GPU.
- **Seed:** Manual seed fixed at `42` for reproducibility.

---

## 12. Loss Function

- **Name:** `CombinedBCEDiceLoss` (Located in `training/losses.py`).
- **Formula:** `Loss = (0.5 * BCEWithLogitsLoss) + (0.5 * DiceLoss)`
- **What it measures:** 
  - *BCE* measures pixel-wise classification accuracy.
  - *Dice* measures the intersection over union (spatial overlap) between the prediction and ground truth.
- **Why it is suitable:** Oil slicks are highly imbalanced (most of the ocean is empty). BCE alone can get stuck predicting all zeros. Dice loss forces the network to care about the structural shape and overlap of the small positive class. Combining them yields stable gradients and sharp spatial boundaries.

---

## 13. Optimizer

- **Optimizer name:** Adam (`torch.optim.Adam`).
- **Learning rate:** 0.001.
- **Weight decay:** 1e-4.
- **Why it is used:** Adam calculates adaptive learning rates for individual parameters using first and second moments of gradients. It is the industry standard for U-Net image segmentation tasks as it converges rapidly and robustly on noisy image data.

---

## 14. Evaluation

Metrics computed during standalone evaluation (inferred from `evaluate.py` references):
- **IoU (Intersection over Union):** Primary spatial accuracy metric.
- **Dice Coefficient:** Used both as a loss and a metric to measure overlap.
- **Precision:** How many predicted slicks are actually slicks (reduces false positives / look-alikes).
- **Recall:** How many true slicks were successfully found.
- **F1-score:** Harmonic mean of precision and recall.

---

## 15. Inference Pipeline

(Defined in `inference/predict.py`)

"User Input Image"
→ "Config Loader" (Reads `config.yaml` to set thresholds)
→ "Model Loading" (Loads U-Net architecture and `.pth` weights to GPU)
→ "EOPreprocessor" (Returns `[1, 3, 256, 256]` tensor)
→ "Model Inference" (`model.predict()` yields probability map)
→ "Original Scaling" (Probability map resized back to original raster dimensions)
→ "MaskProcessor" (Thresholds to 1/0, applies Morphological Open/Close, extracts connected components)
→ "GeospatialProcessor" (Calculates Lat/Lon, Polygons, Area in $km^2$)
→ "SpillValidator" (QC checks for fragmentation, size limits, and confidence)
→ "Event Constructor" (Pydantic models instantiated to build the JSON tree)
→ "Output Generation" (Saves `spill_event.json`, `spill.geojson`, and visualization PNGs)

---

## 16. Output

The model outputs highly structured data contracts built via Pydantic (`app/schemas/spill_event.py`):
- **Prediction format:** JSON and GeoJSON.
- **Classes:** 1 (Slick).
- **Confidence scores:** Mean probability score aggregated across the region.
- **Physical Extent:** Bounding box (`[min_x, min_y, max_x, max_y]`), Centroid Lat/Lon, Area in $km^2$, Perimeter in km.
- **Validation Status:** `VALID`, `WARNING`, or `INVALID`.
- **Visuals:** Probability heatmap, binary mask, and comparative overlay plot.

---

## 17. Project Architecture

```
User / Client
      │ (HTTP POST /api/v1/detect)
      ▼
   FastAPI
      │
      ▼
EO Preprocessor ───► Resized RGB Tensor
                         │
                         ▼
                     PyTorch U-Net ───► Probability Map [0, 1]
                                              │
                                              ▼
                                        Mask Processor ───► Binary Regions
                                              │
                                              ▼
                                    Geospatial Processor ───► Shapely Polygons / Area
                                              │
                                              ▼
                                       Spill Validator ───► QC Checks
                                              │
                                              ▼
                                     Pydantic Serializer
                                              │
                                              ▼
                                      SpillEvent JSON
```

---

## 18. File & Folder Structure

| File/Folder | Purpose | Important Components |
| :--- | :--- | :--- |
| `app/api/` | Web service endpoints | `main.py`, `routes.py` (FastAPI) |
| `app/models/` | Neural network definition | `unet.py` |
| `app/preprocessing/` | Data ingestion & formatting | `eo_preprocessor.py` |
| `app/postprocessing/` | Thresholding & cleaning | `mask_processor.py` |
| `app/geospatial/` | Projection & geometry math | `geometry.py` |
| `app/schemas/` | Output data structures | `spill_event.py` (Pydantic) |
| `training/` | Scripts for training & data prep | `train.py`, `losses.py`, `dataset.py` |
| `inference/` | CLI execution scripts | `predict.py` |
| `config.yaml` | Global hyperparameter store | Architecture, epochs, LR, thresholds |
| `models/` | Checkpoint storage | `.pth` weight files |
| `outputs/` | Inference artifacts | JSON, GeoJSON, overlay images |

---

## 19. APIs / Backend

- **Framework:** FastAPI running on Uvicorn.
- **Endpoints:**
  - `GET /api/v1/health`: Returns API status.
  - `GET /api/v1/model-info`: Returns model architecture, checkpoint status, and config parameters.
  - `POST /api/v1/detect`: Accepts an uploaded file (`UploadFile`) and an optional `threshold` query parameter. Temporarily saves the file, invokes `run_inference()`, cleans up the file, and returns the Pydantic `SpillEvent` model.
  - `GET /api/v1/results/{event_id}`: Fetches a stored JSON result from the disk.
- **Request Format:** `multipart/form-data` for image uploads.
- **Response Format:** `application/json`.
- **Error Handling:** Standard FastAPI `HTTPException` (500 for inference failures, 404 for missing results).

---

## 20. Frontend
Not explicitly specified/determinable from the existing project (Backend and CLI only).

---

## 21. Database / Storage
Not explicitly specified/determinable from the existing project (Outputs are serialized directly to the local filesystem in the `outputs/` directory).

---

## 22. Hardware & Environment

- **Python Version:** 3.10+ (Current environment running 3.13 based on execution logs).
- **GPU & CUDA:** Dynamically targets `cuda` via PyTorch if available. Designed to leverage NVIDIA GPUs (like RTX 3050) with CUDA 12+.
- **Environment Manager:** Standard Python `venv` (`.venv`).
- **OS:** Windows (Uses PowerShell `Activate.ps1` and paths).

---

## 23. Model Configuration

*Extracted directly from `config.yaml`:*
- **Architecture:** `unet`
- **Input Channels:** `3`
- **Number of Classes:** `1`
- **Hidden Dimensions (Features):** `[64, 128, 256, 512]`
- **Target Input Size:** `[256, 256]`
- **Probability Threshold:** `0.5`
- **Morphology Kernel Size:** `3`
- **Minimum Region Pixels:** `50`
- **Batch Size:** `16`
- **Learning Rate:** `0.001`
- **Epochs:** `10`
- **Weight Decay:** `0.0001`
- **Validation Min Area:** `0.001` km²
- **Validation Max Area:** `5000.0` km²
- **Validation Min Confidence:** `0.3`

---

## 24. Design Decisions — MOST IMPORTANT

**Why a U-Net architecture?**
U-Net is the gold standard for medical and remote-sensing image segmentation. Its skip-connections allow the network to combine deep, low-resolution semantic context (is this general shape a slick?) with shallow, high-resolution spatial details (where exactly is the boundary of the slick?).

**Why 3 Input Channels (RGB) instead of 1 (SAR intensity)?**
The project was refactored to support optical Earth Observation (EO) data. Optical sensors provide visible spectrum reflectance, which captures oil sheen coloration, solar reflection dampening, and textural variations that a single-channel radar backscatter cannot provide, potentially reducing "look-alike" false positives.

**Why Combined BCE + Dice Loss?**
Oil slicks are tiny anomalies compared to the massive vastness of the ocean background. BCE alone struggles with severe class imbalance. Dice loss directly maximizes the Intersection-over-Union metric, forcing the network to care about the structural shape of the minority class.

**Why Morphological Opening and Closing?**
Satellite imagery suffers from speckle noise, sunglint, and sensor artifacts. Morphological opening removes tiny isolated false-positive pixels (salt noise). Closing bridges small artificial gaps inside a legitimate oil slick (pepper noise).

**Why project coordinates to ESRI:54009 (World Mollweide)?**
Geographic coordinates (Lat/Lon EPSG:4326) are angular measurements; calculating planar area ($km^2$) directly on them is mathematically invalid and highly distorted near the poles. Projecting the geometry into an equal-area CRS ensures physical measurements are scientifically accurate.

---

## 25. End-to-End Example

**1. Input:** User uploads `mados_scene_01.tif` to `/api/v1/detect`.
**2. Preprocessing:** `EOPreprocessor` opens the TIFF, converts it to 3-channel RGB, resizes it to 256x256, divides by 255.0, and converts to a PyTorch tensor `[1, 3, 256, 256]`.
**3. Model:** The PyTorch U-Net passes the tensor through the encoder/decoder and outputs logits. A Sigmoid function converts logits to a probability map (values 0.0 to 1.0).
**4. Post-processing:** `MaskProcessor` thresholds the map at `0.5`, applies a 3x3 morphology disk to clean noise, and identifies one connected region of 450 pixels with a mean confidence of 0.87.
**5. Geospatial:** `GeospatialProcessor` reads the Affine transform from the TIFF metadata, converts the 450 pixels into a Shapely polygon, projects it to equal-area, and calculates an area of 4.2 km².
**6. Validation:** `SpillValidator` confirms 4.2 km² is between 0.001 and 5000, and confidence (0.87) > 0.3. Status set to `VALID`.
**7. Final Output:** The API returns a `SpillEvent` JSON containing the polygon GeoJSON, centroid, and area, which the user can load directly into GIS software.

---

## 26. Presentation / Viva Preparation

### Basic Questions
**Q: What is the main objective of this project?**
A: To automate the detection and geographic characterization of marine oil spills using a deep learning segmentation pipeline, outputting structured data that can be used for environmental response or downstream drift modeling.

### Technology Questions
**Q: What frameworks do you use?**
A: PyTorch for the core deep learning model, FastAPI for the backend service, OpenCV and scikit-image for computer vision post-processing, and Rasterio/Shapely for geographic extraction.

### AI/ML Questions
**Q: What model architecture is used and why?**
A: A PyTorch U-Net. The symmetric encoder-decoder structure with skip connections is optimal for dense prediction tasks like segmentation, where preserving the exact spatial boundaries of the anomaly is critical.

**Q: What is the loss function?**
A: A combined Binary Cross Entropy (BCE) and Dice Loss, weighted 50/50. This handles the extreme class imbalance of ocean imagery by penalizing pixel-wise errors while directly optimizing for region overlap.

### Data Questions
**Q: How many input channels are there and what do they represent?**
A: 3 channels. They represent the Red, Green, and Blue (RGB) optical bands of Earth Observation imagery. This allows the network to learn spectral reflectance signatures of oil sheens.

### Pipeline Questions
**Q: What happens after the model makes a prediction?**
A: The raw probability map is thresholded to create a binary mask. We apply morphological opening and closing to remove noise. Then, connected component analysis extracts distinct regions. Rasterio maps these pixel regions to geographic polygons, and we calculate physical area in square kilometers before running QC validation.

---

## 27. Quick Reference — ONE-PAGE TECHNICAL SUMMARY

- **Project:** OceanTrace Agent 1.
- **Problem:** Automated oil spill detection.
- **Input Shape:** `[Batch, 3, 256, 256]`.
- **Input Channels:** 3 (RGB Earth Observation data).
- **Model:** PyTorch U-Net (Custom implementation).
- **Loss:** `0.5 * BCEWithLogits + 0.5 * DiceLoss`.
- **Optimizer:** Adam (LR: 0.001, Weight Decay: 1e-4).
- **Preprocessing:** Bilinear resize to 256x256, Min-Max normalization (divide by 255).
- **Postprocessing:** Probability Thresholding (0.5), Morphological Open/Close (kernel=3), Connected Components filtering (<50 pixels removed).
- **Geospatial:** Rasterio Affine mapping, Shapely Polygons, projected to ESRI:54009 for km² area calculation.
- **Output:** Pydantic-validated `SpillEvent` JSON + standard GeoJSON.
- **Backend:** FastAPI + Uvicorn.
- **Key Decision:** Using U-Net for boundary precision; using combined BCE+Dice to solve severe class imbalance; calculating area in an equal-area projection rather than geographic degrees.
