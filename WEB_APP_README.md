# OceanTrace AI Web Application

A production-ready full-stack web application for the Sentinel-1 SAR Oil Spill Segmentation model.

## Folder Structure
- `/backend`: FastAPI backend that loads the tuned PyTorch model and executes the inference pipeline.
- `/frontend`: Premium React SPA built with Vite, featuring dynamic visual dashboards.

## Running the Backend

The backend requires the OceanTrace environment with its specific ML dependencies.

1. Activate your virtual environment:
   ```bash
   .venv\Scripts\activate
   ```
2. Install the backend-specific requirements (FastAPI, Uvicorn, python-multipart):
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Start the FastAPI server (it runs on port 8000):
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

*The backend will automatically load the model specified in `config.yaml` on startup.*

## Running the Frontend

The frontend is a lightweight React app.

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install Node dependencies (only needed the first time):
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

Open the URL provided by Vite (usually `http://localhost:5173`) in your browser to interact with the model!
