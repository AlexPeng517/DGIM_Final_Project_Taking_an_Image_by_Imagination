# DGIM Final Project: Taking an Image by Imagination

This project explores camera-aligned image generation and editing using Gemini analysis and Stable Diffusion with camera-token embeddings.

## Project Structure (Repository Root: `src/`)

This repository is initialized within the `src/` directory.

- `ui/interface.py`: Main entry point for the Gradio User Interface.
- `engine/`: Core logic for image generation, analysis, and camera embeddings.
  - `cameraEngine.py`: Implementation of `CameraAlignedEditingEngine` and `PnPFeatureGuidedEditor`.
  - `chat.py`: `ChatEngine` for Gemini-based image analysis.
- `evaluate_methods.py`: Script for evaluating and comparing different generation methods.
- `config.py`: Configuration and environment variable management.

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (Recommended: 16GB+ VRAM)
- [Optional] `controlnet-aux` if using depth/canny preprocessing.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/AlexPeng517/DGIM_Final_Project_Taking_an_Image_by_Imagination.git
    cd DGIM_Final_Project_Taking_an_Image_by_Imagination
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment Variables**:
    Create a `.env` file in the root of the `src/` directory (or use the one in the project root if running from there) with your API keys:
    ```env
    GOOGLE_API_KEY=your_gemini_api_key
    OPENAI_API_KEY=your_openai_api_key
    ```

## Usage

### 🎨 Launching the Gradio Assistant
The UI allows you to upload an image, analyze its technical flaws via Gemini, and generate improved versions using local diffusion models or Gemini Nano.

```bash
python ui/interface.py
```
*Note: Default configuration uses CUDA:1 (configurable in `config.py` and `ui/interface.py`).*

### 📊 Running Evaluation
To compare different generation methods (Baseline, CLIP PnP, DINO PnP) across various metrics (LPIPS, CLIP Similarity, FID, PSNR):

```bash
python evaluate_methods.py
```

## Core Models

- **Base Diffusion**: [Stable Diffusion 2.1 Base](https://huggingface.co/Manojb/stable-diffusion-2-1-base)
- **ControlNet**: [SD2.1 Depth](https://huggingface.co/thibaud/controlnet-sd21-depth-diffusers)
- **Camera Tokens**: [Camera-Settings-as-Tokens-SD2](https://huggingface.co/ishengfang/Camera-Settings-as-Tokens-SD2)

## Acknowledgements

Developed as part of the DGIM Final Project.