import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Model Settings
    # Use a standard SD model for the PoC, e.g., runwayml/stable-diffusion-v1-5 or similar
    # DIFFUSION_MODEL_ID = os.getenv("DIFFUSION_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    
    # Generation Settings
    # Enforce physical GPU 1 usage
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Since we restricted visibility to 1 GPU, it is now at index 0
    DEVICE = "cuda:0" if os.getenv("USE_CUDA", "true").lower() == "true" else "cpu"
    
    # Output Settings
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
