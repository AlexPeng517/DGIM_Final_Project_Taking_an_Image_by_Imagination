
import os
# Ensure we map physical GPU 1 to logical GPU 0 and restrict visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import gradio as gr
import sys
import json
from PIL import Image
from dotenv import load_dotenv

# --- Path Patching ---
# Ensure the project root and src directories are in sys.path
# so that imports like `from src.engine.chat import ChatEngine` work,
# and internal imports in engine (like `from camera_embed ...`) work.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
src_engine_dir = os.path.join(project_root, "src", "engine")

if project_root not in sys.path:
    sys.path.append(project_root)
if src_engine_dir not in sys.path:
    # Essential for cameraEngine.py imports (e.g. `from camera_embed import ...`)
    sys.path.append(src_engine_dir)

# Now we can import the engine
from src.engine.chat import ChatEngine

# Load environment variables
load_dotenv()

# Global Engine Instance
# We utilize a global variable to load the model once.
chat_engine = None

def get_engine():
    global chat_engine
    if chat_engine is None:
        print("Initializing ChatEngine (this may take a moment)...")
        chat_engine = ChatEngine()
    return chat_engine

# --- Wrapper Functions ---

def analyze_wrapper(image_path: str):
    """
    Analyzes the image using Gemini and returns:
    - The raw JSON result (for display)
    - The extracted "SimulatedPrompt"
    - The extracted Camera Settings (FocalLength, Aperture, ShutterSpeed, ISO)
    """
    if not image_path:
        return None, "", "", "", "", 100

    engine = get_engine()
    print(f"Analyzing: {image_path}")
    
    # Call the engine's analysis
    analysis_result = engine.analyze_image(image_path)
    
    # Display raw JSON in the UI
    json_output = analysis_result
    
    # Extract recommendations to populate UI fields
    recommendations = analysis_result.get("Recommendation", {})
    settings = recommendations.get("cameraSettings", {})
    
    simulated_prompt = analysis_result.get("SimulatedPrompt", "")
    
    # Extract specific values with defaults
    focal_length = settings.get("FocalLength", "50mm")
    aperture = settings.get("Aperture", "f/4.0")
    shutter_speed = settings.get("ShutterSpeed", "1/100")
    iso = settings.get("ISO", 100)
    
    # Parse ISO to number if possible (UI expects number)
    try:
        iso = float(str(iso))
    except:
        iso = 100

    return (
        json_output,      # JSON Output
        simulated_prompt, # Prompt Textbox
        focal_length,     # Focal Length Textbox
        aperture,         # Aperture Textbox
        shutter_speed,    # Shutter Speed Textbox
        iso               # ISO Number
    )

def _construct_analysis_package(simulated_prompt, focal_length, aperture, shutter_speed, iso):
    """
    Helper to reconstruct the analysis dictionary from UI inputs.
    """
    return {
        "Recommendation": {
            "cameraSettings": {
                "FocalLength": str(focal_length),
                "Aperture": str(aperture),
                "ShutterSpeed": str(shutter_speed),
                "ISO": str(iso)
            }
        },
        "SimulatedPrompt": simulated_prompt
    }

def generate_pnp_wrapper(image_path: str, simulated_prompt, focal_length, aperture, shutter_speed, iso):
    """
    Invokes the Local Camera-as-Token PnP model.
    """
    if not image_path:
        return None
    
    engine = get_engine()
    
    # Reconstruct the analysis result dict from user inputs
    analysis_payload = _construct_analysis_package(
        simulated_prompt, focal_length, aperture, shutter_speed, iso
    )
    
    print("Generating with Camera PnP...")
    result_image = engine.invoke_camera_as_token_pnp(image_path, analysis_payload)
    return result_image

def generate_banana_wrapper(image_path: str, simulated_prompt, focal_length, aperture, shutter_speed, iso):
    """
    Invokes the Gemini Nano Banana model with explicit settings from UI.
    """
    if not image_path:
        return None
    
    engine = get_engine()
    
    # Construct settings dict directly from UI inputs
    camera_settings = {
        "FocalLength": str(focal_length),
        "Aperture": str(aperture),
        "ShutterSpeed": str(shutter_speed),
        "ISO": str(iso)
    }
    
    print(f"Generating with Nano Banana with overrides: {camera_settings}")
    
    # Pass explicit overrides. analysis_result can be None since we provide everything.
    result_image = engine.invoke_nano_banana(
        image_path=image_path,
        analysis_result=None, 
        camera_settings=camera_settings,
        prompt=simulated_prompt
    )
    return result_image

# --- UI Layout ---

def create_ui():
    with gr.Blocks(title="Gemini Camera Assistant") as app:
        gr.Markdown("# 📸 Gemini Camera Assistant")
        gr.Markdown("Upload a photo to analyze its technical flaws, then generate an improved version using **Camera PnP (Local)** or **Nano Banana (Gemini)**.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Input Image", height=400)
            
            with gr.Column(scale=1):
                analysis_output = gr.JSON(label="Gemini Analysis Result")
        
        analyze_btn = gr.Button("🔍 Analyze Image", variant="primary")
        
        gr.Markdown("### Recommended Settings (Editable)")
        
        with gr.Group():
            simulated_prompt_input = gr.Textbox(label="Simulated Prompt (Scene Description)", lines=2)
            with gr.Row():
                focal_length_input = gr.Textbox(label="Focal Length (e.g. 50mm)")
                aperture_input = gr.Textbox(label="Aperture (e.g. f/2.8)")
                shutter_speed_input = gr.Textbox(label="Shutter Speed (e.g. 1/500)")
                iso_input = gr.Number(label="ISO", value=100)
        
        gr.Markdown("### Generate Optimal Image")
        with gr.Row():
            gen_pnp_btn = gr.Button("✨ Generate (Camera PnP - Local)", variant="secondary")
            gen_banana_btn = gr.Button("🍌 Generate (Nano Banana - Gemini)", variant="secondary")
        
        with gr.Row():
            with gr.Column():
                output_pnp = gr.Image(label="Result: Camera PnP", interactive=False)
            with gr.Column():
                output_banana = gr.Image(label="Result: Nano Banana", interactive=False)

        # --- Event Wiring ---
        
        # Analyze Button
        analyze_btn.click(
            fn=analyze_wrapper,
            inputs=[image_input],
            outputs=[
                analysis_output,
                simulated_prompt_input,
                focal_length_input,
                aperture_input,
                shutter_speed_input,
                iso_input
            ]
        )
        
        # Generate PnP Button
        gen_pnp_btn.click(
            fn=generate_pnp_wrapper,
            inputs=[
                image_input,
                simulated_prompt_input,
                focal_length_input,
                aperture_input,
                shutter_speed_input,
                iso_input
            ],
            outputs=[output_pnp]
        )
        
        # Generate Banana Button
        gen_banana_btn.click(
            fn=generate_banana_wrapper,
            inputs=[
                image_input,
                simulated_prompt_input,
                focal_length_input,
                aperture_input,
                shutter_speed_input,
                iso_input
            ],
            outputs=[output_banana]
        )

    return app

if __name__ == "__main__":
    # Launch the app
    # allow_flagging="never" to keep it clean
    ui = create_ui()
    print("Launching Gradio Interface...")
    ui.launch(server_name="0.0.0.0", share=False)
