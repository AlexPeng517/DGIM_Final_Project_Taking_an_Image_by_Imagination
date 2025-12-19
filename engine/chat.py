import json
import base64
import os
from google import genai
from google.genai import types
from PIL import Image
from src.config import Config
from src.engine.cameraEngine import EnhancedPnPFeatureGuidedEditor, CameraSettings
import io

class ChatEngine:
    def __init__(self):
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment or .env file. Please create a .env file with your GOOGLE_API_KEY.")

        # New Client for both Analysis and "Nano Banana" (Gemini Edit)
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        
        # Initialize the "Camera As Token PnP" (Local Model)
        # Note: This loads models on init, which might be heavy.
        self.camera_pnp = EnhancedPnPFeatureGuidedEditor(
            model_id="Manojb/stable-diffusion-2-1-base",
            controlnet_model_id="thibaud/controlnet-sd21-depth-diffusers",
            camera_setting_embedding_id="ishengfang/Camera-Settings-as-Tokens-SD2",
            feature_type="clip",
            device=Config.DEVICE # Use device from config
        )

    def analyze_image(self, image_path: str, user_prompt: str = None) -> dict:
        """
        Analyzes an image and returns structured recommendations using Gemini (New SDK).
        """
        
        system_prompt = """
        You are an expert professional photographer and AI image assistant. 
        Analyze the input photo and identify technical issues (e.g., exposure, focus, noise).
        Provide a JSON response with the following structure:
        {
            "Reasons": ["list", "of", "issues"],
            "Recommendation": {
                "cameraSettings": {
                    "ShutterSpeed": "value (e.g. 1/500)",
                    "Aperture": "value (e.g. f/2.8)",
                    "ISO": "value (e.g. 100)",
                    "FocalLength": "value (e.g. 50mm)" 
                },
                "capture": ["suggestions", "for", "shooting"],
                "variations": ["creative", "suggestions"]
            },
            "SimulatedPrompt": "A detailed caption describing the scene as if it were perfectly exposed and shot."
        }
        Return ONLY the JSON string, no markdown formatting.
        """
        
        try:
            # Read image bytes for new SDK
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            prompt = user_prompt if user_prompt else "Analyze this image for technical improvements."
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash', # Or gemini-1.5-pro, updating to recommended flash model
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg', # Assuming jpeg for simplicity, could detect
                    ),
                    system_prompt + "\n\n" + prompt
                ]
            )
            
            content = response.text
            
            # Clean up potential markdown code blocks if gemini adds them
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            elif "```" in content:
                content = content.replace("```", "")
                
            return json.loads(content)
        except Exception as e:
            print(f"Error in ChatEngine.analyze_image: {e}")
            return {"error": str(e)}

    def invoke_camera_as_token_pnp(self, image_path: str, analysis_result: dict) -> Image.Image:
        """
        Invokes the Local Model (PnPFeatureGuidedEditor) with the recommended settings.
        Formerly 'invoke_nano_banana' (renamed).
        """
        try:
            recommendations = analysis_result.get("Recommendation", {}).get("cameraSettings", {})
            prompt = analysis_result.get("SimulatedPrompt", "Photorealistic image")
            
            # Parse settings with defaults
            iso = float(recommendations.get("ISO", 100))
            
            # Handle focal length parsing (remove 'mm')
            fl_str = str(recommendations.get("FocalLength", "50mm"))
            fl = float(''.join(filter(str.isdigit, fl_str))) if any(c.isdigit() for c in fl_str) else 50.0
            
            # Handle aperture parsing (remove 'f/')
            f_str = str(recommendations.get("Aperture", "f/4.0"))
            f_num = float(f_str.replace("f/", "").strip()) if "f/" in f_str else 4.0
            if f_num == 4.0 and any(c.isdigit() for c in f_str) and "f/" not in f_str:
                 try: 
                     f_num = float(f_str)
                 except: 
                     pass

            # Handle shutter speed parsing
            ss_str = str(recommendations.get("ShutterSpeed", "1/100"))
            if "/" in ss_str:
                num, dom = ss_str.split("/")
                exposure_time = float(num) / float(dom)
            else:
                try:
                    exposure_time = float(ss_str)
                except:
                    exposure_time = 0.01

            print(f"Invoking Camera As Token PnP with: ISO={iso}, F={f_num}, FL={fl}, Exp={exposure_time}")
            
            cam_settings = CameraSettings(
                focal_length=fl,
                f_number=f_num,
                iso_speed_rating=iso,
                exposure_time=exposure_time
            )
            
            source_image = Image.open(image_path).convert("RGB")
            
            # Run simulation
            result_image = self.camera_pnp.run_simulation(
                source_image=source_image,
                camera=cam_settings,
                prompt=prompt,
                strength=0.4,
                controlnet_scale=0.75,
                lora_scale=0.55
            )
            
            return result_image
            
        except Exception as e:
            print(f"Error invoking Camera As Token PnP: {e}")
            return None

    def invoke_nano_banana(self, image_path: str, analysis_result: dict = None, camera_settings: dict = None, prompt: str = None) -> Image.Image:
        """
        Invokes the Gemini Image-to-Image editing feature ("Nano Banana") with the specified settings.
        
        Args:
            image_path: Path to the input image.
            analysis_result: (Optional) The full JSON analysis result from analyze_image.
            camera_settings: (Optional) A dictionary of camera settings (ISO, Aperture, etc.) to OVERRIDE 
                             any settings found in analysis_result.
            prompt: (Optional) A specific scene description/prompt to OVERRIDE the SimulatedPrompt 
                    from analysis_result.

        Returns:
            Image.Image: The generated image, or None if failed.
        """
        try:
            print("Invoking Nano Banana (Gemini Edit)...")
            
            # 1. Determine Settings
            base_settings = {}
            if analysis_result:
                base_settings = analysis_result.get("Recommendation", {}).get("cameraSettings", {})
            
            # Explicit camera_settings take precedence
            if camera_settings:
                 recommendations = camera_settings
            else:
                 recommendations = base_settings

            # 2. Determine Prompt
            base_prompt = "A perfect photo"
            if analysis_result:
                base_prompt = analysis_result.get("SimulatedPrompt", base_prompt)
            
            simulated_prompt = prompt if prompt else base_prompt
            
            # Construct a prompt
            settings_desc = ", ".join([f"{k}: {v}" for k, v in recommendations.items()])
            
            prompt_text = (
                f"Edit this image to look like it was shot with these settings: {settings_desc}. "
                #f"The result should match this description: {simulated_prompt}"
            )
            
            print(f"Nano Banana Prompt: {prompt_text}")
            
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            response = self.client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt_text
                ],
                config=types.GenerateContentConfig(
                    response_modalities=['Image']
                )
            )
            
            # Extract image from response
            generated_image = None
             
            # Standard parts parsing from SDK
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        try:
                           generated_image = Image.open(io.BytesIO(part.inline_data.data))
                        except Exception as e:
                            print(f"Failed to open inline data as image: {e}")

            # If still None, check if the response object has helper methods or specific structure for the new SDK
            if generated_image is None and response.parts:
                 for part in response.parts:
                     # Check for as_image method available in some SDK versions
                     if hasattr(part, "as_image"):
                         try:
                             generated_image = part.as_image()
                         except Exception as e:
                             print(f"part.as_image() failed: {e}")

            if generated_image:
                 return generated_image
            else:
                 print("No image found in Gemini response parts.")
                 # Print text if any to debug
                 if response.text:
                     print(f"Gemini Text Response: {response.text}")
                 return None

        except Exception as e:
            print(f"Error invoking Nano Banana (Gemini): {e}")
            return None
