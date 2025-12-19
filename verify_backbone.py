
import sys
import os

# Init paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
src_engine_dir = os.path.join(project_root, "src", "engine")

if project_root not in sys.path:
    sys.path.append(project_root)
if src_engine_dir not in sys.path:
    sys.path.append(src_engine_dir)

from src.engine.chat import ChatEngine

def test_init():
    print("Attempting to initialize ChatEngine...")
    try:
        engine = ChatEngine()
        print("ChatEngine initialized successfully.")
        print(f"Backbone type: {type(engine.camera_pnp)}")
        # Check if it is indeed the Enhanced version
        if "EnhancedPnPFeatureGuidedEditor" in str(type(engine.camera_pnp)):
             print("Verification PASSED: Backbone is EnhancedPnPFeatureGuidedEditor")
        else:
             print("Verification FAILED: Backbone is NOT EnhancedPnPFeatureGuidedEditor")
             exit(1)
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        exit(1)

if __name__ == "__main__":
    test_init()
