from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

# ✅ Use `nitrosocke/Arcane-Diffusion` for high-quality cartoon-style images
model_id = "nitrosocke/Arcane-Diffusion"

# ✅ Check for GPU, else use CPU (Important for Render!)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load the AI Model ONCE (so it doesn’t reload every request)
print("Loading Cartoon Model...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)
print("Model Loaded!")

class ImageRequest(BaseModel):
    story: str

@app.post("/generate_image")
def generate_image(request: ImageRequest):
    """Generate a high-quality cartoon-style image based on the story text."""
    
    cartoon_prompt = (
        f"A colorful, cartoon-style illustration of: {request.story}, "
        "storybook fantasy world, highly detailed, vibrant colors, soft edges, clear lighting"
    )

    print("Generating Image for:", cartoon_prompt[:100])

    # ✅ Generate Image (Directly in HD 1024x1024)
    image = pipe(cartoon_prompt, width=1024, height=1024).images[0]

    # ✅ Save Image
    image_path = "generated_image.png"
    image.save(image_path)

    print("Image Saved!")  

    return {"message": "Image generated successfully!", "image_path": image_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
