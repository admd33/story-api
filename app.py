from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import story_generator  # Import your existing story generation script
from diffusers import StableDiffusionPipeline
import torch

app = FastAPI()

# ✅ Load Stable Diffusion model for fantasy-style image generation
model_id = "dreamlike-art/dreamlike-photoreal-2.0"  # Change to another model if needed
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

# ✅ Request format for generating stories
class StoryRequest(BaseModel):
    theme: str
    reading_level: str

# ✅ Request format for generating images
class ImageRequest(BaseModel):
    story: str

@app.post("/generate_story")
def generate_story_endpoint(request: StoryRequest):
    """Generate a story based on theme and reading level."""
    result = story_generator.generate_story_and_questions(request.theme, request.reading_level)
    return {
        "theme": request.theme,
        "reading_level": request.reading_level,
        "story": result["story"],
        "questions": result["questions"]
    }

@app.post("/generate_image")
def generate_image(request: ImageRequest):
    """Generate a fantasy-style image based on the story text."""
    fantasy_prompt = (
        f"A beautiful fantasy-style illustration: {request.story}, in a colorful dreamlike world, "
        "storybook style, vibrant, highly detailed, whimsical, glowing colors, cinematic lighting"
    )

    # ✅ Generate Image
    image = pipe(fantasy_prompt).images[0]
    
    # ✅ Save Image
    image_path = "generated_image.png"
    image.save(image_path)

    return {"message": "Image generated successfully!", "image_path": image_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
