from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import story_generator  # Import your script

app = FastAPI()

class StoryRequest(BaseModel):
    theme: str
    reading_level: str

@app.post("/generate_story")
def generate_story_endpoint(request: StoryRequest):
    result = story_generator.generate_story_and_questions(request.theme, request.reading_level)
    return {
        "theme": request.theme,
        "reading_level": request.reading_level,
        "story": result["story"],
        "questions": result["questions"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
