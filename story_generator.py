import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ✅ Use /tmp instead of /app (Render allows writing here)
cache_dir = "/tmp/huggingface_models"
os.makedirs(cache_dir, exist_ok=True)

# Define Hugging Face model names
story_repo_name = "abdalraheemdmd/story-api"
question_repo_name = "abdalraheemdmd/question-gene"

# Load Story Generation Model (Use local cache if available)
story_model_path = os.path.join(cache_dir, "story-model")
if not os.path.exists(story_model_path):
    print("Downloading story model...")
    story_tokenizer = GPT2Tokenizer.from_pretrained(story_repo_name, cache_dir=story_model_path)
    story_model = GPT2LMHeadModel.from_pretrained(story_repo_name, cache_dir=story_model_path)
else:
    print("Loading cached story model...")
    story_tokenizer = GPT2Tokenizer.from_pretrained(story_model_path)
    story_model = GPT2LMHeadModel.from_pretrained(story_model_path)

# Load Question Generation Model (Use local cache if available)
question_model_path = os.path.join(cache_dir, "question-model")
if not os.path.exists(question_model_path):
    print("Downloading question model...")
    question_tokenizer = GPT2Tokenizer.from_pretrained(question_repo_name, cache_dir=question_model_path)
    question_model = GPT2LMHeadModel.from_pretrained(question_repo_name, cache_dir=question_model_path)
else:
    print("Loading cached question model...")
    question_tokenizer = GPT2Tokenizer.from_pretrained(question_model_path)
    question_model = GPT2LMHeadModel.from_pretrained(question_model_path)

def generate_story(theme, reading_level, max_length=700, temperature=0.7):
    """Generates a story based on theme and reading level."""
    prompt = f"A {reading_level} story about {theme}:"
    input_ids = story_tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output = story_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=story_tokenizer.eos_token_id,
            eos_token_id=story_tokenizer.eos_token_id
        )

    return story_tokenizer.decode(output[0], skip_special_tokens=True)

def generate_questions(story, max_length=300, temperature=0.7):
    """Generates questions based on the generated story."""
    prompt = f"Generate comprehension questions for the following story:\n\n{story}"
    input_ids = question_tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output = question_model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=question_tokenizer.eos_token_id,
            eos_token_id=question_tokenizer.eos_token_id
        )

    return question_tokenizer.decode(output[0], skip_special_tokens=True)

def generate_story_and_questions(theme, reading_level):
    """Generates a story and corresponding questions."""
    story = generate_story(theme, reading_level)
    questions = generate_questions(story)
    return {"story": story, "questions": questions}
