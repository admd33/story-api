import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load Story Generator from Hugging Face
story_repo_name = "abdalraheemdmd/story-api"  # Change this to your actual repo
story_tokenizer = GPT2Tokenizer.from_pretrained(story_repo_name)
story_model = GPT2LMHeadModel.from_pretrained(story_repo_name)

# Load Question Generator from Hugging Face
question_repo_name = "abdalraheemdmd/story-generator-model"  # Change this to your actual repo
question_tokenizer = GPT2Tokenizer.from_pretrained(question_repo_name)
question_model = GPT2LMHeadModel.from_pretrained(question_repo_name)

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
