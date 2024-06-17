from flask import Flask, request, jsonify
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
import openai
import os

app = Flask(__name__)

# Load the translation model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# OpenAI API key setup (ensure the key is set in the environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_text(source_lang, target_lang, text):
    tokenizer.src_lang = source_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

def make_text_formal(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Please rephrase the following text to make it more formal:\n\n{text}",
        max_tokens=1000
    )
    formal_text = response.choices[0].text.strip()
    return formal_text

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    target_lang = data['target_language']
    source_lang = data['source_language']
    text = data['text']
    formal = data['formal']
    
    translated_text = translate_text(source_lang, target_lang, text)
    
    if formal:
        translated_text = make_text_formal(translated_text)
    
    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    app.run()
