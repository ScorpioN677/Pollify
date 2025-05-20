from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(_name_)

# Load model & tokenizer once at startup
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=100,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({"response": response})

if _name_ == "_main_":
    app.run(host="0.0.0.0", port=5000)
