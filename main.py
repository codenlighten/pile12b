# Step 1: Install the transformers library if you haven't already. 
# You can do this using pip:
# pip install transformers

# Step 2: Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM

# Step 3: Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-12b")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-12b")

# Step 4: Text Generation
def generate_text(prompt, max_length=50):
    # Encode the input prompt to get the input_ids
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Generate a sequence of tokens in response to the input_ids
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)

    # Decode the output to a human-readable format
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Once upon a time"
print(generate_text(prompt))
