import torch
import random

# Receive a string and return a tensor, where each character in the string is converted to its corresponding index
def char2tensor(string, char_to_index):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = char_to_index[string[c]]
    return tensor

def random_training_set(file_len, chunk_len, file, char_to_index, device):
    # Randomly obtain data to avoid overfitting
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    chunk =  file[start_index:end_index]
    # The goal is to predict the next character, so we don't need the last character as input, but as the target
    inp = char2tensor(chunk[:-1], char_to_index)
    # The target tensor is the result of moving the input tensor
    # one bit to the right, because our task is to predict the next character
    target = char2tensor(chunk[1:], char_to_index)
    return inp, target

# Using a trained RNN model to generate new text
def generate(prompt, model, device, index_to_char, temperature=0.8):
    # Initialize hidden state
    hidden = model.init_hidden().to(device)
    # Convert the prompt string into a tensor, which will serve as the initial input for the model
    prime_input = char2tensor(prompt).to(device)
    # Initialize the generated string as a prompt string
    predicted = prompt
    # Traverse each character in the prompt string (except for the last character)
    # and input them with the current hidden state into the model.
    for p in range(len(prompt) - 1):
        _, hidden = model(prime_input[p], hidden)
    # Obtain the last character in the prompt string, which will be used as input for the next step
    inp = prime_input[-1]
    # This loop will continue until a line break (' n') is generated
    while True:
        # Input the current input character and hidden state into the model to obtain the output and new hidden state
        output, hidden = model(inp, hidden)
        # The output of the model is converted into a probability distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        # Randomly extract an index of a character from the probability distribution of the
        # next character predicted by the model
        top_i = torch.multinomial(output_dist, 1)[0]
        # Convert the selected character index to the corresponding character
        predicted_char = index_to_char[top_i.item()]
        # Add newly generated characters to the output string
        predicted += predicted_char
        # Convert the newly generated character into a tensor, which will be used as input for the next step
        inp = char2tensor(predicted_char).to(device)
        # If the generated character is a newline character, the loop ends
        if predicted_char == '\n':  # 如果是'\n'就停止
            break
    return predicted

def gpt_generate_lyrics(prompt, model, tokenizer, device, max_length=100):
    # Set the model to evaluation mode
    model.eval()
    # Encode the input prompt string into an acceptable input ID format for the model
    # using a word breaker and convert it into a PyTorch tensor
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # Create corresponding attention masks
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)
    # Call the generate method of the model to start generating text
    output = model.generate(
        input_ids,
        # Pass attention mask
        attention_mask=attention_mask,
        # Set the maximum length of generated text
        max_length=max_length,
        # During the generation process, if the generated sequence length does not reach
        # max_Length, it will be filled with pad_token_id
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        do_sample=True
    )

    return tokenizer.decode(output[0])