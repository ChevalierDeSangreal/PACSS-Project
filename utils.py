from argparse import ArgumentParser
import torch
import torch.nn as nn
import string
import random
import time
import math
from model import RNN
from utils import *
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from dataset import *
from transformers import GPT2Tokenizer

def get_rand_line(lines):
    return random.choice(lines)





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


if __name__ == "__main__":

    num = 10

    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_layer', type=int, default=1) # The number of hidden layers in the RNN
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=100) # The size of hidden layers in RNN
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="D:\\Software\\Python\\Data\\all_lyrics.txt")
    parser.add_argument('--model_name', type=str, default="RNN") # This couldn't be changed here
    parser.add_argument('--chunk_len', type=int, default=128) # The chunk size when using character RNN
    parser.add_argument('--model_path', type=str, default="D:\\Software\\Python\\PACSS\\project\\gpt2") # The model path of GPT2 when use GPT2
    parser.add_argument('--sequence_length', type=int, default=50) # GPT uses
    parser.add_argument('--save_path_gpt2', type=str, default="D:\\Software\\Python\\PACSS\\project\\gpt2.pth") # The path to save the model
    parser.add_argument('--save_path_RNN', type=str, default="D:\\Software\\Python\\PACSS\\project\\RNN.pth") # The path to save the model
    parser.add_argument('--use_saved', type=bool, default=True) # True if you used saved model arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TextDataset(args)
    model_RNN = RNN(dataset.n_characters, args.hidden_size, dataset.n_characters, args.num_layer).to(device)
    model_gpt2 = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model_gpt2.load_state_dict(torch.load(args.save_path_gpt2))
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [dataset.clean_line(line.strip()) for line in lines if line.strip() and dataset.clean_line(line.strip())]


    for i in range(num):
        ori_line = get_rand_line(lines)
        input_line = ori_line[:len(ori_line)//2]

        res_RNN = generate(input_line, model_RNN, device, dataset.index_to_char)
        res_gpt2 = gpt_generate_lyrics(input_line, model_gpt2, tokenizer, device)

        print('-----------Number:', i+1, '-----------')
        print("Origin sentence: ", ori_line)
        print("Input sentence: ", input_line)
        print("RNN output: ", res_RNN)
        print("GPT2 output: ", res_gpt2)
    