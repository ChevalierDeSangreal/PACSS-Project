import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import string
import random
import time
import math
import re
import os
# Step 1: Load and process the data
with open("E:/dataset/all_lyrics.txt", 'r', encoding='utf-8') as f:
    file = f.read().lower() # Convert all letters to lowercase
    file = re.sub(r"[^a-zA-Z0-9\s]", '', file )
file_len = len(file)
print('file_len =', file_len)

# Get all unique characters in the file
all_characters = sorted(set(file))

# Calculate the number of unique characters
n_characters = len(all_characters)


char_to_index = {char: index for index, char in enumerate(all_characters)}
index_to_char = {index: char for index, char in enumerate(all_characters)}


# Convert file into chunks
chunk_len = 100
# Randomly obtain data to avoid overfitting
def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

# Step 2: Build the model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        # Defined an embedding layer that converts input indexes into dense vectors
        self.encoder = nn.Embedding(input_size, hidden_size)
        # Defined a GRU
        self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        # Defined a linear layer that converts the output of GRU into the final output
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # Change the shape of the input tensor to (1, N), and then convert
        # the character index into character embedding through the embedding layer
        input = self.encoder(input.view(1, -1))
        # The reshaped input and the previous hidden state are fed into the GRU layer
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
        # Then, the output of GRU is decoded and reshaped through the decoding layer
        output = self.decoder(output.view(1, -1))
        return output, hidden

    # This is an auxiliary function of the RNN class used to initialize the hidden state
    def init_hidden(self):
        return torch.zeros(self.n_layers, 1, self.hidden_size)

# Step 3: Training
# Receive a string and return a tensor, where each character in the string is converted to its corresponding index
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = char_to_index[string[c]]
    return tensor

def random_training_set():
    chunk = random_chunk()
    # The goal is to predict the next character, so we don't need the last character as input, but as the target
    inp = char_tensor(chunk[:-1])
    # The target tensor is the result of moving the input tensor
    # one bit to the right, because our task is to predict the next character
    target = char_tensor(chunk[1:])
    return inp, target

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(inp, target):
    # Initialize hidden state
    hidden = rnn.init_hidden().to(device)
    # Before proceeding with new optimization steps, it is necessary to set the gradient of the model to zero
    rnn.zero_grad()
    loss = 0

    inp = inp.to(device)
    target = target.to(device)
    # Loop through each character in the input sequence
    for c in range(chunk_len):
        # Input the current character and the previous hidden state into RNN to obtain output and a new hidden state
        output, hidden = rnn(inp[c], hidden)
        # Calculate the loss between the current output and the target and accumulate it into the total loss
        loss += criterion(output, torch.tensor([target[c]]).to(device))
    # After calculating the loss of all characters, calculate the gradient through backpropagation
    loss.backward()
    # Perform optimization steps to update the weights of the model
    optimizer.step()
    # Calculate and return the average loss per character
    return loss.item() / chunk_len

# Define the hyperparameters to be used during the training process
n_epochs = 3000
hidden_size = 100
n_layers = 1
lr = 0.0005
# Create an RNN model and transfer it to GPU
rnn = RNN(n_characters, hidden_size, n_characters, n_layers).to(device)
# Set the model to training mode
rnn.train()
# Create an Adam optimizer that will be used to update the weights of the model
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
# Define the Loss function, which is used to compare the output of the model
# with the real target in the training process
criterion = nn.CrossEntropyLoss()

start = time.time()
# Used to store the average loss per training cycle
all_losses = []
# Used to calculate the average loss for each training cycle
loss_avg = 0
# In the training cycle, the model will be trained and evaluated in each epoch
# Iteration is performed on each epoch
for epoch in range(1, n_epochs + 1):
    # Select a random training sample in each epoch and perform one-step training
    loss = train(*random_training_set())
    # Accumulate the losses for each epoch for subsequent calculation of average losses
    loss_avg += loss
    # For every 100 epochs, print the training progress and average loss, and then reset the average loss
    if epoch % 100 == 0:
        print('[%s (%d %d%%) %.4f]' % (time.time() - start, epoch, epoch / n_epochs * 100, loss))
        all_losses.append(loss_avg / 100)
        loss_avg = 0

# Step 4: Generating text
# Using a trained RNN model to generate new text
def generate(prompt, temperature=0.8):
    # Initialize hidden state
    hidden = rnn.init_hidden().to(device)
    # Convert the prompt string into a tensor, which will serve as the initial input for the model
    prime_input = char_tensor(prompt).to(device)
    # Initialize the generated string as a prompt string
    predicted = prompt
    # Traverse each character in the prompt string (except for the last character)
    # and input them with the current hidden state into the model.
    for p in range(len(prompt) - 1):
        _, hidden = rnn(prime_input[p], hidden)
    # Obtain the last character in the prompt string, which will be used as input for the next step
    inp = prime_input[-1]
    # This loop will continue until a line break (' n') is generated
    while True:
        # Input the current input character and hidden state into the model to obtain the output and new hidden state
        output, hidden = rnn(inp, hidden)
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
        inp = char_tensor(predicted_char).to(device)
        # If the generated character is a newline character, the loop ends
        if predicted_char == '\n':  # 如果是'\n'就停止
            break
    return predicted

prompt0 = 'are'
print("CharacterRNN:", generate(prompt0))
prompt0 = 'here'
print("CharacterRNN:", generate(prompt0))


class LyricsDataset(Dataset):
    def __init__(self, file_path, tokenizer, sequence_length):
        super().__init__()

        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Using _clean_line each line using the line method
        # Condition line.strip() and self._clean_line(line.strip()) ensure that the line is not
        # only non empty, but also non empty after cleaning
        self.lyrics = [self._clean_line(line.strip()) for line in lines if line.strip() and self._clean_line(line.strip())]

    def _clean_line(self, line):
        # Convert text to lowercase
        line = line.lower()
        # Using regular expressions to remove special characters
        line = re.sub(r"[^a-zA-Z0-9\s]", '', line)
        return line.strip()


    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, index):
        lyric = self.lyrics[index]
        tokens = self.tokenizer.encode_plus(lyric,
                                            truncation=True,
                                            max_length=self.sequence_length,
                                            padding='max_length',
                                            return_tensors='pt')

        return tokens['input_ids'].view(-1), tokens['attention_mask'].view(-1)


def train(model, dataloader, optimizer, scheduler, device, epochs):
    # Set the model to training mode
    model.train()
    # Obtain the total number of batches in the data loader
    total_batches = len(dataloader)
    # Loop through each epoch, each epoch is a complete dataset traversal
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        # Initialize a variable to track the percentage of progress recorded last, ensuring that
        # progress information is not printed repeatedly
        last_percentage_logged = -1
        # Loop through each batch in the data loader
        for i, batch in enumerate(dataloader):
            # Extract the input ID and attention mask from the batch and move them to the GPU
            input_ids, attention_mask = [b.to(device) for b in batch]
            # Pass the input ID and attention mask to the model and receive the output. We use the input ID as
            # the label here because we are conducting self supervised training
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            # Obtain losses from model output.
            loss = outputs.loss
            # Calculate gradient through loss value
            loss.backward()
            # Using the optimizer to update model parameters
            optimizer.step()
            # Update Learning rate Scheduler
            scheduler.step()
            # Clear the calculated gradient and prepare for the next gradient calculation
            optimizer.zero_grad()

            # Print progress for every 10%
            progress_percentage = int((i + 1) / total_batches * 100)
            if progress_percentage % 10 == 0 and progress_percentage != last_percentage_logged:
                print(f"Progress: {progress_percentage}%")
                last_percentage_logged = progress_percentage

        print(f"Finished epoch {epoch + 1}/{epochs}")


def generate_lyrics(prompt, model, tokenizer, device, max_length=100):
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




def main():
    file_path = 'E:/dataset/all_lyrics.txt'
    model_path = 'E:/model/gpt2'
    # Set the length of each input sequence
    sequence_length = 50
    batch_size = 32
    learning_rate = 3e-5
    epochs = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load pre trained model word splitter
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # Load pre trained model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)

    # Set the filling symbol of the word breaker as the ending symbol
    tokenizer.pad_token = tokenizer.eos_token
    # Create a dataset based on the given file path, word breaker, and sequence length
    # using the previously defined lyrics dataset class
    dataset = LyricsDataset(file_path, tokenizer, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_steps = len(dataloader) * epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # This scheduler allows you to linearly increase the Learning rate from a low initial value
    # to a predetermined initial Learning rate, and then linearly decrease as the training progresses
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    # Start training the model
    train(model, dataloader, optimizer, scheduler, device, epochs)

    prompt = "are"
    new_lyrics = generate_lyrics(prompt, model, tokenizer, device)
    print("GPT2:", new_lyrics)
    prompt = "here"
    new_lyrics = generate_lyrics(prompt, model, tokenizer, device)
    print("GPT2:", new_lyrics)



if __name__ == "__main__":
    main()
