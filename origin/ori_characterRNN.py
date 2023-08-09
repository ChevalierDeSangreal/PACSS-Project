import torch
import torch.nn as nn
import string
import random
import time
import math
import re

# Step 1: Load and process the data
with open("D:\\Software\\Python\\Data\\all_lyrics.txt", 'r', encoding='utf-8') as f:
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

prompt = 'are'
print(generate(prompt))
prompt = 'here'
print(generate(prompt))