import torch.nn as nn
import torch

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
        output, hidden = self.rnn(input, hidden)  # Fix the order of input and hidden
        # Then, the output of GRU is decoded and reshaped through the decoding layer
        # print(output.shape)
        output = self.decoder(output)
        return output, hidden


    # This is an auxiliary function of the RNN class used to initialize the hidden state
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device)