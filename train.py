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


def rnn_train_epoch_batch(dataset, model, optimizer, criterion, device, epoch):
    # Set the model to training mode
    model.train()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0

    for (x, y) in tqdm(dataloader, desc=f'Epoch [{epoch + 1}/{args.num_epoch}]', leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        hidden = model.init_hidden(args.batch_size, device)
        loss = 0

        for c in range(args.chunk_len):
            output, hidden = model(x[:,c], hidden)
            loss += criterion(output.view(-1, dataset.n_characters), y[:, c])  # Reshape output for loss calculation

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def gpt_train(model, dataloader, optimizer, scheduler, device, epochs):
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
        for i, batch in tqdm(enumerate(dataloader), desc=f'Epoch [{epoch + 1}/{args.num_epoch}]', leave=False):
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
            # progress_percentage = int((i + 1) / total_batches * 100)
            # if progress_percentage % 10 == 0 and progress_percentage != last_percentage_logged:
            #     print(f"Progress: {progress_percentage}%")
            #     last_percentage_logged = progress_percentage

        print(f"Finished epoch {epoch + 1}/{epochs}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_layer', type=int, default=1) # The number of hidden layers in the RNN
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=100) # The size of hidden layers in RNN
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--data_path', type=str, default="D:\\Software\\Python\\Data\\all_lyrics.txt")
    parser.add_argument('--model_name', type=str, default="GPT")
    parser.add_argument('--chunk_len', type=int, default=128) # The chunk size when using character RNN
    parser.add_argument('--model_path', type=str, default="D:\\Software\\Python\\PACSS\\project\\gpt2") # The model path of GPT2 when use GPT2
    parser.add_argument('--sequence_length', type=int, default=50) # GPT uses
    parser.add_argument('--save_path', type=str, default="D:\\Software\\Python\\PACSS\\project\\gpt2.pth") # The path to save the model
    parser.add_argument('--use_saved', type=bool, default=False) # True if you used saved model arguments
    parser.add_argument('--del_bad_word', type=bool, default=True)
    parser.add_argument('--bad_word_path', type=str, default="D:\\Software\\Python\\Data\\bad_words_en.txt")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TextDataset(args)
    if args.model_name == 'RNN':
        model = RNN(dataset.n_characters, args.hidden_size, dataset.n_characters, args.num_layer).to(device)
    elif args.model_name == 'GPT':
        model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    else:
        parser.error(f"Invalid model option.")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.use_saved == False:
        # Set the model to training mode
        model.train()

        # Create an Adam optimizer that will be used to update the weights of the model
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        if args.model_name == 'RNN':
            # Define the Loss function, which is used to compare the output of the model
            # with the real target in the training process
            criterion = nn.CrossEntropyLoss()
            time_start = time.time()
            # Used to store the average loss per training cycle
            all_losses = []
            # Used to calculate the average loss for each training cycle
            loss_avg = 0
            # In the training cycle, the model will be trained and evaluated in each epoch
            # Iteration is performed on each epoch
            for epoch in range(1, args.num_epoch + 1):
                # Select a random training sample in each epoch and perform one-step training
                loss = rnn_train_epoch_batch(dataset, model, optimizer, criterion, device, epoch)
                # Accumulate the losses for each epoch for subsequent calculation of average losses
                loss_avg += loss
                # For every 100 epochs, print the training progress and average loss, and then reset the average loss
                if epoch % 100 == 0:
                    print('[%s (%d %d%%) %.4f]' % (time.time() - time_start, epoch, epoch / args.num_epoch * 100, loss))
                    all_losses.append(loss_avg / 100)
                    loss_avg = 0

            prompt = 'are'
            print(generate(prompt, model, device, dataset.index_to_char))
            prompt = 'here'
            print(generate(prompt, model, device, dataset.index_to_char))
        elif args.model_name == 'GPT':
            total_steps = len(dataloader) * args.num_epoch
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
            gpt_train(model, dataloader, optimizer, scheduler, device, args.num_epoch)
            prompt = "fly"
            new_lyrics = gpt_generate_lyrics(prompt, model, dataset.tokenizer, device)
            print(new_lyrics)
            new_lyrics = gpt_generate_lyrics(prompt, model, dataset.tokenizer, device)
            print(new_lyrics)

        torch.save(model.state_dict(), args.save_path)
    else:
        model.load_state_dict(torch.load(args.save_path))
        if args.model_name == 'RNN':

            prompt = 'are'
            print(generate(prompt, model, device, dataset.index_to_char))
            prompt = 'here'
            print(generate(prompt, model, device, dataset.index_to_char))

        elif args.model_name == 'GPT':
            prompt = "fly"
            new_lyrics = gpt_generate_lyrics(prompt, model, dataset.tokenizer, device)
            print(new_lyrics)
            new_lyrics = gpt_generate_lyrics(prompt, model, dataset.tokenizer, device)
            print(new_lyrics)

