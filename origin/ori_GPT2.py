import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
import re
import os


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
    file_path = 'D:\\Software\\Python\\Data\\all_lyrics.txt'
    model_path = 'D:\\Software\\Python\\PACSS\\project\\gpt2'
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

    prompt = "fly"
    new_lyrics = generate_lyrics(prompt, model, tokenizer, device)
    print(new_lyrics)
    new_lyrics = generate_lyrics(prompt, model, tokenizer, device)
    print(new_lyrics)
if __name__ == "__main__":
    main()

