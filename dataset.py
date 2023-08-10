import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import re

# Define a custom dataset class
class TextDataset(Dataset):
    def __init__(self, args):

        self.args = args

        print("Preparing data...")
        # Load and process the data
        

        if args.model_name == 'RNN':
            with open(args.data_path, 'r', encoding='utf-8') as f:
                file = f.read().lower() # Convert all letters to lowercase
                file = re.sub(r"[^a-zA-Z0-9'\s]", '', file )
            file_len = len(file)
            print('Prepareing complete. got date file, length =', file_len)
            # Get all unique characters in the file
            all_characters = sorted(set(file))

            # Calculate the number of unique characters
            self.n_characters = len(all_characters)

            self.char_to_index = {char: index for index, char in enumerate(all_characters)}
            self.index_to_char = {index: char for index, char in enumerate(all_characters)}

            self.data = file
            self.chunk_len = self.args.chunk_len
            self.data_len = len(self.data) - self.args.chunk_len
        elif args.model_name == 'GPT':
            self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
            # Set the filling symbol of the word breaker as the ending symbol
            self.tokenizer.pad_token = self.tokenizer.eos_token

            with open(args.data_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Using clean_line each line using the line method
            # Condition line.strip() and self.clean_line(line.strip()) ensure that the line is not
            # only non empty, but also non empty after cleaning
            self.lyrics = [self.clean_line(line.strip()) for line in lines if line.strip() and self.clean_line(line.strip())]
            self.data_len = len(self.lyrics)

    def clean_line(self, line):
        # Convert text to lowercase
        line = line.lower()
        # Using regular expressions to remove special characters
        line = re.sub(r"[^a-zA-Z0-9\s]", '', line)
        return line.strip()
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        if self.args.model_name == 'RNN':
            chunk = self.data[idx:idx + self.chunk_len]
            target = self.data[idx + 1:idx + self.chunk_len + 1]
            chunk_indices = [self.char_to_index[c] for c in chunk]
            target_indices = [self.char_to_index[c] for c in target]
            return torch.tensor(chunk_indices), torch.tensor(target_indices)
        elif self.args.model_name == 'GPT':
            lyric = self.lyrics[idx]
            tokens = self.tokenizer.encode_plus(lyric,
                                            truncation=True,
                                            max_length=self.args.sequence_length,
                                            padding='max_length',
                                            return_tensors='pt')

            return tokens['input_ids'].view(-1), tokens['attention_mask'].view(-1)

