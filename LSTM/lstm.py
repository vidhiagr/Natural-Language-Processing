import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scripts.utils import convert_line2idx
from scripts import utils
import argparse
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
import math
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_to_subsequences(data, k, padding_token=384):
    total_subsequences = sum((len(line) - 1) // k + 1 for line in data)
    input_sequences = np.full((total_subsequences, k), padding_token, dtype=np.int32)
    target_sequences = np.full((total_subsequences, k), padding_token, dtype=np.int32)

    idx = 0
    for line in data:
        for i in range(0, len(line), k):
            end = min(i + k, len(line))
            inp = line[i:end]
            target = line[i + 1:end + 1]
            input_sequences[idx, :len(inp)] = inp
            target_sequences[idx, :len(target)] = target
            if len(target) < k:
                target_sequences[idx, len(target):] = padding_token
            idx += 1
    
    input_sequences_tensor = torch.tensor(input_sequences, dtype=torch.long)
    target_sequences_tensor = torch.tensor(target_sequences, dtype=torch.long)
    
    dataset = TensorDataset(input_sequences_tensor, target_sequences_tensor)

    dataloader = DataLoader(dataset, batch_size=128) 
    
    return dataloader



class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim  
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.fc1 = nn.Linear(hidden_dim, 300)  # First linear layer
        self.reLU = nn.ReLU()
        self.fc2 = nn.Linear(300, output_dim)  # Second linear layer

    def forward(self, input, hidden,target=None):

        embedding = self.embedding(input)
        output, hidden = self.lstm(embedding, hidden)
        prediction = self.reLU(self.fc1(output))
        prediction = self.fc2(prediction)
        return prediction, hidden
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell
    
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell
    

def weighted_loss(data,vocab):
 
    pad_index = vocab['[PAD]']
    all_indices = [index for sublist in data for index in sublist if index != pad_index]
    index_counts = Counter(all_indices)
    total_count = sum(index_counts.values())
    default_weight = 1.0 / len(vocab)
    weights_tensor = torch.full((len(vocab),), default_weight)
    
    for index in range(len(vocab)):
        if index == pad_index:
            
            weights_tensor[index] = 0
        elif index in index_counts:
            
            count = index_counts[index]
            weight = 1 - (count / total_count)
            weights_tensor[index] = weight
    
    return weights_tensor

def train_process(train_loader, dev_loader, weights, embedding_dim, hidden_dim=200, epochs=5):
    
    best_dev_loss = float('inf')
    vocab_size = 386
    output_dim = 386
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_dev_loss = float('inf')
    
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights,ignore_index=384)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(epochs):
        print('epoch=',epoch)
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        

        for inp, lab in train_loader:
            hidden = model.init_hidden(inp.size(0), device)
            optimizer.zero_grad()
            inp, lab = inp.to(device), lab.to(device)

            output, hidden = model(inp, hidden) 

            loss = criterion(output.transpose(1, 2), lab)

            loss.backward()
            optimizer.step()

            hidden = model.detach_hidden(hidden)

            epoch_loss += loss.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count
        print(f'Epoch {epoch}: Average Training Loss: {average_loss}')


        model.eval()  
        with torch.no_grad():
            dev_loss = 0.0
            for inp, lab in dev_loader:
                hidden = model.init_hidden(inp.size(0), device)
                output, hidden = model(inp.to(device), hidden)
                dev_loss += criterion(output.transpose(1, 2), lab.to(device)).item()
            dev_loss /= len(dev_loader)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), f'{args.output_dir}/model/best_model.pth')
            print(f'New best model saved at epoch {epoch} with dev loss: {dev_loss}')

    print(f"Training completed with best dev loss: {best_dev_loss}")
    return model

def compute_perplexity(model, test_loader, weights):
    model.eval()  
    total_perplexity = 0.0
    total_perplexity2 = 0.0
    total_sequences = 0
    criterion = nn.CrossEntropyLoss(weight=weights, reduction='none', ignore_index=384)

    with torch.no_grad():  
        for inp, lab in test_loader:
            inp, lab = inp.to(device), lab.to(device)
            
            hidden = model.init_hidden(inp.size(0), device)
            
            output, hidden = model(inp, hidden)
            
            hidden = model.detach_hidden(hidden)
            
            loss = criterion(output.transpose(1, 2), lab).to(device)
            
            for i in range(loss.shape[0]):
                sequence_loss = loss[i].mean()  # Mean loss per token in the sequence
                # sequence_perplexity = 2 ** sequence_loss.item()
                # total_perplexity += sequence_perplexity
                sequence_perplexity = math.exp(sequence_loss.item())
                total_perplexity += sequence_perplexity
                total_sequences += 1

    # average perplexity over all sequences
    average_perplexity = total_perplexity / total_sequences

    return average_perplexity


def load_vocabulary(file_path):
    with open(file_path, 'rb') as f:
        vocabulary = pickle.load(f)
        print("Vocab loaded")
    return vocabulary


def generate_sequence(model, seed_text, char2idx, idx2char, length=200):
    model.eval()  
    device = next(model.parameters()).device  

    hidden, cell = model.init_hidden(1, device)

    for char in seed_text:
        char_idx = torch.tensor([[char2idx[char]]], device=device)  
    
        _, (hidden, cell) = model(char_idx, (hidden, cell))

    input_char = seed_text[-1]
    
    generated_text = seed_text  

   
    for _ in range(length):
        char_idx = torch.tensor([[char2idx[input_char]]], device=device)
       
        output, (hidden, cell) = model(char_idx, (hidden, cell))

        char_scores = output.squeeze().div(0.8).exp() 
        char_probs = torch.multinomial(char_scores, 1).item()

        input_char = idx2char[char_probs]
        
        generated_text += input_char

    return generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1448738/mp3")
    parser.add_argument('--model_dir', type=str, default="models")
    parser.add_argument('--input_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1448738/mp3/")
    parser.add_argument('--vocab_file', type=str, default="vocab.pkl")
    parser.add_argument('--learning_rate', type=float, default='0.00001')
    parser.add_argument('--batch_size', type=int, default='128')
    parser.add_argument('--embedding_dim', type=int, default='50')
    parser.add_argument('--epochs', type=int, default='5')
    parser.add_argument('--k', type=int, default="500")
    parser.add_argument('--hidden_dim', type=int, default="200")
    
    print('Started')
    

    args, _ = parser.parse_known_args()

    train_files = utils.get_files(f'{args.input_dir}/data/train')
    dev_files = utils.get_files(f'{args.input_dir}/data/dev')
    test_files = utils.get_files(f'{args.input_dir}/data/test')

    vocab = load_vocabulary("/uufs/chpc.utah.edu/common/home/u1448738/mp3/data/vocab.pkl")
    
    int2char = {i: char for char, i in vocab.items()}
    
    padding_token = vocab['[PAD]']
    
    train_data = np.array(utils.convert_files2idx(train_files,vocab),dtype=np.ndarray)
    dev_data = np.array(utils.convert_files2idx(dev_files,vocab),dtype=np.ndarray)
    test_data = np.array(utils.convert_files2idx(test_files,vocab),dtype=np.ndarray)
    # print('Files Loaded')
    
    train_loader = data_to_subsequences(train_data,args.k,padding_token)
    dev_loader = data_to_subsequences(dev_data,args.k,padding_token)
    test_loader = data_to_subsequences(test_data,args.k,padding_token)
    # print('Data Loaded')
    
    weights = weighted_loss(train_data,vocab)
    weights = weights.to(device)
    model = train_process(train_loader, dev_loader, weights, args.embedding_dim, 200, 5) 
    
    test_perplexity = compute_perplexity(model, test_loader,weights)
    print(f'Perplexity: {test_perplexity}')
    
    num_param = sum (p.numel() for p in model.parameters())
    print('Num_Param',num_param)
	
    seeds = [
        "The little boy was",
        "Once upon a time in",
        "With the target in",
        "Capitals are big cities. For example,",
        "A cheap alternative to"
    ]

    for seed in seeds:
        generated = generate_sequence(model, seed,vocab,int2char)
        print(f"Seed: {seed}")
        print(f"Generated: {generated}\n")
