import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)

from scripts.utils import get_word2ix,process_data,get_files

from tqdm import tqdm 
import numpy as np
import time
from pickle import dump,load
from sklearn.metrics import f1_score

import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os

from sklearn.metrics.pairwise import cosine_similarity

from gensim.test.utils import datapath
from gensim.models import KeyedVectors

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from datetime import datetime

class CBOW_Model(nn.Module):

    def __init__(self, vocab_size: int):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=args.embedding_dim
        )
        self.linear = nn.Linear(
            in_features=args.embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

# load data and generate labels(target word)
def preprocess_data(dataset):
  
  data = process_data(dataset,args.context_size,word2idx)
  data_list = [item for sublist in data for item in sublist]

  context_window = args.context_size
  batch_input = []
  batch_output = []

  for i in range(context_window,len(data_list)-context_window):
      target_word = data_list[i]
      context = []

      for j in range(i - context_window, i + context_window + 1):
       
          if j != i and j >= 0 and j < len(data_list):
              context.append(data_list[j])

      batch_input.append(context)
      batch_output.append(target_word)


  return TensorDataset(torch.tensor(batch_input),
                       torch.tensor(batch_output)
                      )

def load_data(input_dir):
    #load training dataset
    train_files = get_files(f'{input_dir}/data/train')
    train_dataset = preprocess_data(train_files)

    #load validation dataset
    dev_files = get_files(f'{input_dir}/data/dev')
    dev_dataset = preprocess_data(dev_files)

    train_loader = DataLoader(train_dataset, args.batch_size)
    dev_loader = DataLoader(dev_dataset, args.batch_size)

    return train_loader,dev_loader
    


def run_training(train_data,
                dev_data,
                vocab_size,
                epochs,
                learning_rate):

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    model = CBOW_Model(vocab_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_perf_dict = {"metric": 0, "epoch": 0}
    train_loss = []       
    dev_loss_list = []
    for ep in range(1, args.epochs+1):
        print(f"Epoch {ep}")
        train_loss = []     
        for inp, lab in tqdm(train_data):
            model.train()
            optimizer.zero_grad()
            out = model(inp.to(device)) 
            loss = loss_fn(out, lab.to(device))
            loss.backward() 
            optimizer.step()  
            train_loss.append(loss.cpu().item()) 

        print(f"Average training batch loss: {np.mean(train_loss)}")

       
        gold_labels = []
        pred_labels = []
        for inp, lab in tqdm(dev_data):
          
            model.eval()
            
            with torch.no_grad():
                out = model(inp.to(device))
                preds = torch.argmax(out, dim=1)

                pred_labels.extend(preds.cpu().tolist())
                gold_labels.extend(lab.tolist())
                
                dev_loss = loss_fn(out, lab.to(device))
                dev_loss_list.append(dev_loss.cpu().item())

        dev_f1 = f1_score(gold_labels, pred_labels, average='macro')
        print(f"Dev F1: {dev_f1}\n")
        print(f"Dev Loss: {np.mean(dev_loss_list)}")

       
        if dev_f1 > best_perf_dict["metric"]:
            best_perf_dict["metric"] = dev_f1
            best_perf_dict["epoch"]  = ep

          
            torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_metric": dev_f1,
                "epoch": ep
            }, f"{args.output_dir}{args.model_dir}/model_{ep}")

    print(f"""\nBest Dev performance of {best_perf_dict["metric"]} at epoch {best_perf_dict["epoch"]}""")

        
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    torch.save({
                "model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "epoch": ep
            }, f"{args.output_dir}{args.model_dir}/model_{time}")


    embeddings = list(model.parameters())[1]
    embeddings = embeddings.cpu().detach().numpy()
        
    return embeddings


def save_embeddings(embeddings):

    word_to_embedding = {word: embedding for word, embedding in zip(word2idx, embeddings)}

    output_file = f'{args.output_dir}/embeddings_1.txt'

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"{embeddings.shape[0]} {embeddings.shape[1]}\n")
        for word, embedding in word_to_embedding.items():
            embedding_str = ' '.join(map(str, embedding))
            file.write(f"{word} {embedding_str}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1448738")
    parser.add_argument('--model_dir', type=str, default="models")
    parser.add_argument('--input_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1448738/nlp_proj_1/mp1_release")
    parser.add_argument('--vocab_dir', type=str, default="vocab.txt")
    parser.add_argument('--learning_rate', type=float, default='0.001')
    parser.add_argument('--batch_size', type=int, default='64')
    parser.add_argument('--embedding_dim', type=int, default='100')
    parser.add_argument('--epochs', type=int, default='10')
    parser.add_argument('--context_size', type=int, default="5")
    

    args, _ = parser.parse_known_args()

    word2idx = get_word2ix(f'{args.input_dir}/{args.vocab_dir}')

    idx2word = {index: word for word, index in word2idx.items()}

    train_data,dev_data = load_data(args.input_dir)
    
    vocab_size = len(word2idx)

    embeddings = run_training(train_data,
                        dev_data,
                        vocab_size,
                        args.epochs,
                        args.learning_rate)

    save_embeddings(embeddings)

    