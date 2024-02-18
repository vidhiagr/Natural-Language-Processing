
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from transformers import BertTokenizer,BertModel

from torch.utils.data import TensorDataset, DataLoader

import torch.nn.functional as F

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

"""Create Dataset"""

class CreateDataset():
  def __init__(self , dataset, tokenizer,task) -> None:
    self.dataset = dataset
    self.tokenizer = tokenizer
    if(task == 'rte'):
      self.tokenized_input_train = tokenizer(dataset['train']['text1'],dataset['train']['text2'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
      self.tokenized_input_dev = tokenizer(dataset['validation']['text1'],dataset['validation']['text2'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
      self.tokenized_input_test = tokenizer(dataset['test']['text1'],dataset['test']['text2'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')

    elif(task == 'sst'):
      self.tokenized_input_train = tokenizer(dataset['train']['text'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
      self.tokenized_input_dev = tokenizer(dataset['validation']['text'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
      self.tokenized_input_test = tokenizer(dataset['test']['text'],max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
    self.train_labels = torch.Tensor(dataset['train']['label'])
    self.dev_labels = torch.Tensor(dataset['validation']['label'])
    self.test_labels = torch.Tensor(dataset['test']['label'])


  def create_dataloader(self, split, batch_size=64, shuffle=True):
        if split == 'train':
            inputs = self.tokenized_input_train
            labels = self.train_labels
        elif split == 'dev':
            inputs = self.tokenized_input_dev
            labels = self.dev_labels
        elif split == 'test':
            inputs = self.tokenized_input_test
            labels = self.test_labels
        else:
            raise ValueError("Invalid split name. Use 'train' or 'dev'.")

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader

def get_dataloader(dataset,tokenizer,task):
  dataset = CreateDataset(dataset, tokenizer,task)
  train_dataloader = dataset.create_dataloader(split='train', batch_size=64, shuffle=True)
  dev_dataloader = dataset.create_dataloader(split='dev', batch_size=64, shuffle=False)
  test_dataloader = dataset.create_dataloader(split='test', batch_size=64, shuffle=False)
  return train_dataloader,dev_dataloader,test_dataloader

"""Define model class"""

class Classifier(torch.nn.Module):
  def __init__(self,in_dim):
    super().__init__()
    self.w = torch.nn.Linear(in_dim, 2, bias=True)

  def forward(self,input):
    k = self.w(input)
    return k

class ClassifierFineTuned(torch.nn.Module):
  def __init__(self,bert_dim,model_checkpoint):
    super().__init__()
    self.model = BertModel.from_pretrained(model_checkpoint)
    self.w = torch.nn.Linear(bert_dim, 2, bias=True)

  def forward(self,input,attention_mask):
    out = self.model(input, attention_mask=attention_mask)
    k = self.w(out['pooler_output'])
    return k



"""Train without fine-tuning"""
def train_without_finetune(train_dataloader,dev_dataloader,clf,optimizer,bert,model_path):
  criterion = torch.nn.CrossEntropyLoss()
  max_epochs=10
  best_acc = 0
  for ep in range(1, max_epochs + 1):
    print(f"Epoch {ep}")
    train_loss = []
    correct_train_predictions = 0
    total_train_samples = 0
    # Training Loop
    for inp, am, lab in tqdm(train_dataloader):
      clf.train()
      optimizer.zero_grad()
      bert_out = bert(inp,am)
      bert_out = bert_out['pooler_output']
      out = clf(bert_out)
      loss = criterion(out, lab.type(torch.long))
      loss.backward()
      optimizer.step()
      train_loss.append(loss.cpu().item())

  # Calculate accuracy
      preds = torch.argmax(out, dim=1)
      correct_train_predictions += (preds == lab).sum().item()
      total_train_samples += lab.size(0)
    train_accuracy = correct_train_predictions / total_train_samples
    print(f"Average training batch loss: {np.mean(train_loss)} | Training accuracy: {train_accuracy * 100:.2f}%")
    # Evaluation Loop
    gold_labels = []
    pred_labels = []
    dev_loss = []
    correct_dev_predictions = 0
    total_dev_samples = 0
    for inp, am, lab in tqdm(dev_dataloader):
      clf.eval ()
      with torch.no_grad():
        bert_out = bert(inp, am)
        bert_out = bert_out['pooler_output']
        out = clf(bert_out)
        preds = torch.argmax(out, dim=1)
        loss = criterion(out,lab.type(torch. long))
        pred_labels.extend(preds.cpu().tolist())
        gold_labels.extend(lab.tolist())
        dev_loss.append(loss.cpu().item())

        # Calculate accuracy
        correct_dev_predictions += (preds == lab).sum().item()
        total_dev_samples += lab.size(0)
    dev_accuracy = correct_dev_predictions / total_dev_samples
    dev_loss_value = np.mean(dev_loss)
    
    if dev_accuracy > best_acc:
      best_acc = dev_accuracy
      torch.save(clf.state_dict(), model_path)
    print(f"Average dev batch loss: {dev_loss_value} | Development accuracy: {dev_accuracy * 100:.2f}%")
  return clf,bert

"""Test Accuracy"""
def test_accuracy_without_finetune(test_dataloader,clf,bert,model_path):
  total=0
  correct=0
  clf.load_state_dict(torch.load(model_path))
  with torch.no_grad():
      for inp,am,lab in test_dataloader:
          bert_out = bert(inp, am)
          bert_out = bert_out['pooler_output'] 
          out = clf(bert_out)
          preds = torch.argmax(out, dim=1)
          
          total += lab.size(0)
          correct += (preds == lab).sum().item()

    # Calculate accuracy
  accuracy = correct / total
  print(f'Test Accuracy: {accuracy * 100:.2f}%')



"""Train with finetuning"""

def train_fine_tuned(train_dataloader,dev_dataloader,clf_finetune, optimizer_fine_tune, model_path):
  criterion = torch.nn.CrossEntropyLoss()
  max_epochs = 10
  best_acc = 0
  for ep in range(1, max_epochs + 1):
    print(f"Epoch {ep}")
    train_loss = []
    correct_train_predictions = 0
    total_train_samples = 0
    # Training Loop
    for inp, am, lab in tqdm(train_dataloader):
      clf_finetune.train()
      optimizer_fine_tune.zero_grad()
      out = clf_finetune(inp, am)
      loss = criterion(out,lab.type(torch.long))
      loss.backward()
      optimizer_fine_tune.step()
      train_loss.append(loss.cpu().item())

  # Calculate accuracy
      preds = torch.argmax(out, dim=1)
      correct_train_predictions += (preds == lab).sum().item()
      total_train_samples += lab.size(0)
    train_accuracy = correct_train_predictions / total_train_samples
    print(f"Average training batch loss: {np.mean(train_loss)} | Training accuracy: {train_accuracy * 100:.2f}%")
    # Evaluation Loop
    gold_labels = []
    pred_labels = []
    dev_loss = []
    correct_dev_predictions = 0
    total_dev_samples = 0
    for inp_dev, am_dev, lab_dev in tqdm(dev_dataloader):
      clf_finetune.eval ()
      with torch.no_grad():
        out = clf_finetune(inp_dev, am_dev)
        preds = torch.argmax(out, dim=1)
        loss = criterion(out,lab_dev.type(torch.long))
        pred_labels.extend(preds.cpu().tolist())
        gold_labels.extend(lab_dev.tolist())
        dev_loss.append(loss.cpu().item())

        # Calculate accuracy
        correct_dev_predictions += (preds == lab_dev).sum().item()
        total_dev_samples += lab_dev.size(0)
    dev_accuracy = correct_dev_predictions / total_dev_samples
    dev_loss_value = np.mean(dev_loss)
    if dev_accuracy > best_acc:
      best_acc = dev_accuracy
      torch.save(clf_finetune.state_dict(), model_path)
    print(f"Average dev batch loss: {dev_loss_value} | Development accuracy: {dev_accuracy * 100:.2f}%")
    return clf_finetune

"""Test Accuracy"""
def test_accuracy_finetune(test_dataloader,clf,model_path):
  clf.load_state_dict(torch.load(model_path))
  total=0
  correct=0
  with torch.no_grad():
      for inp,am,lab in test_dataloader:
          out = clf(inp, am)
          preds = torch.argmax(out, dim=1)

          total += lab.size(0)
          correct += (preds == lab).sum().item()

  # Calculate accuracy
  accuracy = correct / total
  print(f'Test Accuracy: {accuracy * 100:.2f}%')
  
"""Random Classifier"""
def extract_features_and_labels(dataset_split,task):
    if task == 'rte':
      features = [example['text1'] + " " + example['text2'] for example in dataset_split]
    elif task == 'sst':
      features = [example['text'] for example in dataset_split]
    labels = [example['label'] for example in dataset_split]
    return features, labels


def random_classifier(dataset,task):
  train_features, train_labels = extract_features_and_labels(dataset['train'],task)
  test_features, test_labels = extract_features_and_labels(dataset['test'],task)

  dummy_clf = DummyClassifier(strategy='uniform')
  dummy_clf.fit(train_features, train_labels)
  y_pred = dummy_clf.predict(test_features)
  random_accuracy = accuracy_score(test_labels, y_pred)

  print(f"Random Baseline Accuracy on RTE Dataset: {random_accuracy * 100:.2f}%")
  
  
"""Sentence prediction for SST2"""

def predict_sentences_prob_rte(input_ids, attention_mask, model, model_path, pairs):
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask)
        probs_hidden = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs_hidden, dim=1)

    # Print the predictions
    for i, (pair, prediction, probabilities) in enumerate(zip(pairs, predicted_class, probs_hidden)):
        print(f"Pair {i + 1} - Predicted Class: {prediction.item()} | Probabilities: {probabilities.tolist()}")

def sentence_prediction_rte(tokenizer, model, model_path):
  pairs = [
        ("The doctor is prescribing medicine.", "She is prescribing medicine."),
        ("The doctor is prescribing medicine.", "He is prescribing medicine."),
        ("The nurse is tending to the patient.", "She is tending to the patient."),
        ("The nurse is tending to the patient.", "He is tending to the patient.")
        ]

    # # Tokenize and convert to model input format
  inputs = tokenizer(
  [pair[0] for pair in pairs],  # List of premise sentences
  [pair[1] for pair in pairs],  # List of hypothesis sentences
  max_length=512,
  truncation=True,
  padding=True,
  return_tensors='pt'
  ) 
  
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']

  predict_sentences_prob_rte(input_ids, attention_mask, model, model_path, pairs)
  
  
"""Sentence prediction for SST2"""
def predict_sentences_prob_sst(input_ids, attention_mask, model, model_path, sentences):
    
    model.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        model.eval()
        outputs = model(input_ids, attention_mask)
        probs_hidden = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs_hidden, dim=1)

    # Print the predictions
    for sentence, prediction, probabilities in zip(sentences, predicted_class, probs_hidden):
        print(f"Sentence: {sentence} | Predicted Class: {prediction.item()} | Probabilities: {probabilities.tolist()}")


def sentence_prediction_sst(tokenizer, model, model_path):
  sentences = [
        "Kate should get promoted, she is an amazing employee.",
        "Bob should get promoted, he is an amazing employee.",
        "Kate should get promoted, he is an amazing employee.",
        "Bob should get promoted, they are an amazing employee."
    ]

    # Tokenize and convert to model input format
  inputs = tokenizer(sentences,max_length= 512,truncation = True, padding = True, return_tensors = 'pt')

  # Assuming 'inputs' contains 'input_ids', 'attention_mask', and other necessary fields
  # For example, if you have 'input_ids' and 'attention_mask' as model inputs:
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']
  predict_sentences_prob_sst(input_ids,attention_mask ,model, model_path, sentences)
  
  
"""Hidden Predictions"""
def get_dataloader_hidden(hidden_df,model_checkpoint,task):
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    if task == 'rte':
        tokenized_hidden = tokenizer(hidden_df['text1'].tolist(),hidden_df['text2'].tolist(), max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
    elif task == 'sst':
        tokenized_hidden = tokenizer(hidden_df['text'].tolist(), max_length= 512,truncation = True, padding = True, return_tensors = 'pt')
    dataset_hidden = TensorDataset(
        tokenized_hidden['input_ids'],
        tokenized_hidden['attention_mask']
    )

    # Create DataLoader
    hidden_dataloader = DataLoader(dataset_hidden, batch_size=1, shuffle=True)
    
    return hidden_dataloader
  
def get_predictions_and_probabilities(model, hidden_df, dataloader,model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    predictions = []
    probabilities = []

    with torch.no_grad():
        for inp, am in dataloader:  # Assuming label is not needed for hidden dataset
            out = model(inp, am)
            probs = F.softmax(out, dim=1)  # Convert logits to probabilities

            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    hidden_df['prediction'] = predictions
    hidden_df['probab 0'] = [prob[0] for prob in probabilities]
    hidden_df['probab 1'] = [prob[1] for prob in probabilities]
    print(hidden_df.prediction.value_counts())
    # Save to a CSV file
    
    return hidden_df
  
def hidden_prediction_rte(model_checkpoint, model, model_path):
  hidden_dataset_rte = pd.read_csv('hidden_data/hidden_rte.csv')
  hidden_df_rte = pd.DataFrame(hidden_dataset_rte)
  hidden_datalaoder_rte = get_dataloader_hidden(hidden_df_rte, model_checkpoint, 'rte')
  hidden_df = get_predictions_and_probabilities(model,hidden_df_rte,hidden_datalaoder_rte,model_path)
  hidden_df.to_csv("results_rte.csv", index=False)
  

def hidden_prediction_sst(model_checkpoint, model, model_path):
  hidden_dataset_sst = pd.read_csv('hidden_data/hidden_sst2.csv')
  hidden_df_sst = pd.DataFrame(hidden_dataset_sst)
  hidden_datalaoder_sst = get_dataloader_hidden(hidden_df_sst, model_checkpoint, 'sst')
  hidden_df = get_predictions_and_probabilities(model,hidden_df_sst,hidden_datalaoder_sst,model_path)
  hidden_df.to_csv("results_sst.csv", index=False)
  

"""MAIN"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1448738/mp4")
    parser.add_argument('--model_dir', type=str, default="models")
    parser.add_argument('--learning_rate', type=float, default='0.0001')
    parser.add_argument('--task', type=str, default='rte')
    parser.add_argument('--model_checkpoint', type=str, default='prajjwal1/bert-tiny')
    parser.add_argument('--finetune', type=bool, default="True")
    parser.add_argument('--dataset', type=str, default="yangwang825/rte")

    args, _ = parser.parse_known_args()

    """Run Tasks"""
    def run(dataset_path,model_checkpoint,bert_dim,finetune,task,model_dir):
      dataset = load_dataset(dataset_path)
      tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
      train_dataloader,dev_dataloader,test_dataloader = get_dataloader(dataset,tokenizer,task)
      
      if not finetune:   # without finetune
        clf = Classifier(bert_dim)
        bert = BertModel.from_pretrained(model_checkpoint)
        optimizer = torch.optim.Adam(clf.parameters(), args.learning_rate)
        model_path = f'{args.output_dir}/{model_dir}/model_without_finetune_2.pth'
        print(model_path)
        model, bert = train_without_finetune(train_dataloader,dev_dataloader,clf,optimizer,bert,model_path)
        test_accuracy_without_finetune(test_dataloader,model,bert,model_path)
      
      else:  # finetune
        clf = ClassifierFineTuned(bert_dim,model_checkpoint)
        optimizer = torch.optim.Adam(clf.parameters(),args.learning_rate)
        model_path = f'{args.output_dir}/{model_dir}/model_with_finetune_2.pth'
        print(model_path)
        model = train_fine_tuned(train_dataloader,dev_dataloader,clf, optimizer, model_path)
        test_accuracy_finetune(test_dataloader,model,model_path)
      
        # As scores after finetuning were better, so adding sentence and hidden prediction for finetuned model
        if task == 'rte' :
          sentence_prediction_rte(tokenizer, model, model_path)
          hidden_prediction_rte(model_checkpoint, model, model_path)
          
        elif task == 'sst':
          sentence_prediction_sst(tokenizer, model, model_path)
          hidden_prediction_sst(model_checkpoint, model, model_path)
        
    """Call all tasks"""
    print('RTE Tiny- With Fine tuning')
    run("yangwang825/rte", 'prajjwal1/bert-tiny', 128, True, 'rte', 'rte_tiny')
    
    print('RTE Tiny- Without Fine tuning')
    run("yangwang825/rte", 'prajjwal1/bert-tiny', 128, False, 'rte', 'rte_tiny')
    
    print('RTE Mini- With Fine tuning')
    run("yangwang825/rte", "prajjwal1/bert-mini", 256, True, 'rte', 'rte_mini')
    
    print('RTE Mini- Without Fine tuning')
    run("yangwang825/rte", "prajjwal1/bert-mini", 256, False, 'rte', 'rte_mini')
    
    print('SST Tiny- With Fine tuning')
    run("gpt3mix/sst2", 'prajjwal1/bert-tiny', 128, True, 'sst', 'sst_tiny')
    
    print('SST Tiny- Without Fine tuning')
    run("gpt3mix/sst2", 'prajjwal1/bert-tiny', 128, False, 'sst', 'sst_tiny')
    
    print('SST Mini- With Fine tuning')
    run("gpt3mix/sst2", "prajjwal1/bert-mini", 256, True, 'sst', 'sst_mini')
    
    print('SST Mini- Without Fine tuning')
    run("gpt3mix/sst2", "prajjwal1/bert-mini", 256, False, 'sst', 'sst_mini')
    
    """Random Classifier"""
    print('For RTE')
    dataset_rte = load_dataset("yangwang825/rte")
    random_classifier(dataset_rte,'rte')
    
    print('For SST')
    dataset_sst = load_dataset("gpt3mix/sst2")
    random_classifier(dataset_sst,'sst')
    
    
    
    
    
    
    
        
      
    

    
    
    
    