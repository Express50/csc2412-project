# Adapted from: https://github.com/pytorch/opacus/blob/aa31b7399f704a897b9476852e95cbeaf14069be/examples/imdb.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import device, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from data import SentimentData
from model import SentimentAnalysisModel


def binary_accuracy(predictions, label):
    correct = (label.long() == torch.argmax(predictions, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model: SentimentAnalysisModel, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int, device_: device):
    model = model.train().to(device_)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for batch in tqdm(train_loader):
        ids = batch['ids'].to(device_, dtype = torch.long)
        mask = batch['mask'].to(device_, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
        targets = batch['targets'].to(device_, dtype = torch.long)

        optimizer.zero_grad()
        predictions = model(ids, mask, token_type_ids)
        loss = criterion(predictions, targets)
        acc = binary_accuracy(predictions, targets)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    # TODO: add DP reporting
    print(f'Train epoch: {epoch} \t Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


def evaluate(model: SentimentAnalysisModel, test_loader: DataLoader, device_: device):
    model = model.eval().to(device_)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            ids = batch['ids'].to(device_, dtype = torch.long)
            mask = batch['mask'].to(device_, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
            targets = batch['targets'].to(device_, dtype = torch.long)

            predictions = model(ids, mask, token_type_ids)
            loss = criterion(predictions, targets)
            acc = binary_accuracy(predictions, targets)

            losses.append(loss.item())
            accuracies.append(acc.item())

    print(f'Test: Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


if __name__ == "__main__":
    import gc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='roberta-base', nargs='?')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(1234)
    
    print(f'cuda available: {torch.cuda.is_available()}')
    device_ = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load datasets
    print('Loading data...')
    sentiment_data = pd.read_csv('../data/rotten/train.tsv', delimiter='\t')[['Phrase', 'Sentiment']]

    if args.debug:
        sentiment_data = sentiment_data.sample(n=5000)

    train_data = sentiment_data.sample(frac=0.8)
    test_data = sentiment_data.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_set = SentimentData(train_data, tokenizer, 256)
    test_set = SentimentData(test_data, tokenizer, 256)

    train_iterator = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True
    )

    test_iterator = DataLoader(
        test_set,
        batch_size=16,
        shuffle=True
    )

    # init model
    model = SentimentAnalysisModel(args.model, 5).to(device_)
    optimizer = optim.Adam(model.parameters(), lr=1e-05)

    # TODO: attach DP to optimizer

    try:
        print('Training model...')
        for epoch in range(10):
                train(model, train_iterator, optimizer, epoch, device_)
                evaluate(model, test_iterator, device_)
    except RuntimeError as e:
        print(e)
