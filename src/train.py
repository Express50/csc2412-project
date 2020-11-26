# Adapted from: https://github.com/pytorch/opacus/blob/a2419eba1dddd3235d6ddd374e3f9afe4a61a7f5/examples/imdb.py
import os 
os.environ['TRANSFORMERS_CACHE'] = '/w/246/landsmand/csc2412-project/transformers/'

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import device, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer

import torchcsprng as prng
import datasets
from datasets import load_dataset
# opacus

from data import SentimentData
from model import SentimentAnalysisModel

def binary_accuracy(predictions, label):
    correct = (label.long() == torch.argmax(predictions, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def padded_collate(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def train(model: SentimentAnalysisModel, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int, device_: device):
    model = model.train().to(device_)
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for batch in tqdm(train_loader):
        ids = batch['input_ids'].to(device_, dtype = torch.long)
        mask = batch['attention_mask'].to(device_, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
        targets = batch['label'].to(device_, dtype = torch.long)

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
            ids = batch['input_ids'].to(device_, dtype = torch.long)
            mask = batch['attention_mask'].to(device_, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
            targets = batch['label'].to(device_, dtype = torch.long)

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
    raw_dataset = load_dataset("imdb", cache_dir="/w/246/landsmand/csc2412-project/imdb")
    raw_dataset["unsupervised"] = raw_dataset["unsupervised"].select(range(10))
    
    DEBUG_TRAIN_SIZE = 1000
    DEBUG_TEST_SIZE = 500

    if args.debug:
        raw_dataset["train"] = raw_dataset["train"].select(range(DEBUG_TRAIN_SIZE))
        raw_dataset["test"] = raw_dataset["test"].select(range(DEBUG_TEST_SIZE))

    print('Running tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], 
            truncation=True, 
            max_length=256, 
            return_token_type_ids=True, 
            add_special_tokens=True,
            pad_to_max_length=True
        ),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    generator = (None)

    batch_size = 16
    workers = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    # init model
    model = SentimentAnalysisModel(args.model, 2).to(device_)
    optimizer = optim.Adam(model.parameters(), lr=1e-05)

    # TODO: attach DP to optimizer

    try:
        print('Training model...')
        for epoch in range(10):
                train(model, train_loader, optimizer, epoch, device_)
                evaluate(model, test_loader, device_)
    except RuntimeError as e:
        print(e)
