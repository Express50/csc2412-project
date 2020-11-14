# Adapted from: https://github.com/pytorch/opacus/blob/master/examples/imdb.py
import numpy as np
import torch
import torch.nn as nn
import torchtext
from torch import device, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

from model import SentimentAnalysisModel


def binary_accuracy(predictions, label):
    correct = (label.long() == torch.argmax(predictions, dim=1)).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model: SentimentAnalysisModel, train_loader: DataLoader,
          optimizer: Optimizer, epoch: int):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    for batch in tqdm(train_loader):
        data = batch.text.transpose(0, 1)
        label = batch.label

        optimizer.zero_grad()
        predictions = model(data).squeeze(1) # FIXME: Change this to work for transformers (i.e., see forward() method in model)
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accuracies.append(acc.item())

    # TODO: add DP reporting
    print(f'Train epoch: {epoch} \t Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


def evaluate(model: SentimentAnalysisModel, test_loader: DataLoader):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            data = batch.text.transpose(0, 1)
            label = batch.label

            predictions = model(data).squeeze(1)
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)

            losses.append(loss.item())
            accuracies.append(acc.item())

    print(f'Test: Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='roberta-base', nargs='?')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(1234)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load datasets
    text_field = torchtext.data.Field(
        tokenize=get_tokenizer("basic_english"),
        init_token="<sos>",
        eos_token="<eos>",
        fix_length=256,
        lower=True,
    )

    label_field = torchtext.data.LabelField(dtype=torch.long)

    train_data, test_data = torchtext.datasets.imdb.IMDB.splits(
        text_field, label_field, root='../data/'
    )

    print('Building vocab...')
    text_field.build_vocab(train_data, max_size=10000)
    label_field.build_vocab(train_data)

    (train_iterator, test_iterator) = torchtext.data.BucketIterator.splits(
        (train_data, test_data), batch_size=16, device=device
    )

    # init model
    model = SentimentAnalysisModel(args.model, 2)
    optimizer = optim.Adam(model.parameters(), lr=1e-05)

    # TODO: attach DP to optimizer

    print('Training model...')
    for epoch in range(100):
        train(model, train_iterator, optimizer, epoch)
        evaluate(model, test_iterator)
