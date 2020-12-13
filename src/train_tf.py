# Adapted from: https://github.com/pytorch/opacus/blob/a2419eba1dddd3235d6ddd374e3f9afe4a61a7f5/examples/imdb.py
import os 
os.environ['TRANSFORMERS_CACHE'] = '/w/246/landsmand/csc2412-project/transformers/'

import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as optim
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy
# import torch
# import torch.nn as nn
# from torch import device, optim
# from torch.optim.optimizer import Optimizer
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
from tensorflow import argmax
from tensorflow.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from opacus import PrivacyEngine

from model_tf import SentimentAnalysisModel

# flag to stop training when we hit epsilon threshold
eps_threshold_hit = False

BATCH_SIZE = 4
VIRTUAL_BATCH_SIZE = 16

def binary_accuracy(predictions, label):
    correct = (label.long() == argmax(predictions, axis=1)).float()
    acc = correct.sum() / len(correct)
    return acc


# def padded_collate(batch, padding_idx=0):
#     x = pad_sequence(
#         [elem["input_ids"] for elem in batch],
#         batch_first=True,
#         padding_value=padding_idx,
#     )
#     y = torch.stack([elem["label"] for elem in batch]).long()
#     return x, y


# def train(args, model: SentimentAnalysisModel, train_loader: DataLoader,
#           optimizer: Optimizer, epoch: int, device_: device):
#     global eps_threshold_hit
    
#     model = model.train().to(device_)
#     criterion = nn.CrossEntropyLoss()
#     losses = []
#     accuracies = []

#     virtual_batch_rate = VIRTUAL_BATCH_SIZE / BATCH_SIZE

#     for idx, batch in enumerate(tqdm(train_loader)):
#         ids = batch['input_ids'].to(device_, dtype = torch.long)
#         mask = batch['attention_mask'].to(device_, dtype = torch.long)
#         token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
#         targets = batch['label'].to(device_, dtype = torch.long)

#         optimizer.zero_grad()
#         predictions = model(ids, mask, token_type_ids)
#         loss = criterion(predictions, targets)
#         acc = binary_accuracy(predictions, targets)

#         loss.backward()

#         if args.eps_threshold is not None:
#             # do virtual stepping to improve performance
#             if (idx + 1) % virtual_batch_rate == 0 or idx == len(train_loader) - 1:
#                 optimizer.step()
#                 optimizer.zero_grad()
#             else:
#                 optimizer.virtual_step()
#         else:
#             optimizer.step()

#         losses.append(loss.item())
#         accuracies.append(acc.item())

#     if args.eps_threshold is not None:
#         epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()
#         print(
#             f"Train Epoch: {epoch} \t"
#             f"Train Loss: {np.mean(losses):.6f} "
#             f"Train Accuracy: {np.mean(accuracies):.6f} "
#             f"(ε = {epsilon:.2f}, δ = {1e-06}) for α = {best_alpha}"
#         )

#         # stop training if eps >= eps_threshold
#         eps_threshold_hit = epsilon >= args.eps_threshold

#         if eps_threshold_hit:
#             print('Hit epsilon threshold, stopping training.')

#     else:
#         print(f'Train epoch: {epoch} \t Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


# def evaluate(model: SentimentAnalysisModel, test_loader: DataLoader, device_: device):
#     model = model.eval().to(device_)
#     criterion = nn.CrossEntropyLoss()
#     losses = []
#     accuracies = []

#     with torch.no_grad():
#         for batch in tqdm(test_loader):
#             ids = batch['input_ids'].to(device_, dtype = torch.long)
#             mask = batch['attention_mask'].to(device_, dtype = torch.long)
#             token_type_ids = batch['token_type_ids'].to(device_, dtype = torch.long)
#             targets = batch['label'].to(device_, dtype = torch.long)

#             predictions = model(ids, mask, token_type_ids)
#             loss = criterion(predictions, targets)
#             acc = binary_accuracy(predictions, targets)

#             losses.append(loss.item())
#             accuracies.append(acc.item())

#     print(f'Test: Avg Loss: {np.mean(losses)} \t Avg Accuracy: {np.mean(accuracies)}')


if __name__ == "__main__":
    import gc
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('model', default='roberta-base', nargs='?')
    parser.add_argument('eps_threshold', default=None, type=float, nargs='?')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    # torch.manual_seed(1234)
    tf.random.set_seed(1234)
    
    # print(f'cuda available: {torch.cuda.is_available()}')
    # device_ = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load datasets
    print('Loading data...')
    raw_dataset = load_dataset("imdb")#, cache_dir="/w/246/landsmand/csc2412-project/imdb")
    raw_dataset["unsupervised"] = raw_dataset["unsupervised"].select(range(10))
    
    epochs = 10

    DEBUG_TRAIN_SIZE = 10
    DEBUG_TEST_SIZE = 10
    DEBUG_EPOCHS = 3

    if args.debug:
        raw_dataset["train"] = raw_dataset["train"].select(range(DEBUG_TRAIN_SIZE))
        raw_dataset["test"] = raw_dataset["test"].select(range(DEBUG_TEST_SIZE))
        epochs = DEBUG_EPOCHS

    print('Running tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = raw_dataset.map(
        lambda x: tokenizer(
            x["text"], 
            truncation=True, 
            max_length=64, 
            return_token_type_ids=True, 
            add_special_tokens=True,
            pad_to_max_length= (args.model != 'distilgpt2')
        ),
        batched=True,
    )
    dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "token_type_ids", "label"])

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    train_features = {x: train_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.max_len]) for x in ['input_ids', 'token_type_ids', 'attention_mask']}
    test_features = {x: test_dataset[x].to_tensor(default_value=0, shape=[None, tokenizer.max_len]) for x in ['input_ids', 'token_type_ids', 'attention_mask']}


    generator = (None)

    
    
    batch_size = 32
    if args.eps_threshold is not None:
        batch_size = BATCH_SIZE

    workers = 2

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=workers,
    #     pin_memory=True,
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=workers,
    #     pin_memory=True,
    # )

    tfdataset_train = Dataset.from_tensor_slices((train_features, train_dataset["label"])).batch(batch_size)
    tfdataset_test = Dataset.from_tensor_slices((train_features, train_dataset["label"])).batch(batch_size)

    # init model
    model = SentimentAnalysisModel(args.model, 2) #.to(device_)
    optimizer = optim.Adam(learning_rate=1e-05)

    model.compile(
        optimizer=optimizer, 
        loss=BinaryCrossentropy(),
        metrics=[BinaryAccuracy()]
    )

    # attach DP to optimizer
    # sigma = 0.5
    # max_grad_norm = 1.0
    # if args.eps_threshold is not None:
    #     print('Attaching privacy engine...')
    #     privacy_engine = PrivacyEngine(
    #         model,
    #         batch_size = VIRTUAL_BATCH_SIZE,
    #         sample_size = len(train_dataset),
    #         alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
    #         noise_multiplier = sigma,
    #         max_grad_norm = max_grad_norm
    #     )
    #     privacy_engine.attach(optimizer)

    try:
        print('Training model...')
        # for epoch in range(epochs):
        #     # if not eps_threshold_hit:
        #     train(args, model, tfdataset_train, optimizer, epoch, device_)
        #     evaluate(model, tfdataset_test, device_)

        #     # if eps_threshold_hit: break
        model.fit(
            tfdataset_train,
            epochs=epochs,
            validation_data=tfdataset_test
        )

    except RuntimeError as e:
        print(e)
