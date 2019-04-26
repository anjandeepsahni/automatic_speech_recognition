import os
import csv
import copy
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import Levenshtein as Lev
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import *
from model import SpeechRecognizer

# Paths
MODEL_PATH = './../Models'
TEST_RESULT_PATH = './../Results'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_FEATURE_SIZE = 40
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TEST_BATCH_SIZE = 32
DEFAULT_LABEL_MAP = VOCAB
DEFAULT_RANDOM_SEED = 2222

# Hyperparameters.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1.2e-6
GRADIENT_CLIP = 0.25

def parse_args():
    parser = argparse.ArgumentParser(description='Training/testing for Speech Recognition.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=DEFAULT_RUN_MODE, help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int, default=DEFAULT_TRAIN_BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int, default=DEFAULT_TEST_BATCH_SIZE, help='Testing batch size.')
    parser.add_argument('--model_path', type=str, help='Path to model to be reloaded.')
    return parser.parse_args()

def map_label_string(label):
    return DEFAULT_LABEL_MAP[label]

def generate_labels_string(batch_pred):
    # Loop over entire batch list of predicted labels and convert them to strings.
    batch_strings = []
    for pred in batch_pred:
        batch_strings.append(''.join(list(map(map_label_string, list(pred.numpy())))))
    return batch_strings

def character_error_rate(pred, targets):
    assert len(pred) == len(targets)
    #FIXME: Do I need to remove space before calculating edit distance?
    dist = []
    for idx, p in enumerate(pred):
        dist.append(Lev.distance(p, targets[idx]))
    return dist

def save_test_results(predictions, ensemble=False):
    predictions_count = list(range(len(predictions)))
    csv_output = [[i,j] for i,j in zip(predictions_count,predictions)]
    if not ensemble:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_{}.csv'.format((str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    else:
        result_file_path = os.path.join(TEST_RESULT_PATH,\
                'result_ensemble_{}.csv'.format(time.strftime("%Y%m%d-%H%M%S")))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Id', 'Predicted'])
        csv_writer.writerows(csv_output)

def test_model(model, test_loader, decoder, device):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        all_predictions = []
        for batch_idx, (inputs, inp_lens, _, _, seq_order) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs, inp_lens)
            # pad_out: Batch, Beam_Size, Max_Seq_L -> 'Beam_Size' predictions for each item in batch.
            # out_lens: Batch, Beam_Size -> Actual length of each prediction for each item in batch.
            # (Batch, Max_Seq_L, Dict) for CTC Loss.
            outputs = torch.transpose(outputs, 0, 1)
            pad_out, _, _, out_lens = decoder.decode(F.softmax(outputs, dim=2).data.cpu(), torch.IntTensor(inp_lens))
            # Iterate over each item in batch.
            y_pred = []
            for i in range(len(out_lens)):
                y_pred.append(pad_out[i, 0, :out_lens[i, 0]])   # Pick the first output, most likely.
            # Calculate the strings for predictions.
            pred_str = generate_phoneme_string(y_pred)
            # Input is sorted as per length for rnn. Resort the output.
            reorder_seq = np.argsort(seq_order)
            pred_str = [pred_str[i] for i in reorder_seq]
            all_predictions.extend(pred_str)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)), end="\r", flush=True)
        end_time = time.time()
        # Save predictions in csv file.
        save_test_results(all_predictions)
        print('\nTotal Test Predictions: %d Time: %d s' % (len(all_predictions), end_time - start_time))

def greedy_decode(probs):
    eos_idx = DEFAULT_LABEL_MAP.index('<eos>')
    preds = torch.argmax(probs, dim=2)
    # Iterate over each item in batch.
    pred_list = []
    for i in  range(preds.size(0)):
        eos_idx = (preds[i]==eos_idx).nonzero()
        eos_idx = (len(preds[i])-1) if eos_idx.nelement() == 0 else eos_idx[0]
        pred_list.append(preds[i,:eos_idx])
    return pred_list

def val_model(model, val_loader, criterion, device):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        dist = []
        start_time = time.time()
        for batch_idx, (inputs, inp_lens, targets, tar_lens, targets_loss, _) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets, targets_loss = targets.to(device), targets_loss.to(device)
            outputs, attentions = model(inputs, inp_lens, targets)
            loss = criterion(outputs.view(-1,outputs.size(2)), targets_loss.view(-1))
            running_loss += loss.item()
            pred_list = greedy_decode(F.softmax(outputs, dim=2))
            tar_list = [targets[i,:(tar_lens[i]-1)] for i in range(targets.size(0))]   #exclude eos
            # Calculate the strings for predictions.
            pred_str = generate_labels_string(pred_list)
            # Calculate the strings for targets.
            tar_str = generate_labels_string(tar_list)
            # Calculate edit distance between predictions and targets.
            dist.extend(character_error_rate(pred_str, tar_str))
            print('Validation Iteration: %d/%d Loss = %5.4f' % \
                    (batch_idx+1, len(val_loader), (running_loss/(batch_idx+1))), \
                    end="\r", flush=True)
        end_time = time.time()
        dist = sum(dist)/len(dist)    # Average over edit distance.
        running_loss /= len(val_loader)
        print('\nValidation Loss: %5.4f Validation Levenshtein Distance: %5.3f Time: %d s' % \
                (running_loss, dist, end_time - start_time))
        return running_loss, dist

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for batch_idx, (inputs, inp_lens, targets, _, targets_loss, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets, targets_loss = targets.to(device), targets_loss.to(device)
        optimizer.zero_grad()
        outputs, attentions = model(inputs, inp_lens, targets)
        loss = criterion(outputs.view(-1,outputs.size(2)), targets_loss.view(-1))
        loss.backward()
        running_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)    # To avoid exploding gradient issue.
        optimizer.step()
        print('Train Iteration: %d/%d Loss = %5.4f' % \
                (batch_idx+1, len(train_loader), (running_loss/(batch_idx+1))), \
                end="\r", flush=True)
    end_time = time.time()
    running_loss /= len(train_loader)
    print('\nTraining Loss: %5.4f Training Levenshtein Distance: %5.4f Time: %d s'
            % (running_loss, -1, end_time - start_time))
    return running_loss

if __name__ == "__main__":
    # Parse args.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and args.model_path == None and not args.model_ensemble:
        raise ValueError("Input Argument Error: Test mode specified but model_path is %s." % (args.model_path))

    # Check for CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders.
    speechTrainDataset = SpeechDataset(mode='train')
    speechValDataset = SpeechDataset(mode='dev')
    speechTestDataset = SpeechDataset(mode='test')

    train_loader = DataLoader(speechTrainDataset, batch_size=args.train_batch_size,
                                shuffle=True, num_workers=4, collate_fn=SpeechCollateFn)
    val_loader = DataLoader(speechValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)
    test_loader = DataLoader(speechTestDataset, batch_size=args.test_batch_size,
                        shuffle=False, num_workers=4, collate_fn=SpeechCollateFn)

    # Set random seed.
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed(DEFAULT_RANDOM_SEED)

    # Create the model.
    model = SpeechRecognizer()
    model.to(device)
    print('='*20)
    print(model)
    model_params = sum(p.size()[0] * p.size()[1] if len(p.size()) > 1 else p.size()[0] for p in model.parameters())
    print('Total model parameters:', model_params)
    print("Running on device = %s." % (device))

    # Setup learning parameters.
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_ID)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, threshold=0.01, verbose=True)

    if args.model_path != None:
        model.load_state_dict(torch.load(args.model_path))
        print('Loaded model:', args.model_path)

    n_epochs = 50
    print('='*20)

    if args.mode == 'train':
        for epoch in range(n_epochs):
            print('Epoch: %d/%d' % (epoch+1,n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = val_model(model, val_loader, criterion, device)
            # Checkpoint the model after each epoch.
            finalValAcc = '%.3f'%(val_acc)
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), finalValAcc))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            scheduler.step(val_loss)
    else:
        # Only testing the model.
        test_model(model, test_loader, device)
    print('='*20)
