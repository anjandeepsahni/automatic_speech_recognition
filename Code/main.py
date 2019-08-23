import argparse
import csv
import os
import time

import Levenshtein as Lev
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from beamsearch import beam_search
from dataset import *   # noqa F403
from model import Seq2Seq
from scoreboard import Scoreboard

# Paths
MODEL_PATH = './Models'
TEST_RESULT_PATH = './Results'
GRAD_FIGURES_PATH = './Grad_Figures'

# Defaults
DEFAULT_RUN_MODE = 'train'
DEFAULT_FEATURE_SIZE = 40
DEFAULT_TRAIN_BATCH_SIZE = 32
DEFAULT_TEST_BATCH_SIZE = 32
DEFAULT_RANDOM_SEED = 2222
SCOREBOARD_KEY = ["CER", "Mine", "Label"]

# Hyperparameters.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1.2e-6
GRADIENT_CLIP = 0  # 0.25


def parse_args():
    parser = argparse.ArgumentParser(
        description='Training/testing for Speech Recognition.')
    parser.add_argument('--mode', type=str, choices=['train', 'test'],
                        default=DEFAULT_RUN_MODE,
                        help='\'train\' or \'test\' mode.')
    parser.add_argument('--train_batch_size', type=int,
                        default=DEFAULT_TRAIN_BATCH_SIZE,
                        help='Training batch size.')
    parser.add_argument('--test_batch_size', type=int,
                        default=DEFAULT_TEST_BATCH_SIZE,
                        help='Testing batch size.')
    parser.add_argument('--model_path', type=str,
                        help='Path to model to be reloaded.')
    return parser.parse_args()


def generate_labels_string(batch_pred, vocab):
    # Loop over entire batch list of predicted labels
    # and convert them to strings.
    batch_strings = []
    for pred in batch_pred:
        batch_strings.append(''.join([vocab[pred[i]]
                                      for i in range(len(pred))]))
    return batch_strings


def find_best_word(sample, word_dict):
    # find best possible/closest word to predicted word.
    if sample in word_dict:
        return sample, 0
    else:
        best_word = sample
        best_dist = 500
        for idx, word in enumerate(word_dict):
            dist = Lev.distance(sample, word)
            if dist < best_dist:
                best_dist = dist
                best_word = word
        return best_word, best_dist


def map_strings_to_closest_words(pred_list, word_dict):
    # cleans up predicted strings by mapping them
    # to the closest word in word dict
    print('\nCleaning up predicted strings.')
    new_pred_list = []
    for idx, pred_str in enumerate(pred_list):
        print('Predicted String: %d/%d' % (idx+1, len(pred_list)),
              end="\r", flush=True)
        new_pred_str = []
        words = pred_str.split(" ")
        for w in words:
            new_w, _ = find_best_word(w, word_dict)
            new_pred_str.append(new_w)
        new_pred_list.append(" ".join(new_pred_str))
    return new_pred_list


def character_error_rate(pred, targets):
    assert len(pred) == len(targets)
    dist = []
    for idx, p in enumerate(pred):
        dist.append(Lev.distance(p, targets[idx]))
    return dist


def plot_grad_flow(named_parameters, batch, epoch):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(-10)
            else:
                ave_grads.append(p.grad.abs().mean())
    fig = plt.figure()
    plt.plot(ave_grads, alpha=0.3, color="r")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(GRAD_FIGURES_PATH, str(epoch))
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
    fig.savefig(save_path+"/gradient_flow_"+str(batch)+".png")
    plt.close()


def greedy_decode(outputs, eos_token):
    probs = F.softmax(outputs, dim=2)
    preds = torch.argmax(probs, dim=2)
    # Iterate over each item in batch.
    pred_list = []
    for i in range(preds.size(0)):
        eos_idx = (preds[i] == eos_token).nonzero()
        eos_idx = (len(preds[i])-1) if eos_idx.nelement() == 0 else eos_idx[0]
        # pick all predicted chars excluding eos
        pred_list.append(preds[i, :eos_idx])
    return pred_list


def decode_and_cer(outputs, targets, tar_lens, vocab):
    eos_token = vocab.index('<eos>')
    pred_list = greedy_decode(outputs, eos_token)
    # exclude eos and sos
    tar_list = [targets[i, 1:(tar_lens[i]-1)] for i in range(targets.shape[0])]
    # Calculate the strings for predictions.
    pred_str = generate_labels_string(pred_list, vocab)
    # Calculate the strings for targets.
    tar_str = generate_labels_string(tar_list, vocab)
    # Calculate edit distance between predictions and targets.
    return character_error_rate(pred_str, tar_str), pred_str, tar_str


def save_test_results(predictions):
    predictions_count = list(range(len(predictions)))
    csv_output = [[i, j] for i, j in zip(predictions_count, predictions)]
    if not os.path.isdir(TEST_RESULT_PATH):
        os.makedirs(TEST_RESULT_PATH, exist_ok=True)
    result_file_path = os.path.join(TEST_RESULT_PATH, 'result_{}.csv'.format(
        (str.split(str.split(args.model_path, '/')[-1], '.pt')[0])))
    with open(result_file_path, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['Id', 'Predicted'])
        csv_writer.writerows(csv_output)


def test_model(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        all_predictions = []
        for batch_idx, (inputs, _, _, _, _, seq_order) in \
                enumerate(test_loader):
            outputs, attention_weights = model(inputs, None, 0)
            torch.save(attention_weights, 'attention_weights_test.pt')
            # make outputs batch first.
            outputs = outputs.permute(1, 0, 2)
            eos_token = test_loader.dataset.vocab.index('<eos>')
            pred_list = greedy_decode(outputs, eos_token)
            pred_str = generate_labels_string(pred_list,
                                              test_loader.dataset.vocab)
            # Input is sorted as per length for rnn. Resort the output.
            reorder_seq = np.argsort(seq_order)
            pred_str = [pred_str[i] for i in reorder_seq]
            all_predictions.extend(pred_str)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)),
                  end="\r", flush=True)
        end_time = time.time()
        # Try to map words in strings to closest words.
        # all_predictions = map_strings_to_closest_words(all_predictions,
        #                                                word_dict)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        print('\nTotal Test Predictions: %d Time: %d s' % (
            len(all_predictions), end_time - start_time))


# NOTE: Batch size must be one for test_model2 !!
def test_model2(model, test_loader, device, word_dict):
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        all_predictions = []
        eos_token = test_loader.dataset.vocab.index('<eos>')
        for batch_idx, (inputs, _, _, _, _, seq_order) in \
                enumerate(test_loader):
            hypos = beam_search(model.get_initial_state, model.generate,
                                inputs, eos_token, batch_size=1, beam_width=8,
                                num_hypotheses=1, max_length=250)
            pred_list = []
            for n in hypos:
                nn = n.to_sequence_of_values()
                pred_list.append(nn[1:][:-1])
            attention_weights = [n.to_sequence_of_extras() for n in hypos]
            pred_str = generate_labels_string(pred_list,
                                              test_loader.dataset.vocab)
            torch.save(attention_weights, 'attention_weights_test.pt')
            all_predictions.extend(pred_str)
            print('Test Iteration: %d/%d' % (batch_idx+1, len(test_loader)),
                  end="\r", flush=True)
        # Try to map words in strings to closest words.
        # all_predictions = map_strings_to_closest_words(all_predictions,
        #                                                word_dict)
        # Save predictions in csv file.
        save_test_results(all_predictions)
        end_time = time.time()
        print('\nTotal Test Predictions: %d Time: %d s' % (
            len(all_predictions), end_time - start_time))


def val_model(model, val_loader, device, sb):
    with torch.no_grad():
        model.eval()
        dist = []
        start_time = time.time()
        for batch_idx, (inputs, _, targets, _, tar_lens, _) in \
                enumerate(val_loader):
            targets = targets.to(device)
            outputs, attention_weights = model(inputs, None, 0)
            torch.save(attention_weights, 'attention_weights_val.pt')
            # make targets and outputs batch first.
            targets = targets.permute(1, 0)
            outputs = outputs.permute(1, 0, 2)
            # Decode and get edit distance.
            distances, pred_str, tar_str = \
                decode_and_cer(outputs, targets, tar_lens,
                               val_loader.dataset.vocab)
            dist.extend(distances)
            for i in range(len(distances)):
                sb.addItem([distances[i], pred_str[i], tar_str[i]])
            print('Validation Iteration: %d/%d' %
                  (batch_idx+1, len(val_loader)),
                  end="\r", flush=True)
        end_time = time.time()
        dist = sum(dist)/len(dist)    # Average over edit distance.
        print('\nValidation -> Edit Distance: %5.3f Time: %d s' %
              (dist, end_time - start_time))
        return dist


def train_model(model, train_loader, criterion, optimizer, device, tf, epoch,
                sb):
    model.train()
    running_loss = 0.0
    running_lens = 0.0
    dist = []
    measure_training_accuracy = True
    start_time = time.time()
    for batch_idx, (inputs, _, targets, targets_loss, tar_lens, _) in \
            enumerate(train_loader):
        targets, targets_loss = targets.to(device), targets_loss.to(device)
        optimizer.zero_grad()
        outputs, attention_weights = model(inputs, targets, tf)
        torch.save(attention_weights, 'attention_weights_train.pt')
        # make targets and outputs batch first.
        targets, targets_loss = \
            targets.permute(1, 0), targets_loss.permute(1, 0)
        outputs = outputs.permute(1, 0, 2)
        loss = criterion(outputs.contiguous().view(-1, outputs.size(2)),
                         targets_loss[:, 1:].contiguous().view(-1))
        running_loss += loss.item()
        running_lens += float(sum(tar_lens))
        loss = loss/len(tar_lens)   # Average over batch.
        loss.backward()
        plot_grad_flow(model.named_parameters(), batch_idx+1, epoch)
        # To avoid exploding gradient issue.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if measure_training_accuracy:
            distances, pred_str, tar_str = \
                decode_and_cer(outputs, targets, tar_lens,
                               train_loader.dataset.vocab)
            dist.extend(distances)
            for i in range(len(distances)):
                sb.addItem([distances[i], pred_str[i], tar_str[i]])
        curr_avg_loss = (running_loss/running_lens)
        curr_perp = np.exp(curr_avg_loss)
        print('Train Iteration: %d/%d Loss = %5.4f, Perplexity = %5.4f' %
              (batch_idx+1, len(train_loader), curr_avg_loss, curr_perp),
              end="\r", flush=True)
    end_time = time.time()
    # Average over edit distance.
    dist = sum(dist)/len(dist) if measure_training_accuracy else -1
    running_loss = (running_loss/running_lens)
    perplexity = np.exp(running_loss)
    print('\nTraining -> Loss: %5.4f Perplexity: %5.4f '
          'Edit Distance: %5.4f Time: %d s'
          % (running_loss, perplexity, dist, end_time - start_time))
    return running_loss


if __name__ == "__main__":
    # Parse args.
    args = parse_args()
    print('='*20)
    print('Input arguments:\n%s' % (args))

    # Validate arguments.
    if args.mode == 'test' and args.model_path is None \
            and not args.model_ensemble:
        raise ValueError("Input Argument Error: Test mode specified "
                         "but model_path is %s." % (args.model_path))

    # Check for CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders.
    speechTrainDataset = SpeechDataset(mode='train')    # noqa F405
    speechValDataset = SpeechDataset(mode='dev')    # noqa F405
    speechTestDataset = SpeechDataset(mode='test')  # noqa F405

    train_loader = DataLoader(speechTrainDataset,
                              batch_size=args.train_batch_size,
                              shuffle=True, num_workers=1,
                              collate_fn=SpeechCollateFn)   # noqa F405
    val_loader = DataLoader(speechValDataset, batch_size=args.train_batch_size,
                            shuffle=False, num_workers=4,
                            collate_fn=SpeechCollateFn) # noqa F405
    test_loader = DataLoader(speechTestDataset, batch_size=1,
                             shuffle=False, num_workers=4,
                             collate_fn=SpeechCollateFn)    # noqa F405

    # Prepare a dictionary of words. Used to clean up final predictions.
    WORD_DICT = []
    ALL_LABELS = [speechTrainDataset.labels_raw, speechValDataset.labels_raw]
    for curr_labels in ALL_LABELS:
        for utt in curr_labels:
            words = utt.split(" ")
            for w in words:
                if w not in WORD_DICT:
                    WORD_DICT.append(w)
    print('Prepared word dictionary: %d words' % (len(WORD_DICT)))
    print('='*20)

    # Set random seed.
    np.random.seed(DEFAULT_RANDOM_SEED)
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    if device == "cuda":
        torch.cuda.manual_seed(DEFAULT_RANDOM_SEED)

    # Create the model.
    model = Seq2Seq(base=128, vocab_dim=speechTrainDataset.vocab_size,
                    device=device)
    model.to(device)
    print('='*20)
    print(model)
    model_params = sum(p.size()[0] * p.size()[1] if len(p.size()) > 1
                       else p.size()[0] for p in model.parameters())
    print('Total model parameters:', model_params)
    print("Running on device = %s." % (device))

    # Setup learning parameters.
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_ID)    # noqa F405
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2,
                                                     threshold=0.01,
                                                     verbose=True)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print('Loaded model:', args.model_path)

    n_epochs = 100
    start_epoch = 0
    print('='*20)

    train_scoreboard = Scoreboard(sort_param_idx=0, name='Train')
    val_scoreboard = Scoreboard(sort_param_idx=0, name='Val')

    teacher_force = 0.9
    if args.mode == 'train':
        for epoch in range(start_epoch, n_epochs):
            print('Epoch: %d/%d' % (epoch+1, n_epochs))
            train_loss = train_model(model, train_loader, criterion, optimizer,
                                     device, teacher_force, epoch+1,
                                     train_scoreboard)
            val_dist = val_model(model, val_loader, device, val_scoreboard)
            # Print scoreboards.
            train_scoreboard.print_scoreboard(k=10, key=SCOREBOARD_KEY)
            val_scoreboard.print_scoreboard(k=10, key=SCOREBOARD_KEY)
            train_scoreboard.flush()
            val_scoreboard.flush()
            # Checkpoint the model after each epoch.
            finalValDist = '%.3f' % (val_dist)
            if not os.path.isdir(MODEL_PATH):
                os.mkdir(MODEL_PATH)
            model_path = os.path.join(MODEL_PATH, 'model_{}_val_{}.pt'.format(
                time.strftime("%Y%m%d-%H%M%S"), finalValDist))
            torch.save(model.state_dict(), model_path)
            print('='*20)
            # Update learning rate as required.
            optim_state = optimizer.state_dict()
            if epoch < 4:
                optim_state['param_groups'][0]['lr'] = 1e-4     # warmup
            elif epoch < 30:
                optim_state['param_groups'][0]['lr'] = 1e-3
            elif epoch < 75:
                optim_state['param_groups'][0]['lr'] = 1e-4
            else:
                optim_state['param_groups'][0]['lr'] = 1e-5
            # Teacher Forcing Schedule
            if epoch >= 7 and teacher_force > 0.7:
                teacher_force -= 0.005
    else:
        # Only testing the model.
        test_model(model, test_loader, device)
        # test_model2(model, test_loader, device, WORD_DICT)
    print('='*20)
