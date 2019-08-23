import os

import numpy as np
import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset as Dataset

SPEECH_DATA_PATH = './../Data_Orig'

IGNORE_ID = -1
VOCAB = [
    '<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z', '_'
]


class SpeechDataset(Dataset):
    def __init__(self, mode='train'):
        # Check for valid mode.
        self.mode = mode
        valid_modes = {'train', 'dev', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("SpeechDataset Error: Mode must be one of %r." %
                             valid_modes)
        self.vocab = VOCAB
        self.vocab_size = len(VOCAB)
        # Load the data and labels (labels = None for 'test' mode)
        # Labels must be a set of strings.
        self.data, self.labels_raw = self.loadRawData()
        if self.mode != 'test':
            self.labels = self.labels_raw.copy()
            self._preprocess_labels()

    def _preprocess_labels(self):
        for idx in range(len(self.labels)):
            label = []
            for c in self.labels[idx]:
                label.append(self.vocab.index(c))
            label.append(self.vocab.index(' '))
            label[-1] = self.vocab.index('<eos>')
            label = [self.vocab.index('<eos>')] + label
            self.labels[idx] = torch.from_numpy(np.array(label)).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        if self.mode == 'test':
            label = None
        else:
            label = self.labels[idx]
        return data, label

    def loadRawData(self):
        if self.mode == 'train' or self.mode == 'dev':
            return (np.load(os.path.join(SPEECH_DATA_PATH,
                                         '{}.npy'.format(self.mode)),
                            encoding='bytes'),
                    np.load(os.path.join(
                        SPEECH_DATA_PATH,
                        '{}_transcripts.npy'.format(self.mode)),
                            encoding='bytes'))
        else:  # No labels in test mode.
            return (np.load(os.path.join(SPEECH_DATA_PATH, 'test.npy'),
                            encoding='bytes'), None)

    def _generate_vocab(self):
        vocab = set({})
        for utt in self.labels:
            for w in utt:
                for c in w:
                    vocab.add(chr(c))
        vocab = list(sorted(vocab))
        vocab = ['<eos>', ' '] + vocab
        return vocab


# Modify the batch in collate_fn to sort the
# batch in decreasing order of size.
def SpeechCollateFn(seq_list):
    inputs, targets = zip(*seq_list)
    inp_lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(inp_lens)),
                       key=inp_lens.__getitem__,
                       reverse=True)
    inputs = [inputs[i].type(torch.float32)
              for i in seq_order]  # RNN does not accept Float64.
    inp_lens = [len(seq) for seq in inputs]
    tar_lens = []
    targets_loss = None
    if targets[0] is not None:
        targets = [targets[i] for i in seq_order]
        tar_lens = [len(tar) for tar in targets]
        targets_loss = rnn.pad_sequence(targets, padding_value=IGNORE_ID)
        targets = rnn.pad_sequence(targets)
    return inputs, inp_lens, targets, targets_loss, tar_lens, seq_order
