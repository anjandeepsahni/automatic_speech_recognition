import os
import torch
import numpy as np
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset as Dataset

SPEECH_DATA_PATH = './../Data'

IGNORE_ID = -1
VOCAB = ['<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']

class SpeechDataset(Dataset):
    def __init__(self, mode='train'):
        # Check for valid mode.
        self.mode = mode
        valid_modes = {'train', 'dev', 'test'}
        if self.mode not in valid_modes:
            raise ValueError("SpeechDataset Error: Mode must be one of %r." % valid_modes)
        # Load the data and labels (labels = None for 'test' mode)
        self.data, self.labels = self.loadRawData()
        #self.data = [torch.from_numpy(data) for data in self.data]
        self.feature_size = self.data[0].shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = []
        if self.mode != 'test':
            label = []
            for w in self.labels[idx]:
                for c in w:
                    label.append(VOCAB.index(chr(c)))
                label.append(VOCAB.index(' '))
            label[-1] = VOCAB.index('<eos>')
            label = torch.from_numpy(np.array(label)).long()
        return data, label

    def loadRawData(self):
        if self.mode == 'train' or self.mode == 'dev':
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, '{}.npy'.format(self.mode)), encoding='bytes'),
                np.load(os.path.join(SPEECH_DATA_PATH, '{}_transcripts.npy'.format(self.mode)), encoding='bytes')
                )
        else:   # No labels in test mode.
            return (
                np.load(os.path.join(SPEECH_DATA_PATH, 'test.npy'), encoding='bytes'),
                None
                )

    def _generate_vocab():
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
    seq_order = sorted(range(len(inp_lens)), key=inp_lens.__getitem__, reverse=True)
    inputs = [inputs[i].type(torch.float32) for i in seq_order]     # RNN does not accept Float64.
    inp_lens = [len(seq) for seq in inputs]
    inputs = rnn.pad_sequence(inputs)
    tar_lens = []
    if targets:
        targets = [targets[i] for i in seq_order]
        tar_lens = [len(tar) for tar in targets]
        targets_loss = rnn.pad_sequence(targets, batch_first=True, padding_value=IGNORE_ID)
        targets = rnn.pad_sequence(targets, batch_first=True)
    return inputs, inp_lens, targets, tar_lens, targets_loss, seq_order
