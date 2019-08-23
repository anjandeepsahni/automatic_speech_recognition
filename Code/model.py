import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import (pack_padded_sequence, pack_sequence,
                                pad_packed_sequence)


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # x:  (L, B, C)
        if dropout == 0 or not self.training:
            return x
        mask = x.data.new(1, x.size(1), x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


'''
class WeightDrop(nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.dummy_flatten
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(self.dropout,
                                                            name_w))
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w))

    def dummy_flatten(*args, **kwargs):
        return

    def forward(self, *args):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=self.dropout,
                                            training=self.training)
            setattr(self.module, name_w, nn.Parameter(w))
        return self.module.forward(*args)
'''


class BackHook(torch.nn.Module):
    def __init__(self, hook):
        super(BackHook, self).__init__()
        self._hook = hook
        self.register_backward_hook(self._backward)

    def forward(self, *inp):
        return inp

    @staticmethod
    def _backward(self, grad_in, grad_out):
        self._hook()
        return None


class WeightDrop(torch.nn.Module):
    """
    Implements drop-connect, as per Merity, https://arxiv.org/abs/1708.02182
    """
    def __init__(self, module, weights, dropout=0, variational=False):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()
        self.hooker = BackHook(lambda: self._backward())

    def _setup(self):
        for name_w in self.weights:
            print('Applying weight drop of {} to {}'.format(
                self.dropout, name_w))
            w = getattr(self.module, name_w)
            self.register_parameter(name_w + '_raw', nn.Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            if self.training:
                mask = raw_w.new_ones((raw_w.size(0), 1))
                mask = torch.nn.functional.dropout(mask,
                                                   p=self.dropout,
                                                   training=True)
                w = mask.expand_as(raw_w) * raw_w
                setattr(self, name_w + "_mask", mask)
            else:
                w = raw_w
            rnn_w = getattr(self.module, name_w)
            rnn_w.data.copy_(w)

    def _backward(self):
        # transfer gradients from embeddedRNN to raw params
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')
            rnn_w = getattr(self.module, name_w)
            raw_w.grad = rnn_w.grad * getattr(self, name_w + "_mask")

    def forward(self, *args):
        self._setweights()
        return self.module(*self.hooker(*args))


class Encoder(nn.Module):
    def __init__(self, base=128, device="cpu"):
        super(Encoder, self).__init__()
        self.lstm1 = nn.LSTM(40, base, bidirectional=True)
        self.lstm2 = self.__make_layer__(base * 4, base)
        self.lstm3 = self.__make_layer__(base * 4, base)
        self.lstm4 = self.__make_layer__(base * 4, base)

        self.fc1 = nn.Linear(base * 2, base * 2)
        self.fc2 = nn.Linear(base * 2, base * 2)
        self.act = nn.SELU(inplace=True)

        self.drop = LockedDropout()
        self.device = device

    def _stride2(self, x):
        x = x[:x.size(0) // 2 * 2]  # make even
        x = self.drop(x, dropout=0.3)
        x = x.permute(1, 0, 2)  # seq, batch, feature -> batch, seq, feature
        x = x.reshape(x.size(0), x.size(1) // 2, x.size(2) * 2)
        x = x.permute(1, 0, 2)  # batch, seq, feature -> seq, batch, feature
        return x

    def __make_layer__(self, in_dim, out_dim):
        lstm = nn.LSTM(input_size=in_dim,
                       hidden_size=out_dim,
                       bidirectional=True)
        # return lstm
        return WeightDrop(lstm, ['weight_hh_l0', 'weight_hh_l0_reverse'],
                          dropout=0.5)

    def forward(self, x):
        # x is list of variable length inputs.
        x = pack_sequence(x)  # seq, batch, 40
        x = x.to(self.device)

        x, _ = self.lstm1(x)  # seq, batch, base*2
        x, seq_len = pad_packed_sequence(x)
        x = self._stride2(x)  # seq//2, batch, base*4

        x = pack_padded_sequence(x, seq_len // 2)
        x, _ = self.lstm2(x)  # seq//2, batch, base*2
        x, _ = pad_packed_sequence(x)
        x = self._stride2(x)  # seq//4, batch, base*4

        x = pack_padded_sequence(x, seq_len // 4)
        x, _ = self.lstm3(x)  # seq//4, batch, base*2
        x, _ = pad_packed_sequence(x)
        x = self._stride2(x)  # seq//8, batch, base*4

        x = pack_padded_sequence(x, seq_len // 8)
        x, (hidden, _) = self.lstm4(x)  # seq//8, batch, base*2
        x, _ = pad_packed_sequence(x)

        key = self.act(self.fc1(x))
        value = self.act(self.fc2(x))
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1)
        return seq_len // 8, key, value, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hidden2, key, value, mask):
        # key: seq, batch, base     # value: seq, batch, base
        # mask: batch, seq          # hidden2: batch, base
        # batch, 1, base X batch, base, seq -> batch, 1, seq
        attn = torch.bmm(hidden2.unsqueeze(1), key.permute(1, 2, 0))
        attn = F.softmax(attn, dim=2)
        attn = attn * mask.unsqueeze(1).float()
        attn = attn / attn.sum(2).unsqueeze(2)

        # batch, 1, seq X batch, seq, base -> batch, 1, base
        context = torch.bmm(attn, value.permute(1, 0, 2)).squeeze(1)

        # context: batch, 1, base -> batch, base
        # attn: batch, 1, seq -> batch, seq
        return context.squeeze(1), attn.cpu().squeeze(1).data.numpy()


class Decoder(nn.Module):
    def __init__(self, vocab_dim, lstm_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim * 2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_dim, vocab_dim)
        self.fc.weight = self.embed.weight  # weight tying

    def forward(self, x, context, hidden1, cell1, hidden2, cell2, first_step):
        # x is batch x 1. Contains word for previous timestep.
        x = self.embed(x)
        x = torch.cat([x, context], dim=1)
        if first_step:
            hidden1, cell1 = self.lstm1(x)
            hidden2, cell2 = self.lstm2(hidden1)
        else:
            hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
            hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
        x = self.drop(hidden2)
        x = self.fc(x)
        return x, hidden1, cell1, hidden2, cell2


class Seq2Seq(nn.Module):
    def __init__(self, base, vocab_dim, device="cpu"):
        super().__init__()
        self.base = base
        self.device = device
        self.vocab_dim = vocab_dim
        self.encoder = Encoder(base=base, device=device)
        self.attention = Attention()
        self.decoder = Decoder(vocab_dim=vocab_dim, lstm_dim=base * 2)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.constant_(param.data, 0)

    def sample_gumbel(self, shape, eps=1e-10, out=None):
        U = out.resize_(shape).uniform_() if out is not None else torch.rand(
            shape)
        return -torch.log(eps - torch.log(U + eps))

    def forward(self, inputs, words, TF):
        if self.training:
            word, hidden1, cell1, hidden2, cell2 = words[
                0, :], None, None, None, None
            words = words[1:, :]  # Removing sos, already saved in word
            max_len, batch_size = words.shape[0], words.shape[1]
        else:
            max_len, batch_size = 251, len(inputs)
            word = torch.zeros(batch_size).long().to(self.device)
            hidden1, cell1, hidden2, cell2 = None, None, None, None
            TF = 0  # no teacher forcing for test and val.

        prediction = torch.zeros(max_len, batch_size,
                                 self.vocab_dim).to(self.device)

        # Run through encoder.
        lens, key, value, hidden2 = self.encoder(inputs)
        mask = torch.arange(lens.max()).unsqueeze(0) < lens.unsqueeze(1)
        mask = mask.to(self.device)

        attention_weights = []

        for t in range(max_len):
            context, attention = self.attention(hidden2, key, value, mask)
            word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(
                word,
                context,
                hidden1,
                cell1,
                hidden2,
                cell2,
                first_step=(t == 0))
            prediction[t] = word_vec
            teacher_force = torch.rand(1) < TF
            if teacher_force:
                word = words[t]
            else:
                gumbel = torch.autograd.Variable(
                    self.sample_gumbel(shape=word_vec.size(),
                                       out=word_vec.data.new()))
                word_vec += gumbel
                word = word_vec.max(1)[1]
            attention_weights.append(attention)
        return prediction, attention_weights

    def get_initial_state(self, inputs, batch_size):
        self._lens, self._key, self._value, hidden2 = self.encoder(inputs)
        self._mask = torch.arange(
            self._lens.max()).unsqueeze(0) < self._lens.unsqueeze(1)
        self._mask = self._mask.to(self.device)
        word = torch.zeros(batch_size).long().to(self.device)
        return [None, None, hidden2, None], word  # Initial state is none.

    def generate(self, prev_words, prev_states):
        new_states, raw_preds, attention_scores = [], [], []
        for prev_word, prev_state in zip(prev_words, prev_states):
            prev_word = Variable(
                self._value.data.new(1).long().fill_(int(prev_word)))
            hidden1, cell1, hidden2, cell2 = prev_state[0], prev_state[
                1], prev_state[2], prev_state[3]
            context, attention = self.attention(hidden2, self._key,
                                                self._value, self._mask)
            first_step = False
            if prev_state[0] is None:
                first_step = True  # First timestep.
            word_vec, hidden1, cell1, hidden2, cell2 = self.decoder(
                prev_word,
                context,
                hidden1,
                cell1,
                hidden2,
                cell2,
                first_step=first_step)
            new_state = [hidden1, cell1, hidden2, cell2]
            new_states.append(new_state)
            raw_preds.append(
                F.softmax(word_vec, dim=1).squeeze().data.cpu().numpy())
            attention_scores.append(attention)
        return new_states, raw_preds, attention_scores
