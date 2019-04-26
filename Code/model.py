import torch
import numpy as np
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable

# Locked dropout.
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # x: (B, L, C)
        if dropout == 0 or not self.training:
            return x
        mask = x.data.new(x.size(0), 1, x.size(2))
        mask = mask.bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

# Locked dropout.
class DecoderDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        # x: (B, C)
        if dropout == 0 or not self.training:
            return x
        mask = x.data.new(x.size(0), x.size(1)).bernoulli_(1 - dropout)
        mask = Variable(mask, requires_grad=False) / (1 - dropout)
        return mask * x

# Pyramidal Bidirectional RNN Layer.
class pBRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='lstm'):
        super(pBRNNLayer, self).__init__()
        self.rnn_type = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = self.rnn_type(input_size = (input_size*2), hidden_size=hidden_size,
                                num_layers = 1, bidirectional=True, batch_first=True)

    def forward(self, seq, seq_len):
        # Input: [Batch, Seq, Feat]
        batch_size = seq.size(0)
        time_dim = seq.size(1)
        feat_dim = seq.size(2)
        if time_dim % 2 != 0:
            seq = seq[:,:-1,:]      # Make sequence length even.
            time_dim -= 1
        # Reduce time dimension by 2.
        # Increase feature dimension by 2.
        seq = seq.contiguous().view(batch_size,time_dim//2, feat_dim * 2)
        seq_len = [l//2 for l in seq_len]
        out = rnn.pack_padded_sequence(seq, seq_len, batch_first=True)
        out, h = self.rnn(out)
        out, _ = rnn.pad_packed_sequence(out, batch_first=True)
        return out, h

# CNN to extract features.
class CnnBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CnnBlock, self).__init__()
        # Input Dim: [batch_size, feat_size, max_seq_len]
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            #nn.Conv1d(in_channels=input_size, out_channels=hidden_size,
            #            kernel_size=3, padding=1),  #stride=1
            #nn.BatchNorm1d(hidden_size),
            #nn.Hardtanh(inplace=True)   # Helps with exploding gradients.
        )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    #def __init__(self, input_size, hidden_size, nlayers, dropout_all):
    def __init__(self, rnn_type, input_size, hidden_size, nlayers, dropout_all):
        super(Encoder, self).__init__()
        self.nlayers = nlayers
        # Extract features using CNN.
        self.feat_extractor = CnnBlock(input_size, hidden_size)
        # If using feat_extractor, change pBRNNLayer input size to hidden_size.
        self.rnn = nn.ModuleList()
        for i in range(nlayers):
            self.rnn.append(pBRNNLayer(input_size if i==0 else (hidden_size*2),
                                        hidden_size, rnn_type))
        # Variational Dropout for RNN.
        self.rnn_dropout = LockedDropout()
        # RNN input layer, RNN hidden layer, Linear Layer.
        self.dropout_i, self.dropout_h, self.dropout = dropout_all

    def forward(self, seq, seq_len):
        # Input Dim: Padded - [max_seq_len, batch, feat_size]
        seq = seq.permute(1,2,0).contiguous()
        # Dim: [batch, feat_size, max_seq_len]
        out = self.feat_extractor(seq)
        out = out.permute(0,2,1)
        # Dim: [batch, max_seq_len, feat_size]
        out = self.rnn_dropout(out, self.dropout_i)     # Dropout for input layer.
        for l, rnn in enumerate(self.rnn):
            out, _ = rnn(out, seq_len)
            seq_len = [l//2 for l in seq_len]   #pBRNN reduces sequence length by 2x.
            if l != (self.nlayers - 1):
                out = self.rnn_dropout(out, self.dropout_h)
        out = self.rnn_dropout(out, self.dropout)
        return out, seq_len

class Attention(nn.Module):
    def __init__(self, key_query_dim, value_dim, encoder_feat_dim, decoder_feat_dim):
        super(Attention, self).__init__()
        self.fc_query = nn.Linear(decoder_feat_dim, key_query_dim)
        self.fc_key = nn.Linear(encoder_feat_dim, key_query_dim)
        self.fc_value = nn.Linear(encoder_feat_dim, value_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_feat, encoder_feat, seq_len):
        # encoder_feat: BxLxC
        query = self.activation(self.fc_query(decoder_feat))
        key = self.activation(self.fc_key(encoder_feat))
        value = self.activation(self.fc_value(encoder_feat))
        # query: BxC, key: BxLxC, value: BxLxC
        energy = torch.bmm(query.unsqueeze(1), key.transpose(1,2)).squeeze(1)
        # energy: BxL
        attention_score = self.softmax(energy)
        # need to mask out softmax values for indices after sequence length.
        attention_mask = torch.zeros_like(energy)
        for i, s_len in enumerate(seq_len):
            attention_mask[i,:s_len] = 1
        attention_score = attention_score * attention_mask
        # rescale values so that probs sum to 1
        rescale_factor = torch.sum(attention_score,dim=1).unsqueeze(1).expand_as(attention_score)
        attention_score = attention_score/rescale_factor
        # attention_score: BxL
        # context: weighted sum of value using attention_score as weights
        context = torch.bmm(attention_score.unsqueeze(1), value).squeeze(1)
        # context: BxC
        return attention_score, context

class Decoder(nn.Module):
    def __init__(self, hidden_size, context_dim, nlayers, attention, vocab_size,
                        dropout_all, max_iter, teacher_force_rate, eos_idx=0):
        super(Decoder, self).__init__()
        # hidden_size must be equal to encoder_feat_dim
        self.attention = attention
        self.nlayers = nlayers
        # RNN input layer, RNN hidden layer, Linear Layer.
        self.dropout_h, self.dropout = dropout_all
        self.rnn_dropout = DecoderDropout()
        self.max_iter = max_iter
        self.teacher_force_rate = teacher_force_rate
        self.eos_idx = eos_idx
        self.char_embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.ModuleList()
        self.init_h = nn.ParameterList()
        self.init_c = nn.ParameterList()
        for i in range(nlayers):
            self.rnn.append(nn.LSTMCell((hidden_size+context_dim) if i==0 else hidden_size,
                                        hidden_size))
            self.init_h.append(nn.Parameter(torch.rand(1, hidden_size)))
            self.init_c.append(nn.Parameter(torch.rand(1, hidden_size)))
        self.fc = nn.Linear((hidden_size+context_dim), hidden_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        #self.classifier = nn.Embedding(hidden_size, vocab_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.classifier.weight = self.char_embed.weight     # weight tying
        self.mlp = nn.Sequential(self.fc, self.activation, self.classifier)

    def forward(self, encoder_feat, seq_len, ground_truth=None):
        if self.training:
            assert(ground_truth is not None)
        # no teacher forcing during test/val.
        tfr = self.teacher_force_rate if self.training else 0
        tfr_valid = True if np.random.random_sample() < tfr else False
        # get initial hidden/cell state
        batch_size = encoder_feat.size(0)
        hidden, cell = self.init_hidden(batch_size)
        # first input is <eos>/<sos>
        prev_output = torch.empty(batch_size).long().fill_(self.eos_idx)
        # loop through sequence and predict
        max_steps = ground_truth.size(1) if (ground_truth is not None) else self.max_iter
        preds, attention_scores = [], []
        for step in range(max_steps):
            prev_output = self.char_embed(prev_output)
            prev_decoder_feat = hidden[-1]
            attention_score, context = self.attention(prev_decoder_feat, encoder_feat, seq_len)
            rnn_input = torch.cat([prev_output, context], dim=1)
            new_hidden, new_cell = [], []
            for l, rnn in enumerate(self.rnn):
                h, c = rnn(rnn_input, (hidden[l], cell[l]))
                new_hidden.append(h)
                new_cell.append(c)
                if l != (self.nlayers - 1):
                    rnn_input = self.rnn_dropout(h, self.dropout_h)
            rnn_output = self.rnn_dropout(new_hidden[-1], self.dropout)
            mlp_input = torch.cat([rnn_output, context], dim=1)
            pred = self.mlp(mlp_input)
            # update recording lists.
            preds.append(pred)
            attention_scores.append(attention_score)
            # update prev_output for next step
            if tfr_valid:
                prev_output = ground_truth[:, step]
            else:
                _, prev_output = torch.max(pred, dim=1)
            # update hidden and cell states for next step
            hidden, cell = new_hidden, new_cell
        return torch.stack(preds, dim=1), attention_scores

    def init_hidden(self, batch_size):
        hidden = [h.repeat(batch_size, 1) for h in self.init_h]
        cell = [c.repeat(batch_size, 1) for c in self.init_c]
        return [hidden, cell]

class SpeechRecognizer(nn.Module):
    def __init__(self):
        super(SpeechRecognizer, self).__init__()
        self.encoder = Encoder(rnn_type='lstm', input_size=40, hidden_size=256, nlayers=3, dropout_all=(0.3, 0.1, 0.2))
        self.attention = Attention(key_query_dim=128, value_dim=128, encoder_feat_dim=512, decoder_feat_dim=256)
        self.decoder = Decoder(hidden_size=256, context_dim=128, nlayers=3, attention=self.attention,
                                vocab_size=33, dropout_all=(0.1,0.3), max_iter=250, teacher_force_rate=0.8, eos_idx=0)

    def forward(self, seq, seq_len, ground_truth=None):
        encoder_feat, out_seq_len = self.encoder(seq, seq_len)
        preds, attentions = self.decoder(encoder_feat, out_seq_len, ground_truth)
        return preds, attentions
