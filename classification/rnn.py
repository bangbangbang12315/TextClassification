import torch
import torch.nn as nn
import torch.nn.functional as F
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, embedding=None, variable_lengths=False, n_layers=1, decode_function=F.log_softmax):
        super(RNN, self).__init__()

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.hidden_size = hidden_size
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers,
                                 batch_first=True)

        self.out = nn.Linear(self.hidden_size, 2)

    def forward(self, input_variable, input_lengths=None, label=None):
        embedded = self.embedding(input_variable)

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        result = self.out(hidden[-1])
        return result

