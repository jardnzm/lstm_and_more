
from torch.nn.utils import clip_grad_norm_
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
from lstmp import LSTMP

YourHidden = 768
YourProjec = 256
YourInput = 80

class YourClass(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        self.device = device
        self.hidden_size = YourHidden
        self.proj_size = YourProjec
        rnn_input_size = YourInput

        for l in range(model_num_layers):
            rnn = LSTMP(rnn_input_size, self.hidden_size, self.proj_size).to(device)
            self.add_module("lstm{}".format(l), rnn)
            rnn_input_size = self.proj_size

        self.linear = nn.Linear(in_features=YourProjec,
                                out_features=YourProjec).to(device)
        self.relu = torch.nn.ReLU().to(device)
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)

    def forward(self, input_seq, hidden_init=None):
        """
        input with shape batch,len,input_dim

        Computes the embeddings of a batch of input sequence.

        :param input_seq: tensor of shape (batch_size, n_frames, n_channels)
        :param hidden_init: initial hidden state of the LSTMP as a tensor of shape
        (num_layers, batch_size, hidden_size). Will default to a tensor of zeros
        if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        batch_size =  input_seq.size()[0]
        if hidden_init is None:
            cell = torch.zeros((batch_size, self.hidden_size), dtype=input_seq.dtype,
                                device=self.device)
            hidden = torch.zeros((batch_size, self.proj_size), dtype=input_seq.dtype,
                                device=self.device)

        for l in range(model_num_layers):
            lstm_layer = self._modules["lstm{}".format(l)]
            seq_out, cell, hidden = lstm_layer(input_seq, cell, hidden)

        embed_raw = self.relu(self.linear(seq_out.mean(dim=1)))
        # embed_raw = self.relu(self.linear(hidden)) if you prefer last rather than mean
        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds