def lstm_step(fw, state, output, rec_weight, proj_weight, bias,
        ln_ih, ln_hh, ln_ho):
    """
    there are 4 gates in lstm, 
    input, forget, output, updates.
    all gates have activate(wx + wh),
    only the update gate use tanh as activation, 
    the rest use sigmoid.
    state: the value of new cell
    output: the hidden state after projection
    """
    _, cell_size = state.shape
    gates = ln_ih(fw) + ln_hh( torch.matmul(output, rec_weight) ) + bias
    input_gate, forget_gate, output_gate, update = gates.chunk(4, 1)
    state = forget_gate.sigmoid() * state + input_gate.sigmoid() * update.tanh()
    output = output_gate.sigmoid() * ln_ho(state).tanh()
    output = torch.matmul(output, proj_weight)
    return state, output


class LSTMP(nn.Module):

    def __init__(self, input_size, cell_size, proj_size):
        super(LSTMP, self).__init__()

        self.input_size = input_size
        self.cell_size = cell_size
        self.proj_size = proj_size

        self.wx = nn.Parameter(torch.randn((input_size, 4 * cell_size)))
        # take a look at different initializaiton here 
        # https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
        self.wx.data.uniform_(-math.sqrt(2 / (input_size + 4 * cell_size)),
                            math.sqrt(2 / (input_size + 4 * cell_size)))

        self.wh = nn.Parameter(torch.randn((proj_size, 4 * cell_size)))
        self.wh.data.uniform_(-math.sqrt(2 / (proj_size + 4 * cell_size)),
                            math.sqrt(2 / (proj_size + 4 * cell_size)))

        self.wp = nn.Parameter(torch.randn((cell_size, proj_size)))
        self.wp.data.uniform_(-math.sqrt(2 / (proj_size + cell_size)),
                                        math.sqrt(2 / (proj_size + cell_size)))

        self.bias = nn.Parameter(torch.zeros((4 * cell_size)))
        self.bias.data[: cell_size].fill_(-1)
        self.bias.data[cell_size : 2 * cell_size].fill_(1)

        self.ln_ih = nn.LayerNorm(4 * cell_size)
        self.ln_hh = nn.LayerNorm(4 * cell_size)
        self.ln_ho = nn.LayerNorm(cell_size)


    def forward(self, x, state, output):
        """
        x with shape batch,utt_len,project
        """
        seq_len = x.size()[1]
        fw = torch.matmul(x, self.wx)
        fw = fw.unbind(1)

        seq_out = []
        for t in range(seq_len):
            state, output = lstm_step(fw[t], state, output, self.wh, self.wp,
                                self.bias, self.ln_ih, self.ln_hh, self.ln_ho)
            seq_out.append(output)
        seq_out = torch.stack(seq_out, 1)
        return seq_out, state, output
