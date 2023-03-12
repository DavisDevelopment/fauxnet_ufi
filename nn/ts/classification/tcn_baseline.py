import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # Forced truncation,Keep dimensions the same
        self.relu1 = nn.ReLU()  # Remove decimal fraction
        self.dropout1 = nn.Dropout(dropout)  # Prevent over fitting

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, #两个n_outputs, n_outputs
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None # 1x1 conv
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        x: n*emb_size*seq_len
        out: n*layer_outchannel* seq_len"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            """dilated conv"""
            dilation_size = 2 ** i

            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        aa = self.network(x)
        return self.network(x)

class TrafficTCN(nn.Module):
    def __init__(self,emb_size,n_categs,channels_size,
                 kernel_size=2,dropout=0.3,emb_dropout=0.1,tied_weights=False):
        super(TrafficTCN,self).__init__()

        self.encoder = nn.Embedding(n_categs, emb_size)
        self.traffic_tcn = TemporalConvNet(emb_size, channels_size, kernel_size, dropout=dropout)
        self.decoder = nn.Linear(channels_size[-1], n_categs)  #卷积的输出channel 接着 n_categs
        self.drop = nn.Dropout(emb_dropout)
        self.init_weights()  # init weights

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)

    def forward(self, input):
        """
        input: n * sequence_len (sequence_len:你准备用多长的序列预测下一个值，预先可以配置)
        emb: n * sequence_len * emb_size
        """
        emb = self.drop(self.encoder(input.to(torch.int64)))
        """
        emb.transpose(1,2): n * emb_size * sequence_len
        y :                 n *  sequence_len * channes[-1]
        """
        y = self.traffic_tcn(emb.transpose(1,2)).transpose(1,2)
        """
        y: n *  sequence_len * n_cates
        """
        y = self.decoder(y)  # 如果我把这些整数值 全看成类别 不就简单了

        return y.contiguous()

class TCNBaseline(nn.Module):
   def __init__(self, in_channels:int, num_pred_classes:int=1, seq_len:int=1):
      super().__init__()
      self.in_channels = in_channels
      self.num_pred_classes = num_pred_classes
      
      self.tcn = TemporalConvNet(seq_len, in_channels, dropout=0.05)
      
   def forward(self, x):
      return self.tcn(x)