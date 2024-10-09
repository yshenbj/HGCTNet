import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from einops import rearrange, reduce, repeat


cfg = {
    'A': [32, 32, 'M', 64, 64, 'M']
}


def make_layers(cfg, in_channels = 3, batch_norm=True):
    layers = [] 
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1,dilation=1)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens):
        super(AdditiveAttention, self).__init__()
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, queries, keys):
        # queries's shape: [bs, query_size]
        # keys's shape: [bs, seq_len, query_size]
        queries = queries.unsqueeze(1)# [bs, query_size]->[bs, 1, query_size]
        queries = self.w_q(queries)
        keys = self.w_k(keys)
        features = torch.tanh(queries + keys)
        scores = self.w_v(features)
        weights = self.softmax(scores)

        return weights


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, shortcut = True):
        super(BasicBlock, self).__init__()
        self.downsample = None

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm1d(planes)
        self.shortcut = shortcut
        if inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
                    
        if self.shortcut:
            out += identity

        out = self.relu(out)
        return out


class CNN_Net(nn.Module):
    def __init__(self, inplanes=3):
        super(CNN_Net, self).__init__()
        self.features = nn.Sequential(
            make_layers(cfg['A'], in_channels = inplanes, batch_norm=True),
            BasicBlock(inplanes=64, planes=128),
            nn.MaxPool1d(kernel_size=2, stride=3),
            BasicBlock(inplanes=128, planes=256),
            nn.MaxPool1d(kernel_size=2, stride=3),
            BasicBlock(inplanes=256, planes=256),
            nn.MaxPool1d(kernel_size=2, stride=3)
        )

    def forward(self, x):
        x = self.features(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
      
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask= None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input):
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.1,
                 forward_expansion=2,
                 forward_drop_p=0.1):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, num_heads, pos=True):
        super(TransformerEncoder, self).__init__()
        self.emb = PositionalEncoding(d_model=emb_size,dropout=0.1, max_len=256)
        self.layers = nn.ModuleList([TransformerEncoderBlock(emb_size, num_heads = num_heads) for _ in range(depth)])
        self.pos = pos

    def forward(self, x):
        
        if self.pos:
            x = self.emb(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x


class HGCTNet(nn.Module):
    def __init__(self, num_classes=2, num_hid = 8, depth = 3, num_heads = 4, hga = True, trans = True, init_weights = True):
        super(HGCTNet, self).__init__()
        self.hga = hga
        self.trans = trans
        self.cnn = CNN_Net()
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        
        # AdditiveAttention
        self.cnn_attention = AdditiveAttention(key_size = 111, query_size = 19, num_hiddens = num_hid)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1x1 = conv1x1(in_planes=11, out_planes=1) 
        self.emblay = nn.Sequential(
            nn.Linear(51,128, bias=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 256, bias=True),
        )
        self.rnn = TransformerEncoder(depth=depth, emb_size=256, num_heads = num_heads, pos=True)
        self.balance_alpha = torch.nn.Parameter(torch.randn(2))
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x, feas):
        basebp = feas[:,10:12]
        demos = feas[:,0:10]
        hand_feas = feas[:,12:31]

        # CNN module
        x_s = self.cnn(x)
        
        # HGA module
        if self.hga: 
            alpha = self.cnn_attention(hand_feas, x_s)
            x_s = torch.mul(alpha, x_s)
        x_res = self.avgpool(x_s)
        x_res = x_res.view(x_res.size(0), -1)
        
        # Transformer module
        if self.trans:
            x_s = x_s.permute(0, 2, 1)
            x_s = self.rnn(x_s)
            x_s = x_s.permute(0, 2, 1)
            x_s = self.avgpool(x_s)
            x_s = x_s[:,:,0]
        else:
            x_s = self.avgpool(x_s)
            x_s = x_s[:,:,0]
        
        # Feature fusion module
        x = x_s + x_res
        x = torch.mul(x,self.emblay(torch.cat([demos,hand_feas],dim=1)))
        
        # regression
        x = self.regressor(x)
        x = x*self.balance_alpha + basebp*(1-self.balance_alpha)

        return x
   
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d): 
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data,mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()