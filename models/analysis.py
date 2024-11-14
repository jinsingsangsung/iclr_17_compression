#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .basics import *
import pickle
import os
import codecs
from .mambaconv import MambaConv2d

class Analysis_net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, use_ssm=False):
        super(Analysis_net, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (3 + out_channel_N) / (6))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)
        self.gdn3 = GDN(out_channel_N)
        self.conv4 = nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(self.conv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_N) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)
        self.downscaler = nn.ModuleList([
                  nn.AvgPool2d(5, stride=2, padding=2),
                  nn.AvgPool2d(5, stride=2, padding=2),
                  nn.AvgPool2d(5, stride=2, padding=2),
                  nn.AvgPool2d(5, stride=2, padding=2),
             ])
        if use_ssm:     
             self.mambaconvs = nn.ModuleList([
                 MambaConv2d(3, out_channel_N, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, out_channel_N, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, out_channel_N, kernel_size=2, stride=2, padding=0, bias=False, dim_preserve=True),
                 MambaConv2d(3, out_channel_N, kernel_size=2, stride=2, padding=0, bias=False, dim_preserve=True),
             ])
             self.initial_mambaconv = MambaConv2d(3, 3, kernel_size=4, stride=4, padding=0, bias=False, dim_preserve=True)        
        self.use_ssm = use_ssm
        self.out_channel_N = out_channel_N        

    def forward(self, x):
        interm_signals = []
        x_ = x.clone()
        if self.use_ssm:
             for downscaler, mambaconv in zip(self.downscaler, self.mambaconvs):
                 x_ = downscaler(x_)
                 m = mambaconv(x_)
                 if x_.size(2)%2 == 1 or x_.size(3)%2 == 1:
                      padding = (0, x_.size(3)%2, 0, x_.size(2)%2)
                      m = F.pad(m, padding, "replicate")
                 interm_signals.append(m)
             x = self.initial_mambaconv(x)
        else:
             for downscaler in self.downscaler:
                x_ = downscaler(x_)
                h, w = x_.shape[-2:]
                interm_signals.append(torch.zeros(x_.size(0), self.out_channel_N, h, w, \
                                                  device=x.device, requires_grad=False))    
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))
        return self.conv4(x)


def build_model():
        input_image = torch.zeros([4, 3, 256, 256])

        analysis_net = Analysis_net()
        feature = analysis_net(input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()
