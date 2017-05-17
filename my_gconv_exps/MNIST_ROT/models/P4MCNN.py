import chainer.functions as F
from chainer import Chain
from groupy.gconv.chainer_gconv.p4_conv import P4ConvP4, P4ConvZ2

from groupy.gconv.chainer_gconv.p4m_conv import P4MConvZ2, P4MConvP4M
from conv_bn_act import ConvBNAct


class P4MCNN(Chain):

    def __init__(self):
        ksize = 3
        bn = True
        act = F.relu
        self.dr = 0.3
        super(P4MCNN, self).__init__(

            l1=ConvBNAct(
                conv=P4MConvZ2(in_channels=1, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l2=ConvBNAct(
                conv=P4MConvP4M(in_channels=10, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l3=ConvBNAct(
                conv=P4MConvZ2(in_channels=10, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l4=ConvBNAct(
                conv=P4MConvP4M(in_channels=10, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l5=ConvBNAct(
                conv=P4MConvP4M(in_channels=10, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            l6=ConvBNAct(
                conv=P4MConvP4M(in_channels=10, out_channels=10, ksize=ksize, stride=1, pad=0),
                bn=bn,
                act=act
            ),

            top=P4MConvP4M(in_channels=10, out_channels=10, ksize=1, stride=1, pad=0),
        )

    def __call__(self, x, t, train=True, finetune=False):

        h = self.l1(x, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l2(h, train, finetune)
        #print(h.shape) # (128, 10, 8, 24, 24)
        # max pool?
        # h = F.sum(h, axis=-3, keepdims=False)/8
        h = F.max(h, axis=-3, keepdims=False) # mean and max works similarly
        h = F.max_pooling_2d(h, ksize=2, stride=2,
                                pad=0, cover_all=True, use_cudnn=True)
        #print(h.shape)
        #h = F.dropout(h, self.dr, train)
        h = self.l3(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l4(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l5(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        h = self.l6(h, train, finetune)
        h = F.dropout(h, self.dr, train)
        #print(h.shape) # (N, 10, 8, 16, 16), use max pool: (N, 10, 8, 4, 4)
        h = self.top(h)
        #print(h.shape) # (N, 10, 8, 16, 16)

        h = F.sum(h, axis=-3, keepdims=False)
        h = F.sum(h, axis=-1, keepdims=False)
        h = F.sum(h, axis=-1, keepdims=False)
        h = h/(8*4*4)

        return F.softmax_cross_entropy(h, t), F.accuracy(h, t)

    def start_finetuning(self):
        for c in self.children():
            if isinstance(c, ConvBNAct):
                if c.bn:
                    c.bn.start_finetuning()

