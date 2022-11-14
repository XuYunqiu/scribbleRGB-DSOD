import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import VGG16


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[64, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea*3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea*3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=False)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=False)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        ))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=False)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:], mode='bilinear', align_corners=False)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class MAFF(nn.Module):
    def __init__(self, channel, out_channel=None, reduction=16):
        super(MAFF, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.channel = channel
        if out_channel is not None:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(channel, channel, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel // 2, 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
            )
        self.bn = nn.Sequential(
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x = self.bn(torch.cat((x1, x2), dim=1))
        y = self.conv_du(self.avg_pool(x))
        y1_mean = torch.mean(y[:, :self.channel // 2], dim=1, keepdim=True).repeat([1, self.channel // 2, 1, 1])
        y2_mean = torch.mean(y[:, self.channel // 2:], dim=1, keepdim=True).repeat([1, self.channel // 2, 1, 1])
        y = (y + torch.cat((y1_mean, y2_mean), dim=1)) / 2
        ret = self.conv_layers(x * y)
        return ret


class DENet_VGG(nn.Module):
    def __init__(self, channel=32, warmup_stage=True):
        super(DENet_VGG, self).__init__()
        self.vgg = VGG16()
        self.relu = nn.ReLU(inplace=True)
        self.edge_layer = Edge_Module()
        self.edge_layer_depth = Edge_Module()
        self.aspp = _AtrousSpatialPyramidPoolingModule(512, 32, output_stride=16)
        self.aspp_depth = _AtrousSpatialPyramidPoolingModule(512, 32, output_stride=16)

        self.sal_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.edge_conv_depth = nn.Conv2d(1, channel, kernel_size=3, padding=1, bias=False)
        self.rcab_sal_edge = RCAB(channel*2)
        self.after_aspp_conv5 = MAFF(channel*6*2, out_channel=channel)
        self.after_aspp_conv2 = MAFF(128*2, out_channel=channel)
        self.final_sal_seg = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, 1, kernel_size=1, bias=False)
        )
        self.fuse_canny_edge = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fuse_canny_edge_depth = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.fused_edge_sal = nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False)
        self.maff = MAFF(channel * 4)
        self.warmup_stage = warmup_stage


    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, depth, image_edges, depth_edges):
        x_size = x.size()
        x1 = self.vgg.conv1(x)
        x2 = self.vgg.conv2(x1)
        x3 = self.vgg.conv3(x2)
        x4 = self.vgg.conv4(x3)
        x5 = self.vgg.conv5(x4)
        edge_map = self.edge_layer(x1, x3, x4)
        edge_out = torch.sigmoid(edge_map)
        
        d1 = self.vgg.conv1(depth)
        d2 = self.vgg.conv2(d1)
        d3 = self.vgg.conv3(d2)
        d4 = self.vgg.conv4(d3)
        d5 = self.vgg.conv5(d4)
        edge_map_depth = self.edge_layer_depth(d1, d3, d4)
        edge_out_depth = torch.sigmoid(edge_map_depth)

        cat = torch.cat((edge_out, image_edges), dim=1)
        acts = torch.sigmoid(self.fuse_canny_edge(cat))

        cat_depth = torch.cat((edge_out_depth, depth_edges), dim=1)
        acts_depth = torch.sigmoid(self.fuse_canny_edge_depth(cat_depth))
        
        d5 = self.aspp_depth(d5, acts_depth)
        x5 = self.aspp(x5, acts)

        x_conv5 = self.after_aspp_conv5(x5, d5)
        x_conv2 = self.after_aspp_conv2(x2, d2)
        
        x_conv5_up = F.interpolate(x_conv5, x2.size()[2:], mode='bilinear', align_corners=False)
        feat_fuse = torch.cat([x_conv5_up, x_conv2], 1)

        sal_init = self.final_sal_seg(feat_fuse)
        sal_init = F.interpolate(sal_init, x_size[2:], mode='bilinear', align_corners=False)
        sal_feature = self.sal_conv(sal_init)

        edge_feature = self.edge_conv(edge_map)
        sal_edge_feature = self.relu(torch.cat((sal_feature, edge_feature), 1))
        edge_feature_depth = self.edge_conv_depth(edge_map_depth)
        sal_edge_feature_depth = self.relu(torch.cat((sal_feature, edge_feature_depth), 1))

        sal_edge_feature = self.maff(sal_edge_feature, sal_edge_feature_depth)
        sal_edge_feature = self.rcab_sal_edge(sal_edge_feature)
        if self.warmup_stage:
            sal_ref = self.fused_edge_sal(sal_edge_feature)
        else:
            sal_ref = self.fused_edge_sal(F.dropout(self.relu(sal_edge_feature), p=0.25))
        return sal_init, edge_map, sal_ref, edge_map_depth