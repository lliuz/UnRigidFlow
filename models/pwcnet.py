import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.warp_utils import flow_warp
from correlation_package.correlation import Correlation
from .net_blocks import FeaturePyramidExtractor, OpticalFlowEstimator, ContextNetwork


class PWCFlow(nn.Module):
    def __init__(self, lv_chs=(3, 16, 32, 64, 96, 128, 192), n_out=5, n_context=1,
                 search_range=4, bn=False):
        super(PWCFlow, self).__init__()

        self.lv_chs = lv_chs
        self.n_pyramid = len(lv_chs) - 1
        self.n_out = n_out
        self.n_context = n_context
        self.bn = bn

        self.feature_pyramid_extractor = FeaturePyramidExtractor(self.lv_chs, bn=self.bn)
        self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                max_displacement=search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.flow_estimators = []
        self.context_networks = []
        for i, ch in enumerate(self.lv_chs[::-1][:n_out]):
            l = self.n_pyramid - i
            ch_in = (search_range * 2 + 1) ** 2  # corr
            if i > 0:
                ch_in += ch + 2  # corr, x1, up_flow

            f_layer = OpticalFlowEstimator(ch_in, bn=self.bn)
            self.add_module('FlowEstimator(Lv{})'.format(l), f_layer)
            self.flow_estimators.append(f_layer)

            if n_out - i > self.n_context:
                self.context_networks.append(None)
            else:
                c_layer = ContextNetwork(f_layer.final_out + 2, bn=self.bn)
                self.add_module('ContextNetwork(Lv{})'.format(l), c_layer)
                self.context_networks.append(c_layer)

    def init_weights(self):
        for m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward(self, x1_pyramid, x2_pyramid):
        flows = []
        for i, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            if i == 0:
                corr = self.corr(x1, x2)
                feat, flow = self.flow_estimators[i](corr)
            else:
                up_flow = F.interpolate(flow * 2, scale_factor=2,
                                        mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, up_flow)
                corr = self.corr(x1, x2_warp)
                F.leaky_relu_(corr)

                flow_feat = [corr, x1, up_flow]
                feat, flow = self.flow_estimators[i](torch.cat(flow_feat, dim=1))

                flow = flow + up_flow

                if self.context_networks[i]:
                    flow_fine = self.context_networks[i](torch.cat([flow, feat], dim=1))
                    flow = flow + flow_fine

            flows.append(flow)
            if len(flows) == self.n_out:
                break
        flows = [F.interpolate(flow * 4, scale_factor=4,
                               mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward(self, x, with_bk=False):

        im1 = x[:, :3]
        im2 = x[:, 3:]

        x1_pyramid = self.feature_pyramid_extractor(im1) + [im1]
        x2_pyramid = self.feature_pyramid_extractor(im2) + [im2]

        flows_f = self._forward(x1_pyramid, x2_pyramid)
        if with_bk:
            flows_b = self._forward(x2_pyramid, x1_pyramid)
            flows = [torch.cat((flow_f, flow_b), 1)
                     for flow_f, flow_b in zip(flows_f, flows_b)]
            return flows
        else:
            return flows_f


class PWCStereo(nn.Module):
    def __init__(self, lv_chs=(3, 16, 32, 64, 96, 128, 192), n_out=5, n_context=1,
                 search_range=4, bn=False):
        super(PWCStereo, self).__init__()

        self.lv_chs = lv_chs
        self.n_pyramid = len(lv_chs) - 1
        self.n_out = n_out
        self.n_context = n_context
        self.bn = bn

        self.feature_pyramid_extractor = FeaturePyramidExtractor(self.lv_chs, bn=self.bn)
        self.corr = Correlation(pad_size=search_range, kernel_size=1,
                                max_displacement=search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.flow_estimators = []
        self.context_networks = []
        for i, ch in enumerate(self.lv_chs[::-1][:n_out]):
            l = self.n_pyramid - i
            ch_in = (search_range * 2 + 1) ** 2  # corr
            if i > 0:
                ch_in += ch + 1  # corr, x1, up_flow

            f_layer = OpticalFlowEstimator(ch_in, ch_out=1, bn=self.bn)
            self.add_module('FlowEstimator(Lv{})'.format(l), f_layer)
            self.flow_estimators.append(f_layer)

            if n_out - i > self.n_context:
                self.context_networks.append(None)
            else:
                c_layer = ContextNetwork(f_layer.final_out + 1, ch_out=1, bn=self.bn)
                self.add_module('ContextNetwork(Lv{})'.format(l), c_layer)
                self.context_networks.append(c_layer)

    def init_weights(self):
        for m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward(self, x1_pyramid, x2_pyramid, neg=False):
        flows = []
        for i, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            if i == 0:
                corr = self.corr(x1, x2)
                feat, flow = self.flow_estimators[i](corr)
                if neg:
                    flow = -F.relu(-flow)
                else:
                    flow = F.relu(flow)
            else:
                # predict the normalized disparity to keep consistent with MonoDepth
                # for reusing the hyper-parameters
                up_flow = F.interpolate(flow, scale_factor=2,
                                        mode='bilinear', align_corners=True)

                zeros = torch.zeros_like(up_flow)
                x2_warp = flow_warp(x2, torch.cat([up_flow, zeros], dim=1),)

                corr = self.corr(x1, x2_warp)
                F.leaky_relu_(corr)

                feat, flow = self.flow_estimators[i](torch.cat([corr, x1, up_flow], dim=1))

                flow = flow + up_flow

                if neg:
                    flow = -F.relu(-flow)
                else:
                    flow = F.relu(flow)

                if self.context_networks[i]:
                    flow_fine = self.context_networks[i](torch.cat([flow, feat], dim=1))
                    flow = flow + flow_fine

                    if neg:
                        flow = -F.relu(-flow)
                    else:
                        flow = F.relu(flow)

            if neg:
                flows.append(-flow)
            else:
                flows.append(flow)
            if len(flows) == self.n_out:
                break
        flows = [F.interpolate(flow * 4, scale_factor=4,
                               mode='bilinear', align_corners=True) for flow in flows]
        return flows[::-1]

    def forward(self, x):

        im1 = x[:, :3]
        im2 = x[:, 3:]

        x1_pyramid = self.feature_pyramid_extractor(im1) + [im1]
        x2_pyramid = self.feature_pyramid_extractor(im2) + [im2]

        disp_lr = self._forward(x1_pyramid, x2_pyramid, neg=False)
        disp_rl = self._forward(x2_pyramid, x1_pyramid, neg=True)
        disps = [torch.cat((lr + 1e-6, rl + 1e-6), 1) for lr, rl in zip(disp_lr, disp_rl)]
        return disps