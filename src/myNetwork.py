import torch
import torch.nn as nn
import torch.nn.functional as F

class network(nn.Module):

    def __init__(self, numclass, feature_extractor, input_shape=(1, 3, 32, 32)):
        super(network, self).__init__()
        self.feature = feature_extractor
        device = next(self.feature.parameters()).device if any((True for _ in self.feature.parameters())) else torch.device('cpu')
        was_training = self.feature.training
        self.feature.eval()
        with torch.no_grad():
            dummy = torch.zeros(*input_shape, device=device)
            out = self.feature(dummy)
            if out.ndim > 2:
                out = torch.flatten(out, 1)
            self.feat_dim = out.size(1)
        if was_training:
            self.feature.train()
        self.fc = nn.Linear(self.feat_dim, numclass, bias=True)

    def forward(self, x, return_features=False, return_maps=False):
        if return_maps and hasattr(self.feature, 'forward_with_maps'):
            feats, maps = self.feature.forward_with_maps(x)
        else:
            feats = self.feature(x)
            maps = None
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        try:
            logits = self.fc(feats)
        except RuntimeError as exc:
            if 'CUBLAS_STATUS_EXECUTION_FAILED' not in str(exc):
                raise
            feats_retry = feats.contiguous().float()
            weight_retry = self.fc.weight.contiguous().float()
            bias_retry = None if self.fc.bias is None else self.fc.bias.contiguous().float()
            logits = F.linear(feats_retry, weight_retry, bias_retry)
            if feats.dtype != logits.dtype:
                logits = logits.to(feats.dtype)
        if return_maps and return_features:
            return (feats, logits, maps)
        if return_features:
            return (feats, logits)
        if return_maps:
            return (logits, maps)
        return logits

    def Incremental_learning(self, numclass):
        weight = self.fc.weight.data.clone()
        bias = self.fc.bias.data.clone()
        in_feature = self.fc.in_features
        out_feature = self.fc.out_features
        self.fc = nn.Linear(in_feature, numclass, bias=True)
        self.fc.weight.data[:out_feature] = weight
        self.fc.bias.data[:out_feature] = bias

    def feature_extractor(self, inputs):
        feats = self.feature(inputs)
        return feats

    def predict(self, fea_input):
        return self.fc(fea_input)

class LeNet(nn.Module):

    def __init__(self, channel=3, hideen=768, num_classes=10):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(nn.Conv2d(channel, 12, kernel_size=5, padding=5 // 2, stride=2), act(), nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2), act(), nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1), act())
        self.fc = nn.Sequential(nn.Linear(hideen, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def weights_init(m):
    try:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
    try:
        if hasattr(m, 'bias'):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        pass
