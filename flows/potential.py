import torch
import torch.nn as nn

class DeepSet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(DeepSet, self).__init__()
        self.feature_extractor = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, hidden_channels))

        self.regressor = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, out_channels))
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, 'reset_parameters', None)
            if callable(reset_op):
                reset_op()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.sum(dim=-2)
        x = self.regressor(x)
        return x
    
class TimeNetwork(nn.Module):
    def __init__(self, func):
        super(TimeNetwork, self).__init__()
        self.func = func

    def forward(self, t, x):
        if not torch.is_tensor(t):
            t = torch.tensor(t).to(x.dtype)
        t_p = t.expand(x.shape[:-1] + (1,))
        return self.func(torch.cat((x, t_p), -1))

class DeepTimeSet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(DeepTimeSet, self).__init__()
        self.feature_extractor = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, hidden_channels))

        self.regressor = TimeNetwork(nn.Sequential(
                nn.Linear(hidden_channels + 1, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, out_channels)))
        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, 'reset_parameters', None)
            if callable(reset_op):
                reset_op()

    def forward(self, t, x):
        x = self.feature_extractor(x)
        x = x.sum(dim=-2)
        x = self.regressor(t, x)
        return x


class ComplexActivation(nn.Module):

    def __init__(self, activation):
        super(ComplexActivation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(torch.real(x)) + 1j*self.activation(torch.imag(x))


class ComplexDeepTimeSet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(ComplexDeepTimeSet, self).__init__()
        self.feature_extractor = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                ComplexActivation(nn.Tanh()),
                nn.Linear(hidden_channels, hidden_channels)).to(torch.complex64)

        self.regressor = TimeNetwork(nn.Sequential(
                nn.Linear(hidden_channels + 1, hidden_channels),
                ComplexActivation(nn.Tanh()),
                nn.Linear(hidden_channels, out_channels))).to(torch.complex64)

        self.add_module('0', self.feature_extractor)
        self.add_module('1', self.regressor)

    def reset_parameters(self):
        for module in self.children():
            reset_op = getattr(module, 'reset_parameters', None)
            if callable(reset_op):
                reset_op()

    def forward(self, t, x):
        x = self.feature_extractor(x)
        x = x.sum(dim=-2)
        x = self.regressor(t, x)
        return x
