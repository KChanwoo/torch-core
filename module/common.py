from torch import nn


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        outputs = inputs
        for module in self._modules.values():
            outputs = module(*outputs)

        return outputs


class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation: nn.Module = None):
        super().__init__()
        model = [nn.Linear(in_features, out_features, bias=bias)]

        if activation is not None:
            model.append(activation)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
