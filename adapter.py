import torch.nn as nn
import torch.nn.functional as F

class AdapterModule(nn.Module):
    def __init__(self,
                 in_feature,
                 bottleneck
    ):
        super().__init__()

        self.proj_down = nn.Linear(in_features=in_feature, out_features=bottleneck)
        self.proj_up = nn.Linear(in_features=bottleneck, out_features=in_feature)

    def forward(self, x):
        input = x.clone()

        x = self.proj_down(x)
        x = F.relu(x)
        return self.proj_up(x) + input # Skip Connection
