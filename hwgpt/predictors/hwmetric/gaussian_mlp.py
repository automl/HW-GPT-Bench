import torch
import torch.nn as nn
import torch.optim as optim


class GaussianNN(nn.Module):
    def __init__(self, num_layers: int):
        super(GaussianNN, self).__init__()
        self.in_features = 3 + 2 * num_layers
        self.intermediate_features = 256
        self.fc1 = nn.Linear(
            in_features=self.in_features, out_features=self.intermediate_features
        )
        self.fc2 = nn.Linear(self.intermediate_features, self.intermediate_features)
        self.fc_mean = nn.Linear(self.intermediate_features, 1)
        self.fc_logvar = nn.Linear(self.intermediate_features, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = torch.abs(self.fc_mean(x))
        logvar = self.fc_logvar(x)
        return mean, logvar


def gaussian_nll_loss(mean, logvar, target):
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    return nll.mean()


# Assuming data is loaded into variables X (inputs) and y (targets)
"""model = GaussianNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    mean, logvar = model(X)
    loss = gaussian_nll_loss(mean, logvar, y)
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch}, Loss: {loss.item()}')"""
