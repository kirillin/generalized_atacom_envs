import torch
import torch.nn as nn


class SACCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.topology = [n_input] + n_features + [n_output]
        layers = []
        for i in range(len(self.topology) - 2):
            layers.append(nn.Linear(self.topology[i], self.topology[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.topology[-2], self.topology[-1]))
        nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        q = self.mlp(state_action)
        return torch.squeeze(q)


class SACActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SACActorNetwork, self).__init__()

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self.topology = [n_input] + n_features + [n_output]
        layers = []
        for i in range(len(self.topology) - 2):
            layers.append(nn.Linear(self.topology[i], self.topology[i + 1]))
            nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.topology[-2], self.topology[-1]))
        nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('relu'))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, state):
        a = self.mlp(state.float())
        return a
