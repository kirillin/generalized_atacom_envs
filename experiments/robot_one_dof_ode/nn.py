import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()
        
        n_features = n_features[0]

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h22 = nn.Linear(n_features, n_features)
        self._h222 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h22.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h222.weight,
                                gain=nn.init.calculate_gain('relu'))

        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        features1 = F.relu(self._h1(state_action))
        features2 = F.relu(self._h2(features1))

        features2 = F.relu(self._h22(features2))
        features2 = F.relu(self._h222(features2))

        q = self._h3(features2)

        return torch.squeeze(q)


class SimpleActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(SimpleActorNetwork, self).__init__()

        n_features = n_features[0]
        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        features1 = F.relu(self._h1(torch.squeeze(state, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)

        return a



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
        nn.init.xavier_uniform_(layers[-1].weight, gain=nn.init.calculate_gain('linear'))

        self._layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, state):
        a = self.mlp(state.float())
        return a

