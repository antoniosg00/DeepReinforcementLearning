from typing import Callable, Tuple
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
from torch import nn

import gymnasium as gym
from gymnasium import spaces

# https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        ''' 
        history_subspace.shape = (n_features, n_obs)
        other_info_subspace.shape = (n_info, )
        '''
        # Store observation_space as an attribute
        self.observation_space = observation_space

        (history_key, history_subspace), (other_info_key, other_info_subspace) = self.observation_space.spaces.items()
        cnn_input_features = history_subspace.shape[0]
        length_of_sequence = history_subspace.shape[1]

        # We define the extractor for the subspace 'history' (1D CNN)
        # In Pytorch the dimensions are taken: (batch_size, channels, height, width) -> in our case there is no width
        history_extractor = nn.Sequential(
            nn.Conv1d(in_channels=cnn_input_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Default stride = kernel_size

            nn.Flatten(),

            nn.Linear(64 * (length_of_sequence // 4), 128),  # Adjustment of the input size according to the CNN output
            nn.ReLU(),
            nn.Dropout(0.5)  # Add dropout for regularization if needed
        )
        for layer in history_extractor: # Xavier inicialization
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

        # We define the extractor for the subspace 'other_info' (MLP)
        info_extractor = nn.Sequential( # The input will have dimension approx. 3. 32 neurons should be enough (2 layers in addition).
            nn.Linear(other_info_subspace.shape[0], 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        for layer in info_extractor: # Xavier inicialization
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

        self.extractors = nn.ModuleDict({
            "history": history_extractor,
            "other_info": info_extractor
        })

        # Update the features dim manually
        self._features_dim = self._calculate_features_dim()

    def _calculate_features_dim(self):
        # Calculate the total concatenated size of all extractors
        total_concat_size = 0
        for key, extractor in self.extractors.items():
            example_input = th.zeros(1, *self.observation_space[key].shape, dtype=th.float32)
            total_concat_size += extractor(example_input).view(1, -1).size(1)
        return total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
    

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 21,
        last_layer_dim_vf: int = 21,
    ):
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_pi),
            nn.Softmax(-1) # The PPO itself automatically samples over the obtained distribution and passes it as an action.
        )
        for layer in self.policy_net: # Xavier inicialization
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, last_layer_dim_vf),
        )
        for layer in self.value_net: # Xavier inicialization
            if isinstance(layer, nn.Linear): nn.init.xavier_uniform_(layer.weight)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        a, v = self.forward_actor(features), self.forward_critic(features)
        return a, v

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)