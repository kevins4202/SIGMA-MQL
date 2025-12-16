"""
Meta-Q Learning wrapper for SIGMA
Integrates meta-learning components (second Q-network, context encoder, covariate shift correction)
while preserving SIGMA's architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression as LogisticReg
from copy import deepcopy
import configs
from model import Network


class ContextEncoder(nn.Module):
    """
    GRU-based context encoder for task-level context
    Separate from SIGMA's temporal GRU (which processes within-episode sequences)

    Based on meta-q-learning/models/networks.py Context class
    """
    def __init__(
        self,
        input_dim,  # state_dim + action_dim + reward_dim (1)
        hidden_size=configs.hidden_dim // 2,  # 128 by default
        device='cuda'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.device = device

        # GRU for encoding trajectory history
        # Input: concatenated [observation, action, reward] sequences
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )

    def init_hidden(self, batch_size):
        """Initialize hidden state for GRU"""
        return torch.zeros(1, batch_size, self.hidden_size).to(self.device)

    def forward(self, trajectory_data):
        """
        Encode task context from trajectory data

        Args:
            trajectory_data: tuple of (observations, actions, rewards)
                - observations: (batch_size, seq_len, obs_dim)
                - actions: (batch_size, seq_len, 1) - discrete action indices
                - rewards: (batch_size, seq_len, 1)

        Returns:
            context: (batch_size, hidden_size) - encoded task context
        """
        observations, actions, rewards = trajectory_data
        batch_size = observations.size(0)

        # Concatenate obs, action, reward along feature dim
        # actions are discrete indices, convert to one-hot
        if actions.dtype == torch.long:
            actions_onehot = F.one_hot(actions.squeeze(-1), num_classes=5).float()  # 5 actions in SIGMA
        else:
            actions_onehot = actions

        # Concat: [batch_size, seq_len, obs_dim + 5 + 1]
        trajectory = torch.cat([observations, actions_onehot, rewards], dim=-1)

        # Initialize hidden state
        hidden = self.init_hidden(batch_size)

        # Run GRU
        output, hidden = self.gru(trajectory, hidden)

        # Return final hidden state as context
        context = hidden.squeeze(0)  # (batch_size, hidden_size)

        return context


class CovariateShiftCorrector(nn.Module):
    """
    Covariate Shift Correction using logistic regression
    Estimates importance weights to correct for distribution shift between tasks

    Based on meta-q-learning/algs/MQL/mql.py train_cs and get_propensity methods
    """
    def __init__(
        self,
        lam_csc=0.1,  # Logistic regression regularization (C = lam_csc)
        beta_clip=1.2,  # Clip importance weights
        max_iter=2000,
        use_normalized_beta=True,
        device='cuda'
    ):
        super().__init__()
        self.lam_csc = lam_csc
        self.beta_clip = beta_clip
        self.max_iter = max_iter
        self.use_normalized_beta = use_normalized_beta
        self.device = device
        self.r_eps = np.float32(1e-7)  # Numerical stability

        self.logistic_model = None

    def train_discriminator(
        self,
        current_task_contexts,
        other_tasks_contexts
    ):
        """
        Train logistic regression to discriminate current task from other tasks

        Args:
            current_task_contexts: (n_current, context_dim) - contexts from current task
            other_tasks_contexts: (n_other, context_dim) - contexts from other tasks

        Returns:
            train_accuracy: accuracy on training data
            csc_info: tuple of (n_current, n_other, accuracy)
        """
        # Convert to numpy
        pos_contexts = current_task_contexts.cpu().detach().numpy()
        neg_contexts = other_tasks_contexts.cpu().detach().numpy()

        # Create labels: -1 for current task, +1 for other tasks
        # (following meta-q-learning convention)
        X = np.concatenate([pos_contexts, neg_contexts], axis=0)
        y = np.concatenate([
            -np.ones(pos_contexts.shape[0]),
            np.ones(neg_contexts.shape[0])
        ])

        # Train logistic regression
        self.logistic_model = LogisticReg(
            solver='lbfgs',
            max_iter=self.max_iter,
            C=self.lam_csc
        ).fit(X, y)

        # Get training accuracy
        train_accuracy = self.logistic_model.score(X, y)

        csc_info = (pos_contexts.shape[0], neg_contexts.shape[0], train_accuracy)

        return train_accuracy, csc_info

    def get_importance_weights(self, contexts):
        """
        Compute importance weights (beta scores) for given contexts

        Args:
            contexts: (batch_size, context_dim)

        Returns:
            beta_scores: (batch_size, 1) - importance weights
        """
        if self.logistic_model is None:
            # No discriminator trained, return uniform weights
            return torch.ones(contexts.size(0), 1).to(self.device)

        # Convert to numpy for sklearn
        contexts_np = contexts.cpu().detach().numpy()

        # Get decision function value: f(x) = w^T x + b
        f_prop = np.dot(contexts_np, self.logistic_model.coef_.T) + self.logistic_model.intercept_

        # Convert back to torch
        f_prop = torch.from_numpy(f_prop).float().to(self.device)

        # Clip for stability
        f_prop = f_prop.clamp(min=-self.beta_clip)

        # Compute importance weight: exp(-f(x))
        beta_scores = torch.exp(-f_prop)
        beta_scores[beta_scores < 0.1] = 0  # Numerical stability

        # Normalize using logistic regression probabilities
        if self.use_normalized_beta:
            lr_prob = self.logistic_model.predict_proba(contexts_np)[:, 0]  # P(y=-1|x)
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))

            if d_pmax_pmin > self.r_eps:
                beta_scores = (
                    d_pmax_pmin * (beta_scores - torch.min(beta_scores)) /
                    (torch.max(beta_scores) - torch.min(beta_scores) + self.r_eps) +
                    np.float32(np.min(lr_prob))
                )

        return beta_scores.view(-1, 1)


class MetaNetwork(nn.Module):
    """
    Meta-Q Learning wrapper for SIGMA Network

    Architecture:
        - Fast Network: Quick adaptation to current task (inner loop)
        - Meta Network: Slow meta-learning across tasks (outer loop)
        - Context Encoder: Encodes task-level context from trajectories
        - Covariate Shift Corrector: Importance weighting for distribution shift

    Design follows Option B: Wrapper architecture that preserves SIGMA's Network intact
    """
    def __init__(
        self,
        input_shape=configs.obs_shape,
        cnn_channel=configs.cnn_channel,
        hidden_dim=configs.hidden_dim,
        max_comm_agents=configs.max_comm_agents,
        context_hidden_size=128,
        lam_csc=0.1,
        beta_clip=1.2,
        device='cuda'
    ):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim

        # Two Q-networks with shared architecture but separate parameters
        # Fast network: adapts quickly to current task
        self.fast_network = Network(
            input_shape=input_shape,
            cnn_channel=cnn_channel,
            hidden_dim=hidden_dim,
            max_comm_agents=max_comm_agents
        )

        # Meta network: learns across tasks (provides initialization)
        self.meta_network = Network(
            input_shape=input_shape,
            cnn_channel=cnn_channel,
            hidden_dim=hidden_dim,
            max_comm_agents=max_comm_agents
        )

        # Initialize meta network with same weights as fast network
        self.meta_network.load_state_dict(self.fast_network.state_dict())

        # Context encoder: separate from temporal GRU in Network
        # Encodes task-level context from trajectory history
        # Input dim: flattened obs (latent_dim=784) + action_onehot (5) + reward (1)
        latent_dim = 16 * 7 * 7  # SIGMA's latent dimension
        context_input_dim = latent_dim + 5 + 1  # obs + action + reward

        self.context_encoder = ContextEncoder(
            input_dim=context_input_dim,
            hidden_size=context_hidden_size,
            device=device
        )

        # Covariate shift correction
        self.shift_corrector = CovariateShiftCorrector(
            lam_csc=lam_csc,
            beta_clip=beta_clip,
            device=device
        )

        # Flag to determine which network to use
        self.use_fast_network = True

    def switch_to_fast(self):
        """Use fast network for adaptation"""
        self.use_fast_network = True

    def switch_to_meta(self):
        """Use meta network for meta-training"""
        self.use_fast_network = False

    def get_active_network(self):
        """Get currently active network"""
        return self.fast_network if self.use_fast_network else self.meta_network

    def forward(self, obs, context=None, **kwargs):
        """
        Forward pass through active network

        Args:
            obs: observations
            context: optional task context vector
            **kwargs: additional arguments for Network.forward

        Returns:
            output from Network (Q-values, etc.)
        """
        active_network = self.get_active_network()

        # If context is provided, we could condition the network on it
        # For now, pass through to underlying SIGMA network
        return active_network(obs, **kwargs)

    def step(self, obs, greedy=False, **kwargs):
        """
        Take a step (action selection) using active network

        Args:
            obs: observations
            greedy: whether to use greedy action selection
            **kwargs: additional arguments for Network.step

        Returns:
            actions, hidden states, etc.
        """
        active_network = self.get_active_network()
        return active_network.step(obs, greedy=greedy, **kwargs)

    def compute_task_context(self, trajectory_data):
        """
        Compute task context from trajectory data

        Args:
            trajectory_data: tuple of (observations, actions, rewards)

        Returns:
            context: (batch_size, context_hidden_size)
        """
        return self.context_encoder(trajectory_data)

    def train_covariate_shift_correction(
        self,
        current_task_trajectories,
        other_tasks_trajectories
    ):
        """
        Train covariate shift corrector

        Args:
            current_task_trajectories: trajectory data from current task
            other_tasks_trajectories: trajectory data from other tasks

        Returns:
            train_accuracy: accuracy of discriminator
            csc_info: tuple of (n_current, n_other, accuracy)
        """
        # Encode contexts
        with torch.no_grad():
            current_contexts = self.context_encoder(current_task_trajectories)
            other_contexts = self.context_encoder(other_tasks_trajectories)

        # Train discriminator
        return self.shift_corrector.train_discriminator(
            current_contexts,
            other_contexts
        )

    def get_importance_weights(self, trajectory_data):
        """
        Get importance weights for trajectory data

        Args:
            trajectory_data: tuple of (observations, actions, rewards)

        Returns:
            beta_scores: (batch_size, 1) importance weights
        """
        # Encode context
        with torch.no_grad():
            contexts = self.context_encoder(trajectory_data)

        # Get importance weights
        return self.shift_corrector.get_importance_weights(contexts)

    def sync_fast_to_meta(self):
        """Copy fast network weights to meta network"""
        self.meta_network.load_state_dict(self.fast_network.state_dict())

    def sync_meta_to_fast(self):
        """Copy meta network weights to fast network (initialization)"""
        self.fast_network.load_state_dict(self.meta_network.state_dict())

    def save_checkpoint(self):
        """Save current state for rollback"""
        self.checkpoint = {
            'fast_network': deepcopy(self.fast_network.state_dict()),
            'meta_network': deepcopy(self.meta_network.state_dict()),
            'context_encoder': deepcopy(self.context_encoder.state_dict()),
        }

    def load_checkpoint(self):
        """Restore from checkpoint"""
        if hasattr(self, 'checkpoint'):
            self.fast_network.load_state_dict(self.checkpoint['fast_network'])
            self.meta_network.load_state_dict(self.checkpoint['meta_network'])
            self.context_encoder.load_state_dict(self.checkpoint['context_encoder'])

    def eval(self):
        """Set all networks to eval mode"""
        super().eval()
        self.fast_network.eval()
        self.meta_network.eval()
        self.context_encoder.eval()

    def train(self, mode=True):
        """Set all networks to train mode"""
        super().train(mode)
        self.fast_network.train(mode)
        self.meta_network.train(mode)
        self.context_encoder.train(mode)
