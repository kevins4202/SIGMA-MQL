import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression as LogisticReg
from copy import deepcopy
import configs
from model import Network


class ContextEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_size=128,
        device='cuda'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.device = device

        # GRU for encoding trajectory history
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

        # Convert discrete actions to one-hot
        if actions.dtype == torch.long:
            actions_onehot = F.one_hot(
                actions.squeeze(-1),
                num_classes=self.action_dim
            ).float()
        else:
            actions_onehot = actions

        # Concatenate: [batch_size, seq_len, obs_dim + 5 + 1]
        trajectory = torch.cat([observations, actions_onehot, rewards], dim=-1)

        # Initialize and run GRU
        hidden = self.init_hidden(batch_size)
        output, hidden = self.gru(trajectory, hidden)

        # Return final hidden state as context
        context = hidden.squeeze(0)  # (batch_size, hidden_size)
        return context


class CovariateShiftCorrector(nn.Module):
    def __init__(
        self,
        lam_csc=0.1,  # Logistic regression regularization (C = lam_csc)
        beta_clip=1.2,  # Clip importance weights to prevent extreme values
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
        if self.logistic_model is None:
            # No discriminator trained, return uniform weights
            return torch.ones(contexts.size(0), 1).to(self.device)

        # Convert to numpy for sklearn
        contexts_np = contexts.cpu().detach().numpy()

        # Get decision function value: f(x) = w^T x + b
        f_prop = np.dot(contexts_np, self.logistic_model.coef_.T) + self.logistic_model.intercept_
        f_prop = torch.from_numpy(f_prop).float().to(self.device)

        # Clip for stability
        f_prop = f_prop.clamp(min=-self.beta_clip)

        # Compute importance weight: exp(-f(x))
        beta_scores = torch.exp(-f_prop)
        beta_scores[beta_scores < 0.1] = 0  # Numerical stability

        # Normalize using logistic regression probabilities
        if self.use_normalized_beta:
            lr_prob = self.logistic_model.predict_proba(contexts_np)[:, 0]
            d_pmax_pmin = np.float32(np.max(lr_prob) - np.min(lr_prob))

            if d_pmax_pmin > self.r_eps:
                beta_scores = (
                    d_pmax_pmin * (beta_scores - torch.min(beta_scores)) /
                    (torch.max(beta_scores) - torch.min(beta_scores) + self.r_eps) +
                    np.float32(np.min(lr_prob))
                )

        return beta_scores.view(-1, 1)


class MetaNetwork(nn.Module):
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

        self.network = Network(
            input_shape=input_shape,
            cnn_channel=cnn_channel,
            hidden_dim=hidden_dim,
            max_comm_agents=max_comm_agents
        ).to(device)

        latent_dim = self.network.latent_dim
        context_input_dim = latent_dim + configs.action_dim + 1

        self.context_encoder = ContextEncoder(
            input_dim=context_input_dim,
            action_dim=configs.action_dim,
            hidden_size=context_hidden_size,
            device=device
        ).to(device)

        # Covariate shift corrector for handling distribution shift
        self.shift_corrector = CovariateShiftCorrector(
            lam_csc=lam_csc,
            beta_clip=beta_clip,
            device=device
        )

        # Checkpoint storage for meta-learning
        self.checkpoint = None

    def reset(self):
        """Reset SIGMA network's hidden state"""
        self.network.reset()

    def step(self, obs, pos):
        return self.network.step(obs, pos)

    def forward(self, obs, steps, hidden, comm_mask):
        return self.network(obs, steps, hidden, comm_mask)

    def train_covariate_shift_correction(
        self,
        current_task_trajectories,
        other_tasks_trajectories
    ):
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
        # Encode context
        with torch.no_grad():
            contexts = self.context_encoder(trajectory_data)

        # Get importance weights
        return self.shift_corrector.get_importance_weights(contexts)

    def eval(self):
        """Set all components to eval mode"""
        super().eval()
        self.network.eval()
        self.context_encoder.eval()

    def train(self, mode=True):
        """Set all components to train mode"""
        super().train(mode)
        self.network.train(mode)
        self.context_encoder.train(mode)
