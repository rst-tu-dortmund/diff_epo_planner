from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PredictionOutput:
    """Stores results of a forward pass of a model that returns multi-modal
    predictions with assigned probabilities.

    :param prediction: predicted trajectories of each agent and each mode. shape=[modes, time_steps, batch_size, agents,
        states].
    :param prediction_prob: predicted probabilities of each trajectory in prediction. shape=[batch_size, agents, modes]
    """

    prediction: torch.Tensor
    prediction_prob: torch.Tensor


@dataclass
class LSTMOutput:
    """Additional output of an LSTM Decoder.

    :param hidden_state: hidden state of the LSTM
    :param cell_state: cell state of the LSTM
    """

    hidden_state: torch.Tensor
    cell_state: torch.Tensor


@dataclass
class AttentionPredictionOutput(PredictionOutput):
    """Prediction output of model with map attention.

    :param interaction: output of module returning interactions between agents
    :param attention_mask: attention mask between nodes
    """

    attention_mask: torch.Tensor
    interaction: Optional[torch.Tensor]


@dataclass
class LSTMAttentionPredictionOutput(AttentionPredictionOutput, LSTMOutput):
    """Prediction output of model with map attention and LSTM outputs."""

    pass


@dataclass
class TwoStageAttentionPredictionOutput(PredictionOutput):
    """Prediction output of two stage attention model.

    :param attention_mask_map: attention mask between agents and lanes
    :param attention_mask_agent: attention mask between agents and agents
    """

    attention_mask_map: torch.Tensor
    attention_mask_agent: torch.Tensor
    interaction: torch.Tensor


@dataclass
class LSTMTwoStageAttentionPredictionOutput(
    TwoStageAttentionPredictionOutput,
    LSTMOutput,
):
    """Prediction output of two stage attention model with LSTM outputs."""

    pass


@dataclass
class LSTMPredictionOutput(PredictionOutput, LSTMOutput):
    """Prediction output of model with LSTM outputs."""

    pass


