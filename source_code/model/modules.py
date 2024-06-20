# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import logging

from abc import ABC
from typing import Tuple, Union, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb


import data.utils.helpers as du
import model.gnn.layers as mgl
import model.multimode_modules as mmm
from data import constants as c
from model.interfaces.outputs import (
    AttentionPredictionOutput,
    LSTMAttentionPredictionOutput,
)
import model.gnn.layers as mgl
import model.multimode_modules as mmm

from data import constants as c
import einops as einops


class ModeProbabilityPredictor(nn.Module):
    def __init__(
            self,
            input_size: int,
            modes_num: int,
            intermediate_neurons_num: int = 16,
    ):
        super(ModeProbabilityPredictor, self).__init__()
        self.mode_prob_predictor = nn.Sequential(
            nn.Linear(input_size, intermediate_neurons_num),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(intermediate_neurons_num, intermediate_neurons_num),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(intermediate_neurons_num, modes_num),
            nn.Softmax(dim=1),  # first dim is batch
        )

    def forward(self, input):
        try:
            probs = self.mode_prob_predictor(input=input)
            logging.debug("-------output probs---------")
            logging.debug(probs.clone().detach().cpu().numpy())
            logging.debug("-----------------")
        except Exception as e:
            message = (
                "Mode probability predictor input corrupted! Please investigate "
                "with crashed_sample.pkl in log dir"
            )
            print(message)
            logging.error(message)
        return probs




class VectorSubGraph(nn.Module):
    def __init__(self, opts: dict, map: bool = False):
        """Encoder utilizing subgraph operations.

        :param opts:
        """
        super(VectorSubGraph, self).__init__()
        self.layer_width = opts["hyperparameters"]["vn_encoder_width"]
        if map:
            self.depth = opts["hyperparameters"]["vn_map_encoder_depth"]
            self.in_features = c.VN_INPUT_SIZE_MAP
        else:
            self.depth = opts["hyperparameters"]["vn_dyn_encoder_depth"]
            self.in_features = c.VN_INPUT_SIZE_DYN
        self.subgraph_layers = nn.ModuleList()
        for i in range(self.depth):
            if i == 0:
                self.subgraph_layers.append(
                    mgl.SubGraphLayer(
                        in_features=self.in_features,
                        hidden_size=self.layer_width,
                    ),
                )
            else:
                self.subgraph_layers.append(
                    mgl.SubGraphLayer(
                        in_features=self.layer_width * 2,
                        hidden_size=self.layer_width,
                    ),
                )

    def forward(
        self,
        input_data: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_data: agent_data of shape(num_of_seqs, batch_size, num_of_seqs, max_seq_size, in_features)
            mask: mask of shape(num_of_seqs, batch_size, max_seq_size, in_features)

        Returns:
            out: out of shape(batch_size, num_of_seqs, hidden_size * 2): polyline level feature
        """

        input_data = input_data.permute(c.ONE, c.TWO, c.ZERO, c.THREE)
        x = input_data
        for layer in self.subgraph_layers:
            x = layer(x, mask)
        out = self.aggregate(x)
        return out

    @staticmethod
    def aggregate(x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: x of shape(batch_size, num_of_seqs, max_seq_size, hidden_size * 2)

        Returns:
            y: y of shape(batch_size, num_of_seqs, hidden_size * 2)
        """
        y, _ = torch.max(x, dim=2)
        return y


class LstmVectorEncoder(nn.Module):
    def __init__(self, opts: dict, in_features: int = c.VN_INPUT_SIZE_DYN):
        """LSTM Encoder for Vector net For consistency of output with subgraph
        operation encoder width is multiplied with 2.

        :param opts: dict with configurations
        :param in_features: number of input features per timestep
        """
        super(LstmVectorEncoder, self).__init__()
        self.pred_length = opts["hyperparameters"]["prediction_length"]  # in seconds
        self.mode_num = opts["hyperparameters"]["num_modes"]
        self.use_embedding = opts["config_params"]["use_embedding"]
        self.layer_width = opts["hyperparameters"]["vn_encoder_width"] * 2
        self.depth = opts["hyperparameters"]["vn_dyn_encoder_depth"]
        self.in_features = in_features
        if self.use_embedding:
            self.embedding_dim = opts["hyperparameters"]["embedding_size"]
            self.spatial_embedding = nn.Linear(self.in_features, self.embedding_dim)
            self.lstm = nn.LSTM(
                input_size=self.embedding_dim,
                hidden_size=self.layer_width,
                num_layers=self.depth,
            )
        else:
            self.lstm = nn.LSTM(
                input_size=self.in_features,
                hidden_size=self.layer_width,
                num_layers=self.depth,
            )

    def forward(
        self,
        input_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_data: agent_data of shape(max_seq_size, batch_size, num_of_seqs,
            in_features)

        Returns:
            out: out of shape(batch_size, num_of_seqs, hidden_size):
            polyline level feature
        """
        batch_size = input_data.shape[c.ONE]
        num_agents = input_data.shape[c.TWO]
        input_data = du.reduce_batch_dim(input_data)
        if self.use_embedding:
            input_data = self.spatial_embedding(input_data)

        _, (hidden_state, cell_state) = self.lstm(input_data)
        inner_final_h = hidden_state[-1, ...].reshape(batch_size, num_agents, -1)
        return inner_final_h, hidden_state, cell_state


class LinearVectorEncoder(nn.Module):
    """easy linear encoder creating embeddings for each input node."""

    def __init__(
        self,
        opts: dict,
    ):
        super(LinearVectorEncoder, self).__init__()
        self.layer_width = opts["hyperparameters"]["vn_encoder_width"] * 2
        self.linear = nn.Linear(
            in_features=c.POINT_PER_SUBGRAPH * c.VN_INPUT_SIZE_MAP,
            out_features=self.layer_width,
        )

    def forward(self, input_data: torch.Tensor):
        input_data = input_data.permute(c.ONE, c.TWO, c.ZERO, c.THREE)
        input_data = input_data.flatten(start_dim=c.TWO)
        output = self.linear(input_data)
        return output


class VectorGlobalGraph(nn.Module):
    """models inteaction between query and ke nodes by application of attention
    mechanism:

    - multihead-attention
    - self-attention
    """

    def __init__(
        self,
        opts: dict,
        in_features: int = None,
        in_features_keys: int = None,
    ):
        super(VectorGlobalGraph, self).__init__()
        self.opts = opts
        self.in_features = in_features
        if in_features_keys is None:
            self.in_features_keys = self.in_features
        else:
            self.in_features_keys = in_features_keys
        self.out_features = self.opts["hyperparameters"]["vn_global_graph_width"]
        self.depth = self.opts["hyperparameters"]["vn_global_graph_depth"]
        self.attention_layers = nn.ModuleList()
        if is_multihead(self.opts):
            self.head_number = self.opts["hyperparameters"]["head_number"]
            for i in range(self.depth):
                self.attention_layers.append(
                    torch.nn.MultiheadAttention(
                        embed_dim=self.in_features,
                        num_heads=self.head_number,
                    ),
                )
        else:
            for i in range(self.depth):
                if i == 0:
                    self.attention_layers.append(
                        mgl.SelfAttentionLayer(
                            self.in_features,
                            self.out_features,
                            self.in_features_keys,
                        ),
                    )
                else:
                    self.attention_layers.append(
                        mgl.SelfAttentionLayer(
                            self.out_features,
                            self.out_features,
                            self.in_features_keys,
                        ),
                    )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple:
        masks = []
        for layer in self.attention_layers:
            if is_multihead(self.opts):
                query, attention_mask = layer(query, key, key)
            else:
                query, attention_mask = layer(query, key)
            masks.append(attention_mask)
        attention_mask = torch.mean(torch.stack(masks), dim=c.FIRST)
        return query, attention_mask


class VectorDecoder(nn.Module):
    def __init__(self, opts: dict, out_params: int):
        super(VectorDecoder, self).__init__()
        self.out_params = out_params
        (
            in_features,
            self.nbr_decoder_layers,
            self.decoder_width,
        ) = self.get_decoder_config(opts)
        self.layers = self.create_decoder_layers(opts, out_params)

    @staticmethod
    def get_norm_layer(opts: dict) -> Union[Type[nn.LayerNorm], Type[nn.BatchNorm1d]]:
        if opts["hyperparameters"]["norm_layer"] == "layer":
            norm_layer = nn.LayerNorm
        elif opts["hyperparameters"]["norm_layer"] == "batch":
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError(
                "norm_layer is {} but so far only batch and layer "
                "is implemented".format(opts["hyperparameters"]["norm_layer"]),
            )
        return norm_layer

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_shape = input.shape
        input = input.reshape((input_shape[0] * input_shape[1], -1))
        out = self.layers(input)
        out = out.reshape(input_shape[0], input_shape[1], -1)
        return out

    @staticmethod
    def get_decoder_config(opts: dict) -> Tuple[int, int, int]:
        """Encapsulate some of VectorDecoder's configs in one method to
        facilitate comparison of decoders."""
        if opts["config_params"]["multihead_attention"]:
            in_features = opts["hyperparameters"]["vn_encoder_width"] * 2
        else:
            in_features = opts["hyperparameters"]["vn_global_graph_width"]
        nbr_decoder_layers = opts["hyperparameters"]["vn_decoder_depth"]
        decoder_width = opts["hyperparameters"]["vn_decoder_width"]
        return in_features, nbr_decoder_layers, decoder_width

    @classmethod
    def create_decoder_layers(cls, opts: dict, out_params: int) -> nn.Sequential:
        """Encapsulate the creation of VectorDecoder's layers in one method to
        facilitate comparison of decoders."""
        (
            in_features,
            nbr_decoder_layers,
            decoder_width,
        ) = cls.get_decoder_config(opts)

        norm_layer = cls.get_norm_layer(opts)
        layers = []
        for _ in range(nbr_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=decoder_width),
                    norm_layer(decoder_width),
                    nn.ReLU(True),
                ],
            )
            in_features = decoder_width
        layers.append(nn.Linear(in_features=in_features, out_features=out_params))
        layers = nn.Sequential(*layers)
        return layers


class VectorBase(ABC, nn.Module):
    """Abstract class for Vector modules used as net."""

    def __init__(self, opts: dict, device: torch.device):
        super().__init__()
        self.opts = opts
        self.device = device
        self.fps = du.get_fps_for_dataset(opts["general"]["dataset"])


class VNet(VectorBase):
    """VectorNet base module."""

    def __init__(self, opts: dict, device: torch.device):
        """
        :param opts: options from config.json
        :param device: device of module
        """
        super().__init__(opts, device)
        prediction_length = self.opts["hyperparameters"]["prediction_length"]
        self.mode_num = self.opts["hyperparameters"]["num_modes"]
        self.num_predicted_samples = du.int_with_check(prediction_length * self.fps)
        self.nbr_out_params = du.get_output_format(opts)
        out_params = self.num_predicted_samples * self.nbr_out_params * self.mode_num
        self.out_params = du.int_with_check(out_params)
        self.map_enc = None
        self.dynamic_enc = None
        self.interactionModule = None
        self.decoder = None
        self.set_map_encoder()
        self.set_dyn_encoder()
        self.set_decoder()
        self.set_prob_predictor()
        self.set_interaction_module()
        self.inter_out = None
        self.probs = None

    def set_prob_predictor(self):
        """
        assign and configure mode probability predictor module based on config file
        :return:
        """
        if self.mode_num > 1 and (not self.opts["config_params"]["no_probs"]):
            if is_multihead(self.opts):
                prob_in_params = (
                        self.opts["hyperparameters"]["vn_encoder_width"] * 2
                        + self.out_params
                )
            else:
                prob_in_params = (
                        self.opts["hyperparameters"]["vn_global_graph_width"]
                        + self.out_params
                )
            self.mode_prob_predictor = ModeProbabilityPredictor(
                input_size=prob_in_params,
                modes_num=self.mode_num,
            ).to(self.device)

    def set_decoder(self):
        """assign and configure decoder module based on config file."""
        decoder = VectorDecoder(self.opts, self.out_params)
        self.decoder = decoder.to(self.device)

    def set_map_encoder(self):
        """
        assign and configure map encoder module based on config file
        :return:
        """
        if self.opts["config_params"]["vn_map_encoder_type"] == c.SUB_GRAPH_MAP:
            self.map_enc = VectorSubGraph(self.opts, map=True).to(self.device)
        elif self.opts["config_params"]["vn_map_encoder_type"] == c.LINEAR_MAP:
            self.map_enc = LinearVectorEncoder(self.opts).to(self.device)
        else:
            raise ValueError(
                "vn_map_encoder_type {} not defined".format(
                    self.opts["config_params"]["vn_map_encoder_type"],
                ),
            )

    def set_dyn_encoder(self):
        """
        assign and configure dynamic encoder module based on config file
        :return:
        """
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.SUB_GRAPH_DYN:
            self.dynamic_enc = VectorSubGraph(self.opts).to(self.device)
        elif self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            self.dynamic_enc = LstmVectorEncoder(self.opts).to(self.device)
        elif is_transformer_encoder(self.opts):
            self.dynamic_enc = TransformerVectorEncoder(self.opts).to(self.device)
        else:
            raise ValueError(
                "vn_dyn_encoder_type {} not defined".format(
                    self.opts["config_params"]["vn_dyn_encoder_type"],
                ),
            )

    def set_interaction_module(self):
        """
        assign and configure interaction module based on config file
        :return:
        """
        in_features_agent = self.opts["hyperparameters"]["vn_encoder_width"] * 2
        in_features_map = self._in_features_map(self.opts)
        assert in_features_agent == in_features_map
        self.interactionModule = VectorGlobalGraph(
            opts=self.opts,
            in_features=in_features_agent,
        ).to(self.device)

    @staticmethod
    def _in_features_map(opts: dict) -> int:
        """compute number of features per node that are given input into
        interaction module.

        :param opts:
        :return:
        """
        if opts["config_params"]["vn_map_encoder_type"] == c.FIXED_FEATURES:
            features = c.NBR_MAP_FEATURES
        else:
            features = opts["hyperparameters"]["vn_encoder_width"] * 2
        return features

    def _interaction(self, sg_agent, sg_lane) -> Tuple[torch.Tensor, torch.Tensor]:
        sg_agent_lane = torch.cat((sg_agent, sg_lane), dim=c.ONE)  # (b, nodes, e_dim)
        gg_out, attention_mask = self.interactionModule(sg_agent_lane, sg_agent_lane)
        return gg_out, attention_mask

    def _probability(
        self,
        inter_out: torch.Tensor,
        out: torch.Tensor,
        agent_data: torch.Tensor,
        agent_num: int,
    ) -> torch.Tensor:
        """Predict probabilities for each mode.

        :param inter_out: interaction tensor
        :param out: predicted trajectories
        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param agent_num: number of predicted agents
        :return:
        """
        if not self.opts["config_params"]["no_probs"] and self.mode_num > 1:
            prob_head_input = []
            if self.opts["config_params"]["detach_probability_head"]:
                gg_out_clone = inter_out.clone().detach()
            else:
                gg_out_clone = inter_out.clone()
            gg_out_clone = du.reduce_batch_dim(gg_out_clone)
            prob_head_input.append(gg_out_clone)
            prob_head_input.append(
                out.clone().reshape(out.shape[c.FIRST_VALUE], c.LAST_VALUE).detach(),
            )
            prob_head_input = torch.cat(prob_head_input, dim=c.LAST_VALUE)
            probs = self.mode_prob_predictor(prob_head_input)
            probs = probs.reshape(c.MATCH_SHAPE, agent_num, self.mode_num)
        else:
            probs = mmm.get_dummy_prob_distribution(
                sample=agent_data,
                opts=self.opts,
                device=out.device,
                vectors=True,
            )  # (batch, agents, modes)
        return probs

    def _format_output(self, out: torch.Tensor, agent_num: int) -> torch.Tensor:
        """

        :param out:
        :param agent_num:
        :return:
        """
        out = out.permute(c.ONE, c.TWO, c.ZERO, c.THREE)
        out = out.reshape(
            self.mode_num,
            self.num_predicted_samples,
            c.MATCH_SHAPE,
            agent_num,
            self.nbr_out_params,
        )
        return out

    def _encode(
        self,
        agent_data,
        lane_graph,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param lane_graph: map in vectorized format
        :return: subgraph agents, subgraph lanes. for LSTM: hidden states, cell states
        """
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            sg_agent, hidden, cell = self.dynamic_enc(agent_data.to(self.device))
        else:
            sg_agent = self.dynamic_enc(agent_data.to(self.device))
            hidden, cell = torch.tensor([-1]), torch.tensor([-1])
        sg_lane = self.map_enc(lane_graph.to(self.device))

        return sg_agent, sg_lane, hidden, cell

    def _get_targets(
        self,
        inter_out,
        sg_agent,
        targets_indices,
    ) -> Tuple[torch.Tensor, int]:
        """

        :param inter_out: interaction tensor
        :param sg_agent:
        :param targets_indices: list of indices of target agent in each batch.
        :return: interactions,  number of agents
        """
        if self.opts["config_params"]["predict_target"]:
            inter_out = du.get_targets(inter_out, targets_indices).unsqueeze(c.ONE)
            agent_num = 1
        else:
            agent_num = sg_agent.shape[c.SECOND_VALUE]
            inter_out = du.get_agents(inter_out, agent_num)

        return inter_out, agent_num

    def _decode(
        self,
        inter_out: torch.Tensor,
        agent_data: torch.Tensor,
        agent_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode predictions and probabilities from interactions and agent
        data.

        :param inter_out: interaction tensor
        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param agent_num:
        :return: predictions and their predicted probabilities
        """
        predictions = self.decoder(inter_out).reshape(
            c.LAST_VALUE,
            self.mode_num,
            self.num_predicted_samples,
            self.nbr_out_params,
        )

        probs = self._probability(
            inter_out=inter_out,
            out=predictions,
            agent_data=agent_data,
            agent_num=agent_num,
        )

        return predictions, probs

    def _forward(
        self,
        agent_data: torch.Tensor,
        targets_indices: list,
        lane_graph: torch.Tensor,
    ) -> Union[AttentionPredictionOutput, LSTMAttentionPredictionOutput]:
        """forward function computing prediction and attention based on the
        given input.

        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param targets_indices: list of indices of target agent in each batch.
        :param lane_graph: map in vectorized format
        :return:
        """
        sg_agent, sg_lane, hidden, cell = self._encode(agent_data, lane_graph)
        interactions, attention_mask = self._interaction(
            sg_agent=sg_agent,
            sg_lane=sg_lane,
        )
        interactions, agent_num = self._get_targets(
            interactions,
            sg_agent,
            targets_indices,
        )
        predictions, probabilities = self._decode(interactions, agent_data, agent_num)
        predictions = self._format_output(predictions, agent_num)
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            out = LSTMAttentionPredictionOutput(
                prediction=predictions,
                prediction_prob=probabilities,
                attention_mask=attention_mask,
                cell_state=cell,
                hidden_state=hidden,
                interaction=interactions,
            )
        else:
            out = AttentionPredictionOutput(
                prediction=predictions,
                prediction_prob=probabilities,
                attention_mask=attention_mask,
                interaction=interactions,
            )
        return out

    def forward(
        self,
        agent_data: torch.Tensor,
        targets_indices: list,
        lane_graph: torch.Tensor,
    ) -> Union[AttentionPredictionOutput, LSTMAttentionPredictionOutput]:
        """handle function to only return predictions.

        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param targets_indices: list of indices of target agent in each batch. list of indices of target agent in each batch.
        :param lane_graph: map in vectorized format
        :return: prediction output
        """
        out = self._forward(
            agent_data,
            targets_indices,
            lane_graph,
        )
        return out

    def get_attention_mask(
        self,
        agent_data: torch.Tensor,
        targets_indices: list,
        lane_graph: torch.Tensor,
    ) -> torch.Tensor:
        """handles forward pass to return attention mask.

        :param agent_data: observed trajectories in vector format. shape: [time, batch, agents, params]
        :param targets_indices: list of indices of target agent in each batch.
        :param lane_graph: map in vectorized format map in vectorized format
        :return:
        """
        out = self._forward(
            agent_data,
            targets_indices,
            lane_graph,
        )
        return out.attention_mask




def is_multihead(cfgs: dict) -> bool:
    """check if globl graph multihead attention is used as interaction.

    :param cfgs:
    :return:
    """
    return cfgs["config_params"]["interaction"] == c.MHA


def is_self_attention(cfgs: dict) -> bool:
    """check if globl graph self attention is used as interaction.

    :param cfgs:
    :return:
    """
    return cfgs["config_params"]["interaction"] == c.SELF_ATTENTION


def is_transformer_encoder(cfgs: dict, map: bool = False):
    if not map:
        type = cfgs["config_params"]["vn_dyn_encoder_type"]
    else:
        type = cfgs["config_params"]["vn_map_encoder_type"]
    my_check = ((type == c.TRANSFORMER_SUM) or
                (type == c.TRANSFORMER_MAXPOOL) or
                (type == c.TRANSFORMER_LAST))
    return my_check




# ------------------------------------------------------------------------------
# Copyright:    Technical University Dortmund
# Project:      KISSaF
# Created by:   Institute of Control Theory and System Engineering
# ------------------------------------------------------------------------------
class ControlDecoder(nn.Module):
    def __init__(
            self,
            opts: dict,
            out_params: int
    ):
        super(ControlDecoder, self).__init__()
        self.out_params = out_params

        in_features = opts["hyperparameters"]["vn_global_graph_width"]

        self.nbr_decoder_layers = opts["hyperparameters"]["vn_decoder_depth"]
        self.decoder_width = opts["hyperparameters"]["vn_decoder_width"]
        norm_layer = self.get_norm_layer(opts)
        layers = []
        for _ in range(self.nbr_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=self.decoder_width),
                    norm_layer(self.decoder_width),
                    nn.ReLU(True)
                ]
            )
            in_features = self.decoder_width
        layers.append(nn.Linear(in_features=in_features, out_features=self.out_params))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def get_norm_layer(
            opts: dict
    ) -> nn.Module:
        if opts["hyperparameters"]["norm_layer"] == 'layer':
            norm_layer = nn.LayerNorm
        elif opts["hyperparameters"]["norm_layer"] == 'batch':
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError(
                'norm_layer is {} but so far only batch and layer '
                'is implemented'.format(opts["hyperparameters"]["norm_layer"])
            )
        return norm_layer

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        input_shape = input.shape
        input = input.reshape((input_shape[0] * input_shape[1], -1))
        out = self.layers(input)
        out = out.reshape(input_shape[0], input_shape[1], -1)
        return out


class SceneBaseNet(nn.Module):
    # Base class for scene prediction networks such as scene control net and epo f net
    def __init__(
            self,
            opts: dict,
            device: torch.device
    ):
        super(SceneBaseNet, self).__init__()
        self.opts = opts
        self.device = device
        self.fps = du.get_fps_for_dataset(opts["general"]["dataset"])
        prediction_length = self.opts["hyperparameters"]["prediction_length"]
        self.mode_num = self.opts["hyperparameters"]["num_modes"]
        self.target_states = self.opts["differentiable_optimization"]["num_pred_states"]
        self.inter_out = None
        self.num_predicted_samples = du.int_with_check(
            prediction_length * self.fps
        )
        self.nbr_out_params = du.get_output_format(opts)
        out_params = self.num_predicted_samples * self.nbr_out_params * self.mode_num
        self.out_params = du.int_with_check(out_params)

        # Set neural network modules
        self.set_map_encoder()
        self.set_dyn_encoder()
        self.set_scene_prob_predictor()
        self.set_interaction_module()

        # Note: Decoder differ between instances of this class

    def set_map_encoder(
            self
    ):
        """
        assign and configure map encoder module based on config file
        :return:
        """
        if self.opts["config_params"]["vn_map_encoder_type"] == c.SUB_GRAPH_MAP:
            self.map_enc = VectorSubGraph(self.opts, map=True).to(self.device)
        elif self.opts["config_params"]["vn_map_encoder_type"] == c.LINEAR_MAP:
            self.map_enc = LinearVectorEncoder(self.opts).to(self.device)
        else:
            raise ValueError(
                "vn_map_encoder_type {} not defined".format(
                    self.opts["config_params"]["vn_map_encoder_type"]
                )
            )

    def set_dyn_encoder(
            self
    ):
        """
        assign and configure dynamic encoder module based on config file
        """
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.SUB_GRAPH_DYN:
            self.dynamic_enc = VectorSubGraph(self.opts).to(self.device)
        elif self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            self.dynamic_enc = LstmVectorEncoder(self.opts).to(self.device)
        else:
            raise ValueError(
                "vn_dyn_encoder_type {} not defined".format(
                    self.opts["config_params"]["vn_dyn_encoder_type"]
                )
            )


    def set_interaction_module(
            self
    ):
        """
        assign and configure interaction module based on config file
        """
        in_features_agent = self.opts["hyperparameters"]["vn_encoder_width"] * 2
        in_features_map = self._in_features_map(self.opts)
        assert in_features_agent == in_features_map
        self.interactionModule = VectorGlobalGraph(
            opts=self.opts,
            in_features=in_features_agent
        ).to(self.device)

    @staticmethod
    def _in_features_map(
            opts: dict
    ) -> int:
        """
        compute number of features per node that are given input into interaction module
        :param opts:
        :return:
        """
        if opts["config_params"]["vn_map_encoder_type"] == c.FIXED_FEATURES:
            features = c.NBR_MAP_FEATURES
        else:
            features = opts["hyperparameters"]["vn_encoder_width"] * 2
        return features

    def _interaction(
            self,
            sg_agent,
            sg_lane
    ):
        sg_agent_lane = torch.cat(
            (sg_agent, sg_lane),
            dim=c.ONE
        )  # (b, nodes, e_dim)

        gg_out, attention_mask = self.interactionModule(
            sg_agent_lane,
            sg_agent_lane
        )
        return gg_out, attention_mask

    def set_scene_prob_predictor(self):
        """
        assign and configure mode probability predictor module based on config file
        :return:
        """
        prob_in_params = self.opts["differentiable_optimization"]["number_player"] * (self.opts["hyperparameters"]["vn_global_graph_width"] + self.out_params)
        self.scene_prob_predictor = SceneProbabilityPredictor(
            input_size=int(prob_in_params),
            modes_num=self.mode_num,
            opts=self.opts
        ).to(self.device)

    def _format_output(
            self,
            out,
            agent_num
    ):
        out = out.permute(c.ONE, c.TWO, c.ZERO, c.THREE)
        out = out.reshape(
            self.mode_num,
            self.num_predicted_samples,
            c.MATCH_SHAPE,
            agent_num,
            self.nbr_out_params
        )  # (modes, timesteps, batch, agents, params(x,y))
        if self.nbr_out_params == c.OUTPUT_PARAMS_GAUSSIAN:
            out = du.outputActivation(out)
        return out


class SceneControlNet(SceneBaseNet):

    def __init__(
            self,
            opts: dict,
            device: torch.device
    ):
        super(SceneControlNet, self).__init__(opts=opts, device=device)
        self.set_decoder()

    def set_decoder(
            self
    ):
        init_decoder = ControlDecoder(self.opts, self.out_params)
        self.init_decoder = init_decoder.to(self.device)

    def _forward(
            self,
            agent_data: torch.Tensor,
            lane_graph: torch.Tensor
    ) -> tuple:
        """
        forward function computing prediction and attention based on the given input
        :param agent_data: agent histories
        :param lane_graph: hd map
        :return:
        """
        # Encoding
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            sg_agent, hidden, cell = self.dynamic_enc(agent_data.to(self.device))
        else:
            sg_agent = self.dynamic_enc(agent_data.to(self.device))
            hidden, cell = torch.tensor([-1]), torch.tensor([-1])
        sg_lane = self.map_enc(lane_graph.to(self.device))
        # Interaction Modeling
        inter_out, attention_mask = self._interaction(
            sg_agent=sg_agent,
            sg_lane=sg_lane
        )

        agent_num = sg_agent.shape[c.SECOND_VALUE]
        inter_out = du.get_agents(inter_out, agent_num)

        self.inter_out = inter_out

        # Predict multimodal control sequence initializations for all scenarios
        u_input = inter_out.type(torch.float32)
        out = self.init_decoder(u_input)
        out = out.reshape(
            c.LAST_VALUE,
            self.mode_num,
            self.num_predicted_samples,
            self.nbr_out_params
        )
        self.out = out
        out = self._format_output(out, agent_num)

        return out

    def forward(
            self,
            agent_data: torch.Tensor,
            lane_graph: torch.Tensor
    ) -> tuple:
        out = self._forward(
            agent_data,
            lane_graph
        )

        return out


class EPONetF(SceneBaseNet):
    def __init__(
            self,
            opts: dict,
            device: torch.device
    ):
        super(EPONetF, self).__init__(opts=opts, device=device)
        self.num_of_agents = self.opts["differentiable_optimization"]["number_player"]
        self.set_decoder()

    def set_decoder(
            self
    ):
        init_decoder = ControlDecoder(self.opts, self.out_params)
        self.init_decoder = init_decoder.to(self.device)

        if self.opts["differentiable_optimization"]["multimodal_weights"]:
            int_weight_out_params = self.opts["differentiable_optimization"]["num_int_weights"] * self.mode_num
            own_weight_out_params = self.opts["differentiable_optimization"]["num_own_weights"] * self.mode_num
        else:
            int_weight_out_params = self.opts["differentiable_optimization"]["num_int_weights"]
            own_weight_out_params = self.opts["differentiable_optimization"]["num_own_weights"]

        interaction_weight_decoder = GameInteractionWeightDecoder(self.opts, int_weight_out_params)
        self.interaction_weight_decoder = interaction_weight_decoder.to(self.device)

        self_weight_decoder = GameSelfWeightDecoder(self.opts, own_weight_out_params)
        self.self_weight_decoder = self_weight_decoder.to(self.device)

        # use regression for target point prediction
        target_out_params = self.opts["hyperparameters"]["num_modes"] * 2  # 2: (x,y)-Position
        targetpoint_decoder = MMTargetDecoderRegression(self.opts, target_out_params)
        self.targetpoint_decoder = targetpoint_decoder.to(self.device)

    def forward(
            self,
            agent_data: torch.Tensor,
            lane_graph: torch.Tensor
    ) -> tuple:
        """
        forward pass computing prediction and attention based on the given input
        :param agent_data: agent histories
        :param lane_graph: hd map
        :return: control init, game parameters and attention mask
        """
        # ------------------- ENCODING ---------------------------------------  #
        if self.opts["config_params"]["vn_dyn_encoder_type"] == c.LSTM_DYN:
            sg_agent, hidden, cell = self.dynamic_enc(agent_data.to(self.device))
        else:
            sg_agent = self.dynamic_enc(agent_data.to(self.device))
        sg_lane = self.map_enc(lane_graph.to(self.device))
        inter_out, attention_mask = self._interaction(
            sg_agent=sg_agent,
            sg_lane=sg_lane
        )

        agent_num = sg_agent.shape[c.SECOND_VALUE]

        # Extract agent features
        inter_out = du.get_agents(inter_out, agent_num)
        self.inter_out = inter_out


        # Game Parameter Decoding
        u_input, w_own_out, w_int_out, target_out = self.game_parameter_decoding(inter_out)

        # Initial Strategy Decoder
        out = self.init_decoder(u_input)
        out = out.reshape(
            c.LAST_VALUE,
            self.mode_num,
            self.num_predicted_samples,
            self.nbr_out_params
        )

        self.out = out
        out = self._format_output(out, agent_num)
        return out, w_own_out, w_int_out, target_out, attention_mask



    def game_parameter_decoding(
            self,
            inter_out
    ):
        """
        Predicts the game parameters and sets them in the planner
        :param inter_out: encoded features
        :param planner: object of differentiable optimization
        :return:
        """
        # Predict the game_goal
        if self.opts["loss"]["goal_point_mode"] == "MM-SCPR-Regression":
            target_out = self.targetpoint_decoder(inter_out)
            scene_points = einops.rearrange(target_out, 'b p (m s) -> b m p s', m=self.opts["hyperparameters"]["num_modes"], s=2)
            theta_vel_zero = torch.zeros_like(scene_points)
            game_goal = torch.cat([scene_points, theta_vel_zero], dim=-1)[:, :, :, None, :]
        else:
            raise NotImplementedError("Current selected goal point mode. Plase check the config and set it to MM-SCPR-Regression")

        # Predict the cost/ energy function weights
        weight_input = inter_out.type(torch.float32)

        w_own_out = self.self_weight_decoder(weight_input, self.num_of_agents)
        w_int_out = self.interaction_weight_decoder(weight_input, self.num_of_agents)

        # Control decoding
        u_input = inter_out.type(torch.float32)

        return u_input, w_own_out, w_int_out, game_goal

    def get_attention_mask(
            self,
            agent_data: torch.Tensor,
            targets_indices: list,
            lane_graph: torch.Tensor
    ) -> torch.Tensor:
        """
        handles forward pass to return attention mask
        :param agent_data:
        :param targets_indices:
        :param lane_graph:
        :return:
        """
        _, _, _, _, attention_mask = self._forward(
            agent_data,
            targets_indices,
            lane_graph
        )
        return attention_mask



class GameInteractionWeightDecoder(nn.Module):
    def __init__(
            self,
            opts: dict,
            out_params: int
    ):
        super(GameInteractionWeightDecoder, self).__init__()

        # Out params is the number of interaction weights
        self.out_params = out_params
        self.number_player = opts["differentiable_optimization"]["number_player"]

        in_features = opts["hyperparameters"]["vn_global_graph_width"] * opts["differentiable_optimization"]["number_player"]

        self.nbr_decoder_layers = opts["hyperparameters"]["vn_decoder_depth"]
        self.decoder_width = opts["hyperparameters"]["vn_decoder_width"]
        norm_layer = self.get_norm_layer(opts)
        layers = []
        for _ in range(self.nbr_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=self.decoder_width),
                    norm_layer(self.decoder_width),
                    nn.ReLU(True)
                ]
            )
            in_features = self.decoder_width
        layers.append(nn.Linear(in_features=in_features, out_features=self.out_params))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def get_norm_layer(
            opts: dict
    ) -> nn.Module:
        """
        chooses a normalization layer based on the config
        :param opts: config
        :return:
        """
        if opts["hyperparameters"]["norm_layer"] == 'layer':
            norm_layer = nn.LayerNorm
        elif opts["hyperparameters"]["norm_layer"] == 'batch':
            norm_layer = nn.BatchNorm1d
        else:
            raise ValueError(
                'norm_layer is {} but so far only batch and layer '
                'is implemented'.format(opts["hyperparameters"]["norm_layer"])
            )
        return norm_layer

    def forward(
            self,
            input: torch.Tensor,
            num_of_agents
    ) -> torch.Tensor:

        input_shape = input.shape
        input = input.reshape((input_shape[0], -1))
        out = self.layers(input)
        out = out.reshape(input_shape[0], -1)
        return out


class GameSelfWeightDecoder(nn.Module):
    def __init__(
            self,
            opts: dict,
            out_params: int
    ):
        super(GameSelfWeightDecoder, self).__init__()
        self.out_params = out_params
        self.number_player = opts["differentiable_optimization"]["number_player"]

        in_features = opts["hyperparameters"]["vn_global_graph_width"]
        self.nbr_decoder_layers = opts["hyperparameters"]["vn_decoder_depth"]
        self.decoder_width = opts["hyperparameters"]["vn_decoder_width"]
        norm_layer = self.get_norm_layer(opts)
        layers = []
        for _ in range(self.nbr_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=self.decoder_width),
                    norm_layer(self.decoder_width),
                    nn.ReLU(True)
                ]
            )
            in_features = self.decoder_width
        layers.append(nn.Linear(in_features=in_features, out_features=self.out_params))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def get_norm_layer(
            opts: dict
    ) -> nn.Module:
        norm_layer = nn.BatchNorm1d
        return norm_layer

    def forward(
            self,
            input: torch.Tensor,
            num_of_agents
    ) -> torch.Tensor:
        input_shape = input.shape
        input = einops.rearrange(input, 'b a f -> (b a) f')
        out = self.layers(input)
        out = out.view(input_shape[0], -1)

        return out


class MMTargetDecoderRegression(nn.Module):
    """
    Decoder for which outputs one goal point for each mode, which are learned via regression using the minSFDE loss
    """
    def __init__(
            self,
            opts: dict,
            out_params: int
    ):
        super(MMTargetDecoderRegression, self).__init__()
        self.num_modes = opts["hyperparameters"]["num_modes"]
        self.out_params = out_params
        in_features = opts["hyperparameters"]["vn_global_graph_width"]

        self.nbr_decoder_layers = opts["hyperparameters"]["vn_decoder_depth"]
        self.decoder_width = opts["hyperparameters"]["vn_decoder_width"] * opts["hyperparameters"]["vn_mm_goal_decoder_multiplier"]

        norm_layer = self.get_norm_layer(opts)
        layers = []
        for _ in range(self.nbr_decoder_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_features=in_features, out_features=self.decoder_width),
                    norm_layer(self.decoder_width),
                    nn.ReLU(True)
                ]
            )
            in_features = self.decoder_width
        layers.append(nn.Linear(in_features=in_features, out_features=self.out_params))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def get_norm_layer(
            opts: dict
    ) -> nn.Module:
        norm_layer = nn.BatchNorm1d
        return norm_layer

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        input_shape = input.shape
        input = einops.rearrange(input, 'b a f -> (b a) f')
        out = self.layers(input)
        out = einops.rearrange(out, '(b a) f -> b a f', b=input_shape[0])
        return out


class SceneProbabilityPredictor(nn.Module):
    def __init__(self, input_size: int,
                 modes_num: int,
                 intermediate_neurons_num: int = 16,
                 opts=None):
        super(SceneProbabilityPredictor, self).__init__()
        self.decoder_width = opts["hyperparameters"]["vn_decoder_width"]
        norm_layer = nn.BatchNorm1d
        intermediate_neurons_num = intermediate_neurons_num * modes_num
        self.input_size = input_size
        self.scene_prob_predictor = nn.Sequential(
            nn.Linear(input_size, intermediate_neurons_num),
            norm_layer(intermediate_neurons_num),
            nn.ReLU(True),
            nn.Linear(intermediate_neurons_num, intermediate_neurons_num),
            norm_layer(intermediate_neurons_num),
            nn.ReLU(True),
            nn.Linear(intermediate_neurons_num, modes_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        if self.input_size != x.shape[1]:
            print("Warning: input size of scene_prob_predictor is not equal to the size of the input data")
            print(f"input size: {self.input_size}, input data size: {x.shape}")
            return x
        scene_probs = self.scene_prob_predictor(x)
        return scene_probs



