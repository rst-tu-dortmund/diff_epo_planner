# ------------------------------------------------------------------------------
# Copyright:    ZF Friedrichshafen AG
# Project:      KISSaF
# Created by:   ZF AI LAB SBR
# ------------------------------------------------------------------------------
import data.constants as c
import torch
import torch.nn as nn


class SubGraphLayer(nn.Module):
    def __init__(
        self,
        in_features: int = c.VN_INPUT_SIZE_DYN,
        hidden_size: int = c.VN_HIDDEN_SIZE,
    ):
        super(SubGraphLayer, self).__init__()
        self.hidden_size = hidden_size
        self.in_features = in_features
        self.linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.hidden_size,
            bias=True,
        )
        self.lm = nn.LayerNorm(normalized_shape=self.hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x_encoded = self.encode(x)
        aggregate = self.aggregate(x_encoded).repeat(
            c.ONE,
            c.ONE,
            x_encoded.shape[c.TWO],
            c.ONE,
        )
        out = torch.cat([x_encoded, aggregate], dim=c.THREE)
        assert out.shape == (
            x_encoded.shape[c.ZERO],
            x_encoded.shape[c.ONE],
            x_encoded.shape[c.TWO],
            self.hidden_size * c.TWO,
        )

        if mask is not None:
            out = out * (
                mask.unsqueeze(dim=c.LAST_VALUE).repeat(
                    c.ONE,
                    c.ONE,
                    c.ONE,
                    self.hidden_size * c.TWO,
                )
            )
            assert out.shape == (
                x_encoded.shape[c.ZERO],
                x_encoded.shape[c.ONE],
                x_encoded.shape[c.TWO],
                self.hidden_size * c.TWO,
            )

        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if c.ENC_NORM==1:
            x = self.lm(x)
        return torch.relu(x)

    @staticmethod
    def aggregate(x: torch.Tensor) -> torch.Tensor:
        y, _ = torch.max(x, dim=2)
        return y.unsqueeze(2)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int = c.VN_HIDDEN_SIZE * c.TWO,
        out_features: int = c.VN_HIDDEN_SIZE * c.TWO,
        in_features_keys: int = None,
    ):
        """Self-attention layer keys and values are the same tensor."""
        super(SelfAttentionLayer, self).__init__()
        if in_features_keys is None:
            in_features_keys = in_features
        self.Proj_Q = nn.Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
        )
        self.Proj_K = nn.Linear(
            in_features=in_features_keys,
            out_features=out_features,
            bias=False,
        )
        self.Proj_V = nn.Linear(
            in_features=in_features_keys,
            out_features=out_features,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=c.TWO)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """generates embeddings and attention mask.

        :param query:
        :param key:
        :param attention_mask:
        :return:
        """
        P_q = self.Proj_Q(query)
        P_k = self.Proj_K(key)
        P_v = self.Proj_V(key)
        out = torch.bmm(P_q, P_k.transpose(c.ONE, c.TWO))
        # mask for self attention
        if attention_mask is not None:
            out = out.masked_fill(
                attention_mask.unsqueeze(1).expand(
                    c.LAST_VALUE,
                    query.shape[c.ONE],
                    c.LAST_VALUE,
                )
                == c.ZERO,
                -1e9,
            )
        attention_mask = self.softmax(out)
        out = torch.bmm(attention_mask, P_v)
        return out, attention_mask
