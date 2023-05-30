import copy
import warnings
import math
from typing import List, Tuple, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from src.models.sequence.base import SequenceModule

from src.ops.sru_ops import (elementwise_recurrence_inference,
                     elementwise_recurrence_gpu,
                     elementwise_recurrence_naive)


class SRUCell(SequenceModule):
    """
    A single SRU layer as per `LSTMCell`, `GRUCell` in Pytorch.
    """

    __constants__ = ['input_size', 'hidden_size', 'output_size', 'rnn_dropout',
                     'dropout', 'bidirectional', 'has_skip_term', 'highway_bias',
                     'v1', 'rescale', 'activation_type', 'activation', 'custom_m',
                     'projection_size', 'num_matrices', 'layer_norm', 'weight_proj',
                     'scale_x', 'normalize_after', 'weight_c_init',]

    scale_x: Tensor
    weight_proj: Optional[Tensor]

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 d_output: int,
                 dropout: float = 0.0,
                 rnn_dropout: float = 0.0,
                 bidirectional: bool = False,
                 n_proj: int = 0,
                 use_tanh: bool = False,
                 highway_bias: float = 0.0,
                 has_skip_term: bool = True,
                 layer_norm: bool = False,
                 rescale: bool = True,
                 v1: bool = False,
                 custom_m: Optional[nn.Module] = None,
                 amp_recurrence_fp16: bool = False,
                 normalize_after: bool = False,
                 weight_c_init: Optional[float] = None,
                 **kwargs):
        """Initialize the SRUCell module.

        Parameters
        ----------
        input_size: int
            the number of features in the input `x`
        hidden_size: int
            the number of features in the hidden state *for each
            direction*
        dropout: float, optional
            the dropout value applied between layers (default=0)
        rnn_dropout: float, optional
            [DEPRECATED] the variational dropout value (default=0)
            This option is deprecated because minimal performance
            improvement, and increases codebase size. This option will
            be removed at the next major version upgrade
        bidirectional: bool, optional
            if True, set the module as a bidirectional SRU
            (default=False)
        n_proj: int, optional
            if non-zero, factorize the ``weight`` parameter matrix as a
            product of two parameter matrices, using an innder dimension
            ``n_proj`` (default=0)
        use_tanh: bool, optional
            [DEPRECATED] if True, apply `tanh` activation to the hidden
            state (default=False). `tanh` is deprecated because minimal
            performance improvement, and increases codebase size. This
            option will be removed at the next major version upgrade.
        highway_bias: float, optional
            the initial value of the bias used in the highway (sigmoid)
            gate (defulat=0)
        has_skip_term: bool, optional
            whether to include a residual connection for output hidden
            state `h` (default=True)
        layer_norm: bool, optional
            whether to apply pre- layer normalization for this layer
            (default=False)
        rescale: bool, optional
            whether to apply a constant rescaling multiplier for the
            residual term (default=True)
        v1: bool, optional
            [DEPRECATED] whether to use the an ealier v1 implementation
            of SRU (default=False)
        custom_m: nn.Module, optional
            use the give module instead of the batched matrix
            multiplication to compute the intermediate representations U
            needed for the elementwise recurrrence operation
            (default=None)
        amp_recurrence_fp16: Type, optional
            When using AMP autocast, selects which type to use
            for recurrence custom kernel.
            False: torch.float32, True: torch.float16
        normalize_after: bool
            if True use post layer norm, else pre layer norm
        weight_c_init: Optional[float]
            if not None, then size of uniform initiatialization of weight_c
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size  # hidden size per direction
        self.d_output = d_output
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = float(rnn_dropout)
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_m: Optional[nn.Module] = custom_m
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.amp_recurrence_fp16 = amp_recurrence_fp16
        self.normalize_after = normalize_after
        self.weight_c_init = weight_c_init

        # projection dimension
        self.projection_size = 0
        if n_proj > 0 and n_proj < self.input_size and n_proj < self.output_size:
            self.projection_size = n_proj

        # number of sub-matrices used in SRU
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4

        # make parameters
        if self.custom_m is None:
            if self.projection_size == 0:
                self.weight_proj = None
                self.weight = nn.Parameter(torch.Tensor(
                    input_size,
                    self.output_size * self.num_matrices
                ))
            else:
                self.weight_proj = nn.Parameter(torch.Tensor(input_size, self.projection_size))
                self.weight = nn.Parameter(torch.Tensor(
                    self.projection_size,
                    self.output_size * self.num_matrices
                ))
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))

        # scaling constant used in highway connections when rescale=True
        self.register_buffer('scale_x', torch.FloatTensor([0]))

        self.layer_norm: Optional[nn.Module]= None
        if layer_norm:
            if normalize_after:
                self.layer_norm = nn.LayerNorm(self.output_size)
            else:
                self.layer_norm = nn.LayerNorm(self.input_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Properly initialize the weights of SRU, following the same
        recipe as:
        Xavier init:
            http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        Kaiming init:
            https://arxiv.org/abs/1502.01852

        """
        # initialize bias and scaling constant
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            # scalar used to properly scale the highway output
            scale_val = (1 + math.exp(bias_val) * 2)**0.5
            self.scale_x.data[0] = scale_val

        if self.custom_m is None:
            # initialize weights such that E[w_ij]=0 and Var[w_ij]=1/d
            d = self.weight.size(0)
            val_range = (3.0 / d)**0.5
            self.weight.data.uniform_(-val_range, val_range)
            if self.projection_size > 0:
                val_range = (3.0 / self.weight_proj.size(0))**0.5
                self.weight_proj.data.uniform_(-val_range, val_range)

            # projection matrix as a tensor of size:
            #    (input_size, bidirection, hidden_size, num_matrices)
            w = self.weight.data.view(d, -1, self.hidden_size, self.num_matrices)

            # re-scale weights for dropout and normalized input for better gradient flow
            if self.dropout > 0:
                w[:, :, :, 0].mul_((1 - self.dropout)**0.5)
            if self.rnn_dropout > 0:
                w.mul_((1 - self.rnn_dropout)**0.5)

            # making weights smaller when layer norm is used. need more tests
            if self.layer_norm:
                w.mul_(0.1)
                # self.weight_c.data.mul_(0.25)

            # properly scale the highway output
            if self.rescale and self.has_skip_term and self.num_matrices == 4:
                scale_val = (1 + math.exp(bias_val) * 2)**0.5
                w[:, :, :, 3].mul_(scale_val)
        else:
            if hasattr(self.custom_m, 'reset_parameters'):
                self.custom_m.reset_parameters()
            else:
                warnings.warn("Unable to reset parameters for custom module. "
                              "reset_parameters() method not found for custom module. "
                              + self.custom_m.__class__.__name__)

        if not self.v1:
            # intialize weight_c such that E[w]=0 and Var[w]=1
            if self.weight_c_init is None:
                self.weight_c.data.uniform_(-3.0**0.5, 3.0**0.5)
                self.weight_c.data.mul_(0.5**0.5)
            else:
                self.weight_c.data.uniform_(-self.weight_c_init, self.weight_c_init)

            # rescale weight_c and the weight of sigmoid gates with a factor of sqrt(0.5)
            if self.custom_m is None:
                w[:, :, :, 1].mul_(0.5**0.5)
                w[:, :, :, 2].mul_(0.5**0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self,
                input: Tensor,
                state: Optional[Tensor] = None,
                mask_pad: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """The forward method of the SRU layer.
        """
        # print("SRU Cell forward...")
        # breakpoint()

        if input.dim() != 2 and input.dim() != 3:
            raise ValueError("Input must be 2 or 3 dimensional")

        batch_size = input.size(-2)
        if state is None:
            state = torch.zeros(batch_size, self.output_size, dtype=input.dtype,
                             device=input.device)

        # apply layer norm before activation (i.e. before SRU computation)
        residual = input
        if self.layer_norm is not None and not self.normalize_after:
            input = self.layer_norm(input)

        # apply dropout for multiplication
        if self.training and (self.rnn_dropout > 0):
            mask = self.get_dropout_mask_((batch_size, input.size(-1)), self.rnn_dropout)
            input = input * mask.expand_as(input)

        # get the scaling constant; scale_x is a scalar
        scale_val: Optional[Tensor] = None
        scale_val = self.scale_x if self.rescale else None

        # get dropout mask
        mask_c: Optional[Tensor] = None
        if self.training and (self.dropout > 0):
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                                            self.dropout)

        # compute U, V
        #   U is (length, batch_size, output_size * num_matrices)
        #   V is (output_size*2,) or (length, batch_size, output_size * 2) if provided
        U, V = self.compute_UV(input, state, mask_pad)

        # apply elementwise recurrence to get hidden states h and c
        h, c = self.apply_recurrence(U, V, residual, state, scale_val, mask_c, mask_pad)

        if self.layer_norm is not None and self.normalize_after:
            h = self.layer_norm(h)

        return h, c

    def apply_recurrence(self,
                         U: Tensor,
                         V: Tensor,
                         residual: Tensor,
                         state: Tensor,
                         scale_val: Optional[Tensor],
                         mask_c: Optional[Tensor],
                         mask_pad: Optional[Tensor]) -> List[Tensor]:
        """
        Apply the elementwise recurrence computation on given input
        tensors

        """
        if not torch.jit.is_scripting():
            if self.bias.is_cuda:
                return elementwise_recurrence_gpu(U, residual, V, self.bias, state,
                                                  self.activation_type,
                                                  self.hidden_size,
                                                  self.bidirectional,
                                                  self.has_skip_term,
                                                  scale_val, mask_c, mask_pad,
                                                  self.amp_recurrence_fp16)
            else:
                return elementwise_recurrence_naive(U, residual, V, self.bias, state,
                                                    self.activation_type,
                                                    self.hidden_size,
                                                    self.bidirectional,
                                                    self.has_skip_term,
                                                    scale_val, mask_c, mask_pad)
        else:
            return elementwise_recurrence_inference(U, residual, V, self.bias, state,
                                                    self.activation_type,
                                                    self.hidden_size,
                                                    self.bidirectional,
                                                    self.has_skip_term,
                                                    scale_val, mask_c, mask_pad)


    def compute_UV(self,
                   input: Tensor,
                   state: Optional[Tensor],
                   mask_pad: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices).

        When a custom module `custom_m` is given, U will be computed by
        the given module. In addition, the module can return an
        additional tensor V (length, batch_size, output_size * 2) that
        will be added to the hidden-to-hidden coefficient terms in
        sigmoid gates, i.e., (V[t, b, d] + weight_c[d]) * c[t-1].

        """
        if self.custom_m is None:
            U = self.compute_U(input)
            V = self.weight_c
        else:
            ret = self.custom_m(input)
            if isinstance(ret, tuple) or isinstance(ret, list):
                if len(ret) > 2:
                    raise Exception("Custom module must return 1 or 2 tensors but got {}.".format(
                        len(ret)
                    ))
                U, V = ret[0], ret[1] + self.weight_c
            else:
                U, V = ret, self.weight_c

            if U.size(-1) != self.output_size * self.num_matrices:
                raise ValueError("U must have a last dimension of {} but got {}.".format(
                    self.output_size * self.num_matrices,
                    U.size(-1)
                ))
            if V.size(-1) != self.output_size * 2:
                raise ValueError("V must have a last dimension of {} but got {}.".format(
                    self.output_size * 2,
                    V.size(-1)
                ))
        return U, V

    def compute_U(self,
                  input: Tensor) -> Tensor:
        """
        SRU performs grouped matrix multiplication to transform the
        input (length, batch_size, input_size) into a tensor U of size
        (length * batch_size, output_size * num_matrices)
        """
        # collapse (length, batch_size) into one dimension if necessary
        x = input if input.dim() == 2 else input.contiguous().view(-1, self.input_size)
        weight_proj = self.weight_proj
        if weight_proj is not None:
            x_projected = x.mm(weight_proj)
            U = x_projected.mm(self.weight)
        else:
            U = x.mm(self.weight)
        return U

    def get_dropout_mask_(self,
                          size: Tuple[int, int],
                          p: float) -> Tensor:
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new_empty(size).bernoulli_(1 - p).div_(1 - p)

    def extra_repr(self):
        s = "{input_size}, {hidden_size}"
        if self.projection_size > 0:
            s += ", projection_size={projection_size}"
        if self.dropout > 0:
            s += ", dropout={dropout}"
        if self.rnn_dropout > 0:
            s += ", rnn_dropout={rnn_dropout}"
        if self.bidirectional:
            s += ", bidirectional={bidirectional}"
        if self.highway_bias != 0:
            s += ", highway_bias={highway_bias}"
        if self.activation_type != 0:
            s += ", activation={activation}"
        if self.v1:
            s += ", v1={v1}"
        s += ", rescale={rescale}"
        if not self.has_skip_term:
            s += ", has_skip_term={has_skip_term}"
        if self.layer_norm:
            s += ", layer_norm=True"
        if self.custom_m is not None:
            s += ",\n  custom_m=" + str(self.custom_m)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return "{}({})".format(self.__class__.__name__, s)
        else:
            return "{}({}\n)".format(self.__class__.__name__, s)
