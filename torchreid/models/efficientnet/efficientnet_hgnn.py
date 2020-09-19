"""model.py - Model and module class for EfficientNet_HGNN
   They are built to mirror those in the official TensorFlow implementation.
"""

# Author: lukemelas (github username)
# Github repo: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from .utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size,
    calc_splits,
    weights_init_kaiming,
    weights_init_classifier
)
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H

VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block.

    Args:
        block_args (namedtuple): BlockArgs, defined in utils.py.
        global_params (namedtuple): GlobalParam, defined in utils.py.
        image_size (tuple or list): [image_height, image_width].

    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock's forward function.

        Args:
            inputs (tensor): Input tensor.
            drop_connect_rate (bool): Drop connect rate (float, between 0 and 1).

        Returns:
            Output of this block after processing.
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet_HGNN(nn.Module):
    """EfficientNet model.
       Most easily loaded with the .from_name or .from_pretrained methods.

    Args:
        blocks_args (list[namedtuple]): A list of BlockArgs to construct blocks.
        global_params (namedtuple): A set of GlobalParams shared between blocks.

    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)

    Example:


        import torch
        >>> from efficientnet.model import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('efficientnet-b0')
        >>> model.eval()
        >>> outputs = model(inputs)
    """

    def __init__(self, blocks_args=None, global_params=None, other_params=None,**kwargs):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self.other_params = other_params
        self.training = False
        self.loss = {'xent', 'htri'}

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # modify block_args to keep same output size
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # global branch, Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._global_bottleneck = nn.BatchNorm1d(out_channels)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes) if 'num_classes' not in self.other_params \
            else nn.Linear(out_channels, self.other_params['num_classes'])
        self._swish = MemoryEfficientSwish()

        # hgnn branch
        # split branch, from layer4_2
        self.num_split = self.other_params['num_split']
        self.total_split_list = calc_splits(self.num_split) if self.other_params['pyramid_part'] else [self.num_split]
        self.total_split = sum(self.total_split_list)

        self.parts_avgpool = nn.ModuleList()
        for n in self.total_split_list:
            self.parts_avgpool.append(nn.AdaptiveAvgPool2d((n, 1)))

        # hgnn graph layers
        self.m_prob = self.other_params['m_prob']  # parameter in hypergraph incidence matrix construction
        self.K_neigs = self.other_params['K_neigs']  # the number of neighbor expansion
        self.is_probH = self.other_params['is_probH']  # probability Vertex-Edge matrix or binary
        self.dropout = self.other_params['dropout']
        self.hgc1 = HGNN_conv(out_channels,
                              out_channels)  # self.hgc1 = HGNN_conv(self.feature_dim, self.n_hid)
        self.hgc2 = HGNN_conv(out_channels, out_channels)  # self.hgc2 = HGNN_conv(self.n_hid, num_classes)

        # attention branch
        # if self.other_params['learn_attention']:
        #     self.attention_weight = Parameter(torch.Tensor(self.seq_len, self.total_split, 1))
        #     self._reset_attention_parameters()

        self.consistent_loss =self.other_params['consistent_loss']

        self.att_bottleneck = nn.BatchNorm1d(out_channels)  # nn.BatchNorm1d(num_classes)
        self.att_bottleneck.bias.requires_grad_(False)
        self.att_classifier = nn.Linear(out_channels, self.other_params['num_classes'], bias=False)
        weights_init_kaiming(self.att_bottleneck)
        weights_init_classifier(self.att_classifier)


    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.

        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of the final convolution
            layer in the efficientnet model.
        """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output of this model after processing.
        """
        # Convolution layers
        x = self.extract_features(inputs)
        b, c, h, w = x.shape

        # # Pooling and final linear layer
        # x = self._avg_pooling(x4_1)
        # if self._global_params.include_top:
        #     x = x.flatten(start_dim=1)
        #     x = self._dropout(x)
        #     x = self._fc(x)

        # global branch
        if self.other_params['global_branch']:
            x4_1 = x.view(b, c, h, w).contiguous()
            g_f = self._avg_pooling(x4_1).view(b, -1)
            g_bn = self._global_bottleneck(g_f)

        # split branch
        v_f = list()
        for idx, n in enumerate(self.total_split_list):
            v_f.append(self.parts_avgpool[idx](x).view(b, c, n))
        v_f = torch.cat(v_f, dim=2)
        f = v_f.transpose(1, 2).contiguous()

        # construct the hgnn graph
        G = []
        for feature in f:
            H = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.other_params['K_neigs'],
                                     is_probH=self.other_params['is_probH'],
                                     m_prob=self.other_params['m_prob'])
            g = generate_G_from_H(H, variable_weight=False)
            G.append(g.A)
        G = torch.tensor(G, dtype=torch.float32)

        # hgnn graph propogation
        f = F.relu(self.hgc1(f, G))
        f = F.dropout(f, self.dropout)
        f = self.hgc2(f, G)

        # attention branch
        if self.other_params['learn_attention']:  # learn the weight of attention
            f_fuse = self._learn_attention_op(f)
        else:  # calculate the norm as the attention weight
            f_fuse = self._attention_op(f)

        att_f = f_fuse.view(b, -1)
        att_bn = self.att_bottleneck(att_f)

        if not self.training or self.other_params['isFinal']:
            if self.other_params['global_branch']: # use two branch
                return torch.cat([g_bn, att_bn], dim=1)
            else:
                return att_bn

        if self.other_params['global_branch']:
            g_out = self._fc(g_bn)
        att_out = self.att_classifier(att_bn)

        if self.loss == {'xent'}:
            out_list = [g_out, att_out] if self.other_params['global_branch'] else [att_out]
            # if self.consistent_loss:
            #     out_list.extend(satt_out_list)
            return out_list
        elif self.loss == {'xent', 'htri'}:
            out_list = [g_out, att_out] if self.other_params['global_branch'] else [att_out]
            f_list = [g_f, att_f] if self.other_params['global_branch'] else [att_f]
            # if self.consistent_loss:
            #     out_list.extend(satt_out_list)
            #     f_list.extend(satt_f_list)
            return out_list, f_list
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))

    def _attention_op(self, feat):
        """
        do attention fusion
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        att = F.normalize(feat.norm(p=2, dim=2, keepdim=True), p=1, dim=1)
        f = feat.mul(att).sum(dim=1)
        return f

    # def _learn_attention_op(self,feat):
    #     """
    #     do attention fusion, with the weight learned
    #     :param feat: (batch, seq_len, num_split, c)
    #     :return: feat: (batch, num_split, c)
    #     """
    #     f = feat.mul(self.attention_weight)
    #     f = f.sum(dim=1)
    #     return f

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params, other_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params, other_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """create an efficientnet model according to name.

        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path, load_fc=(num_classes == 1000),
                                advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """Get the input image size for a given efficientnet model.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """Validates model name.

        Args:
            model_name (str): Name for efficientnet.

        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """Adjust model's first convolution layer to in_channels, if in_channels not equals 3.

        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


def efficientnet_hgnn(num_classes, isFinal=False, global_branch=False, arch="efficientnet-b0"):
    # arch = "resnet50" # can be changed
    # layers_dict = {"resnet50":[3, 4, 6, 3],
    #                "resnet101":[3, 4, 23, 3],
    #                "resnet152":[3, 8, 36, 3]}
    # layers = layers_dict[arch]
    model = EfficientNet_HGNN.from_pretrained(
        model_name = arch,
        num_classes=num_classes,
        # block=Bottleneck,
        # layers=layers,
        last_stride=1,
        num_split=8,
        pyramid_part=True,
        num_gb=2,
        use_pose=False,
        learn_graph=True,
        consistent_loss=False,
        m_prob=1.0,
        K_neigs=[3],
        is_probH=True,
        dropout=0.5,
        learn_attention=False,
        isFinal = isFinal,
        global_branch=global_branch
    )

    return model