# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.nn.embedding import Embedding as Embedding
from fairseq2.nn.embedding import ShardedEmbedding as ShardedEmbedding
from fairseq2.nn.embedding import StandardEmbedding as StandardEmbedding
from fairseq2.nn.embedding import VocabShardedEmbedding as VocabShardedEmbedding
from fairseq2.nn.embedding import init_scaled_embedding as init_scaled_embedding
from fairseq2.nn.incremental_state import IncrementalState as IncrementalState
from fairseq2.nn.incremental_state import IncrementalStateBag as IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm as LayerNorm
from fairseq2.nn.normalization import RMSNorm as RMSNorm
from fairseq2.nn.normalization import StandardLayerNorm as StandardLayerNorm
from fairseq2.nn.position_encoder import (
    InterpolatedPositionEncoder as InterpolatedPositionEncoder,
)
from fairseq2.nn.position_encoder import (
    LearnedPositionEncoder as LearnedPositionEncoder,
)
from fairseq2.nn.position_encoder import PositionEncoder as PositionEncoder
from fairseq2.nn.position_encoder import RotaryEncoder as RotaryEncoder
from fairseq2.nn.position_encoder import (
    Sinusoidal2dPositionEncoder as Sinusoidal2dPositionEncoder,
)
from fairseq2.nn.position_encoder import (
    Sinusoidal3dPositionEncoder as Sinusoidal3dPositionEncoder,
)
from fairseq2.nn.position_encoder import (
    SinusoidalNdPositionEncoder as SinusoidalNdPositionEncoder,
)
from fairseq2.nn.position_encoder import (
    SinusoidalPositionEncoder as SinusoidalPositionEncoder,
)
from fairseq2.nn.projection import ColumnShardedLinear as ColumnShardedLinear
from fairseq2.nn.projection import Linear as Linear
from fairseq2.nn.projection import Projection as Projection
from fairseq2.nn.projection import RowShardedLinear as RowShardedLinear
from fairseq2.nn.projection import TiedProjection as TiedProjection
