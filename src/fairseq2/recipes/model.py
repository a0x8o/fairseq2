# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from fairseq2.models import ModelHandler
from fairseq2.typing import ContextManager


class Model(ABC):
    @abstractmethod
    def no_sync(self) -> ContextManager: ...

    @abstractmethod
    def clip_gradient_norm(self, max_norm: float | None) -> Tensor: ...

    @abstractmethod
    def state_dict(self) -> dict[str, object]: ...

    @abstractmethod
    def optim_state_dict(self, optim: Optimizer) -> dict[str, object]: ...

    @abstractmethod
    def load_optim_state_dict(
        self, optim: Optimizer, state_dict: Mapping[str, object]
    ) -> None: ...

    @abstractmethod
    def summon_parameters(self) -> ContextManager: ...

    @property
    @abstractmethod
    def module(self) -> Module: ...

    @property
    @abstractmethod
    def base_module(self) -> Module: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def config(self) -> object: ...

    @property
    @abstractmethod
    def handler(self) -> ModelHandler: ...

    @property
    @abstractmethod
    def is_empty_init(self) -> bool: ...
