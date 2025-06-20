# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import final

import torch
from torch.nn import Module

from fairseq2.assets import (
    AssetCardError,
    AssetCardFieldNotFoundError,
    AssetCardNotFoundError,
    AssetStore,
)
from fairseq2.context import RuntimeContext
from fairseq2.data_type import DataType
from fairseq2.error import NotSupportedError
from fairseq2.gang import GangError, Gangs
from fairseq2.logging import log
from fairseq2.models import (
    InvalidModelTypeError,
    ModelConfigLoadError,
    ModelHandler,
    ModelLoadError,
    UnknownModelError,
    UnknownModelFamilyError,
    model_asset_card_error,
)
from fairseq2.nn.utils.module import remove_parametrizations
from fairseq2.recipes import Model, RecipeError
from fairseq2.recipes.config import ReferenceModelSection
from fairseq2.recipes.utils.log import log_model
from fairseq2.registry import Provider

# isort: split

from fairseq2.recipes.common._compile import compile_model
from fairseq2.recipes.common._distributed import broadcast_model
from fairseq2.recipes.common._error import ModelParallelismNotSupportedError
from fairseq2.recipes.common._model import _LocalModel


def setup_reference_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ReferenceModelSection,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
) -> Model:
    model = load_reference_model(kls, context, model_section, gangs, dtype, mp)

    broadcast_model(model, gangs)

    prepare_reference_model(context, model, model_section)

    log_model(log, model.module, gangs)

    return model


def load_reference_model(
    kls: type[Module],
    context: RuntimeContext,
    model_section: ReferenceModelSection,
    gangs: Gangs,
    dtype: DataType,
    mp: bool,
) -> Model:
    model_handlers = context.get_registry(ModelHandler)

    loader = _ReferenceModelLoader(kls, context.asset_store, model_handlers)

    try:
        return loader.load(model_section, gangs, dtype, mp)
    except ModelLoadError as ex:
        raise RecipeError(
            f"The '{ex.model_name}' model cannot be loaded. See the nested exception for details."
        ) from ex


@final
class _ReferenceModelLoader:
    _kls: type[Module]
    _asset_store: AssetStore
    _model_handlers: Provider[ModelHandler]

    def __init__(
        self,
        kls: type[Module],
        asset_store: AssetStore,
        model_handlers: Provider[ModelHandler],
    ) -> None:
        self._kls = kls
        self._asset_store = asset_store
        self._model_handlers = model_handlers

    def load(
        self,
        model_section: ReferenceModelSection,
        gangs: Gangs,
        dtype: DataType,
        mp: bool,
    ) -> Model:
        model_name = model_section.name

        try:
            card = self._asset_store.retrieve_card(model_name)
        except AssetCardNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            model_family = card.field("model_family").as_(str)
        except AssetCardFieldNotFoundError:
            raise UnknownModelError(model_name) from None
        except AssetCardError as ex:
            raise model_asset_card_error(model_name) from ex

        try:
            handler = self._model_handlers.get(model_family)
        except LookupError:
            raise UnknownModelFamilyError(model_family, model_name) from None

        if not issubclass(handler.kls, self._kls):
            raise InvalidModelTypeError(model_name, handler.kls, self._kls) from None

        if gangs.root.size != gangs.dp.size:
            if not handler.supports_model_parallelism:
                raise ModelParallelismNotSupportedError(model_name)

        try:
            model_config = handler.load_config(card)
        except ModelConfigLoadError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model configuration cannot be loaded. See the nested exception for details."  # fmt: skip
            ) from ex

        # Load the model.
        log.info("Loading '{}' model on data parallel rank 0.", model_name)

        if mp:
            dtype = torch.float32

        try:
            if gangs.dp.rank == 0:
                module = handler.load(
                    card, gangs, dtype, model_config, mmap=model_section.mmap
                )
            else:
                module = handler.create(
                    model_config, gangs, dtype, meta=handler.supports_meta
                )
        except NotSupportedError as ex:
            raise ModelLoadError(
                model_name, f"The '{model_name}' model cannot be constructed due to an unsupported operation. See the nested exception for details."  # fmt: skip
            ) from ex

        try:
            gangs.root.barrier()
        except GangError as ex:
            raise ModelLoadError(
                model_name, f"The collective barrier after the '{model_name}' model load operation has failed. See the nested exception for details."  # fmt: skip
            ) from ex

        module.eval()

        log.info("Model loaded on data parallel rank 0.")

        return _LocalModel(model_name, module, model_config, handler)


def prepare_reference_model(
    context: RuntimeContext, model: Model, model_section: ReferenceModelSection
) -> Model:
    remove_parametrizations(model.module)

    if model_section.compile:
        compile_model(model, model_section.compile_options)

    return model
