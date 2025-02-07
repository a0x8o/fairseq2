# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Set


class UnknownDatasetError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str) -> None:
        super().__init__(f"'{dataset_name}' is not a known dataset.")

        self.dataset_name = dataset_name


class UnknownDatasetFamilyError(Exception):
    family: str
    dataset_name: str | None

    def __init__(self, family: str, dataset_name: str | None = None) -> None:
        if dataset_name is None:
            message = f"'{family}' is not a known dataset family."
        else:
            message = f"The '{dataset_name}' dataset has an unknown family '{family}'"

        super().__init__(message)

        self.family = family
        self.dataset_name = dataset_name


class DatasetLoadError(Exception):
    dataset_name: str

    def __init__(self, dataset_name: str, message: str) -> None:
        super().__init__(message)

        self.dataset_name = dataset_name


def dataset_asset_card_error(dataset_name: str) -> DatasetLoadError:
    return DatasetLoadError(
        dataset_name, f"The '{dataset_name}' asset card cannot be read. See the nested exception for details."  # fmt: skip
    )


class InvalidDatasetTypeError(Exception):
    kls: type[object]
    expected_kls: type[object]
    dataset_name: str | None

    def __init__(
        self,
        kls: type[object],
        expected_kls: type[object],
        dataset_name: str | None = None,
    ) -> None:
        if dataset_name is None:
            message = f"The dataset is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."
        else:
            message = f"The '{dataset_name}' dataset is expected to be of type `{expected_kls}`, but is of type `{kls}` instead."

        super().__init__(message)

        self.kls = kls
        self.expected_kls = expected_kls
        self.dataset_name = dataset_name


class UnknownSplitError(ValueError):
    dataset_name: str
    split: str
    available_splits: Set[str]

    def __init__(
        self, dataset_name: str, split: str, available_splits: Set[str]
    ) -> None:
        s = ", ".join(sorted(available_splits))

        super().__init__(
            f"'{split}' is not a known split of the '{dataset_name}' dataset. The following splits are available: {s}"
        )

        self.dataset_name = dataset_name
        self.split = split
        self.available_splits = available_splits
