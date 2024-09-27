# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, final

import numpy as np
import numba as nb
import torch
from fairseq2.assets import AssetCard, AssetError
from fairseq2.data import Collater, DataPipelineBuilder, FileMapper, read_sequence
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets.batching import Batching, LengthBatching
from fairseq2.datasets.data_reader import DataPipelineReader, DataReader
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import AbstractDatasetLoader, DelegatingDatasetLoader
from fairseq2.gang import Gang
from fairseq2.logging import get_log_writer
from fairseq2.models.sequence import SequenceBatch
from fairseq2.typing import DataType
from torch import Tensor
from torch.nn.functional import layer_norm
from typing_extensions import override

from fairseq2.datasets.mmap_indexed import MMapIndexedDataset
from fairseq2.datasets.speech import SpeechDataset, load_speech_dataset

log = get_log_writer(__name__)


@nb.jit(nopython=True)
def batch_by_size_vec(indices, num_tokens_vec, max_tokens, max_sentences, bsz_mult):
    assert max_tokens <= 0 or np.max(num_tokens_vec) <= max_tokens

    indices_len = len(indices)
    batches_ends = np.zeros(indices_len, dtype=np.int32)
    pos = 0
    new_batch_end = 0
    new_batch_max_tokens = 0
    new_batch_sentences = 0
    new_batch_num_tokens = 0
    overflow = False
    size_matches_with_bsz_mult = False
    batches_count = 0
    batch_start = 0
    tail_max_tokens = 0
    batch_max_tokens = 0

    for pos in range(indices_len):
        tail_max_tokens = max(tail_max_tokens, num_tokens_vec[pos])
        new_batch_end = pos + 1
        new_batch_max_tokens = max(batch_max_tokens, tail_max_tokens)
        new_batch_sentences = new_batch_end - batch_start
        new_batch_num_tokens = new_batch_sentences * new_batch_max_tokens
        overflow = (
            new_batch_sentences > max_sentences > 0
            or new_batch_num_tokens > max_tokens > 0
        )
        size_matches_with_bsz_mult = (
            new_batch_sentences < bsz_mult or new_batch_sentences % bsz_mult == 0
        )
        if overflow:
            tail_num_tokens = tail_max_tokens * (
                new_batch_end - batches_ends[batches_count]
            )
            tail_overflow = tail_num_tokens > max_tokens > 0
            if tail_overflow:
                batches_count += 1
                batches_ends[batches_count] = pos
                tail_max_tokens = num_tokens_vec[pos]
            batch_start = batches_ends[batches_count]
            batches_count += 1
            new_batch_max_tokens = tail_max_tokens
        if overflow or size_matches_with_bsz_mult:
            batches_ends[batches_count] = new_batch_end
            batch_max_tokens = new_batch_max_tokens
            tail_max_tokens = 0
    if batches_ends[batches_count] != indices_len:
        batches_count += 1

    return np.split(indices, batches_ends[:batches_count])


# TODO: FIX, INFER
if "FAIR_ENV_CLUSTER" in os.environ:
    npc = 10
else:
    npc = 12


class AudioCropper:
    def __init__(self, max_audio_len: int, rng: np.random.Generator) -> None:
        self.rng = rng
        self.max_audio_len = max_audio_len

    def crop_audio(self, audio: Tensor, crop_size: int) -> Tensor:
        size = audio.size(0)
        if size > crop_size:
            start = self.rng.integers(0, size - crop_size + 1)
            return audio[start : start + crop_size]
        return audio

    def crop_audios_in_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        min_audio_len_batch = min(
            (item["audio"]["data"]["waveform"].size(0) for item in batch)
        )
        crop_size = min(self.max_audio_len, min_audio_len_batch)
        for item in batch:
            item["audio"]["data"]["waveform"] = self.crop_audio(
                item["audio"]["data"]["waveform"], crop_size
            )
        return batch


@final
class MmsSpeechDataset(SpeechDataset):
    _dataset_name: str
    _manifest_dir: Path
    _splits: set[str]

    def __init__(self, manifest_dir: Path, splits: set[str]) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param manifest_dir:
            The directory under which the manifest files resides.
        """
        self._manifest_dir = manifest_dir
        self._splits = splits

    @classmethod
    def from_path(cls, path: Path) -> GenericSpeechDataset:
        """Load a :class:`GenericSpeechDataset` from ``path``."""
        path = path.expanduser().resolve()

        if not path.is_dir():
            return MmsSpeechDataset(manifest_dir=path.parent, splits={path.stem})

        try:
            splits = {f.stem for f in path.glob("*.tsv")}
        except OSError as ex:
            raise RuntimeError(
                "The splits cannot be determined. See nested exception for details."
            ) from ex

        return MmsSpeechDataset(path, splits)

    @override
    def create_reader(
        self,
        split: str,
        gang: Gang,
        max_audio_len: int,
        batching: Batching,
        *,
        dtype: DataType = torch.float32,
        min_audio_len: int = 1,
        normalize_audio: bool = False,
        example_shuffle_window: int = 1,
        batch_shuffle_window: int = 1,
        is_binarized: bool = False,
        drop_remainder: bool = False,
        sync_batches: bool = True,
        max_num_batches: int | None = None,
        num_accumulate: int = 1,
        num_prefetch: int = 1,
        seed: int = 2,
        bsz_mult: int = 8,
        cached_fd_count: int = 1000,
        **extras: Any,
    ) -> DataPipelineReader[SequenceBatch]:
        """
        :param cached_fd_count:
            The maximum number of file descriptors to keep open while reading
            audio files.
        """
        if split not in self._splits:
            raise ValueError(
                f"`split` must be one of the following splits, but is '{split}' instead: {', '.join(sorted(self._splits))}"
            )

        if is_binarized:
            symbols = ["<s>", "<pad>", "</s>", "<unk>"]
            with open(self._manifest_dir.joinpath("dict.txt"), "r") as f:
                for line in f:
                    word, _ = line.split(maxsplit=1)
                    symbols.append(word)

            file_paths = MMapIndexedDataset(self._manifest_dir.joinpath(split))
            log.info("loaded {} binarized examples for {}", len(file_paths), split)
            sizes = []
            with open(self._manifest_dir.joinpath(f"{split}.lengths"), "r") as f:
                for line in f:
                    size = int(line.rstrip())
                    assert size >= min_audio_len, f"{size} less than min_audio_len."
                    sizes.append(size)
            sizes = np.ascontiguousarray(sizes, dtype=np.uint32)
            # Cap sizes by max_audio_len.
            sizes = np.minimum(sizes, max_audio_len)
            root_path = self._manifest_dir.joinpath(f"{split}.root")
            if root_path.exists():
                with open(root_path, "r") as f:
                    audio_dir = Path(next(f).strip())
            else:
                audio_dir = None
        else:
            audio_dir = self._retrieve_data_directory(split)
            builder = self._build_manifest_pipeline(
                split, max_audio_len, min_audio_len, audio_dir
            )
            manifest = list(builder.and_return())
            sizes = np.ascontiguousarray(
                [sample["audio_size"] for sample in manifest], dtype=np.uint32
            )
            file_paths = [sample["audio"] for sample in manifest]
            log.info("loaded {} examples for {}", len(file_paths), split)

        rng = np.random.default_rng(seed)
        seed += 1
        indices = np.lexsort((rng.permutation(len(sizes)), sizes))[::-1]
        log.info("Sorted the manifest.")

        if isinstance(batching, LengthBatching):
            indices = np.ascontiguousarray(indices, dtype=np.uint32)
            num_tokens_vec = np.take(sizes, indices)
            batches = batch_by_size_vec(
                indices,
                num_tokens_vec,
                batching.max_num_elements,
                -1,
                bsz_mult,
            )
            log.info("Created {} batches.", split)
        else:
            raise RuntimeError(f"`{batching}` is not supported.")

        # Shuffle buckets.
        if batch_shuffle_window != 1:
            rng = np.random.default_rng(seed)
            seed += 1
            rng.shuffle(batches)
            log.info("Shuffled {} batches.", len(batches))

        builder = read_sequence(batches)

        # Shard.
        builder.shard(gang.rank, gang.size, allow_uneven=True)

        seed += gang.rank

        if is_binarized:

            def get_paths(batch: list[np.ndarray]) -> list[dict[str, str]]:
                batched_paths = []
                for index in batch:
                    file_path_tokens = file_paths[index].tolist()
                    file_path = "".join(
                        [
                            symbols[token]
                            for token in file_path_tokens
                            if token not in (0, 2)
                        ]
                    )
                    batched_paths.append({"audio": file_path})
                return batched_paths

            builder.map(get_paths, num_parallel_calls=npc)
        else:
            builder.map(
                lambda batch: [{"audio": file_paths[index]} for index in batch],
                num_parallel_calls=npc,
            )

        # Memory map audio files.
        file_mapper = FileMapper(audio_dir, cached_fd_count=cached_fd_count)

        builder.map(file_mapper, selector="[*].audio")

        # Decode audio.
        audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)

        builder.map(audio_decoder, selector="[*].audio.data")

        # Normalize audio if requested.
        def normalize(waveform: Tensor) -> Tensor:
            with torch.no_grad():
                waveform = layer_norm(waveform, waveform.shape)

            return waveform.to(dtype)

        if normalize_audio:
            builder.map(normalize, selector="[*].audio.data.waveform")

        rng = np.random.default_rng(seed)
        seed += 1

        audio_cropper = AudioCropper(max_audio_len, rng)

        builder.map(audio_cropper.crop_audios_in_batch)

        collater = Collater()

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        def to_batch(example: dict[str, Any]) -> SequenceBatch:
            seqs = example["audio"]["data"]["waveform"].to(gang.device)

            return SequenceBatch(seqs, None, example=example)

        pipeline = builder.map(to_batch).and_return()

        return DataPipelineReader[SequenceBatch](
            pipeline,
            gang,
            num_accumulate=num_accumulate,
            drop_remainder=drop_remainder,
            sync_batches=sync_batches,
        )

    def _retrieve_data_directory(self, split: str) -> Path | None:
        manifest_file = self._manifest_dir.joinpath(f"{split}.tsv")
        try:
            with manifest_file.open() as fp:
                header = fp.readline().rstrip()
        except OSError as ex:
            raise DatasetError(
                f"{manifest_file} cannot be read. See nested exception for details."
            ) from ex

        try:
            audio_dir = Path(header)
            if audio_dir.exists():
                return audio_dir
            return None
        except ValueError:
            raise DatasetError(
                f"The first line of {manifest_file} must point to a data directory."
            ) from None

    def _build_manifest_pipeline(
        self, split: str, max_audio_len: int, min_audio_len: int, audio_dir: Path | None
    ) -> DataPipelineBuilder:
        tsv_file = self._manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(tsv_file, rtrim=True, memory_map=True)

        if audio_dir is not None:
            builder.skip(1)  # Path to the data directory.

        field_splitter = StrSplitter(names=["audio", "audio_size"])

        builder.map(field_splitter, num_parallel_calls=npc)

        # Cap audio sizes by max_audio_len.
        builder.map(lambda x: min(int(x), max_audio_len), selector="audio_size")

        builder.filter(lambda sample: sample["audio_size"] >= min_audio_len)

        return builder

    @override
    def splits(self) -> set[str]:
        return self._splits


@final
class MmsSpeechDatasetLoader(AbstractDatasetLoader[MmsSpeechDataset]):
    @override
    def _load(self, path: Path, card: AssetCard) -> MmsSpeechDataset:
        try:
            return MmsSpeechDataset.from_path(path)
        except RuntimeError as ex:
            raise AssetError(
                f"{card.name} cannot be loaded. See nested exception for details."
            ) from ex


load_mms_speech_dataset = MmsSpeechDatasetLoader()

load_speech_dataset.register("mms_speech", load_mms_speech_dataset)
