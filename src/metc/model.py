"""Offline token-classification inference for the released METC checkpoint."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from .errors import (
    CheckpointContractError,
    MarkerError,
    PredictionCountError,
)
from .checkpoint import (
    EXPECTED_PARAMETER_COUNT,
    checkpoint_inventory_sha256,
    verify_checkpoint_files,
)
from .mapping import EXPECTED_LABEL_COUNT, load_label_mapping
from .types import EditPrediction

EDIT_OPEN = "[EDIT]"
EDIT_CLOSE = "[/EDIT]"
EXPECTED_EDIT_OPEN_TOKEN_ID = 256000
EXPECTED_EDIT_CLOSE_TOKEN_ID = 256001
DEFAULT_MAX_SEQUENCE_LENGTH = 512
MAX_CHARACTERS_PER_TOKEN = 7
FALLBACK_LABEL_ID = 23
INFERENCE_PROFILES = ("portable", "paper")
_EDIT_PATTERN = re.compile(r"\[EDIT\](.*?)\[/EDIT\]", re.DOTALL)


def count_well_formed_edits(annotated: str) -> int:
    """Count edit spans and reject unbalanced or nested marker sequences."""

    opens = annotated.count(EDIT_OPEN)
    closes = annotated.count(EDIT_CLOSE)
    matches = len(_EDIT_PATTERN.findall(annotated))
    if opens != closes or matches != opens:
        raise MarkerError("malformed edit markers: expected non-nested [EDIT]...[/EDIT] pairs")
    return matches


class TokenClassifier:
    """Run the 42-way METC classifier at ``[EDIT]`` token positions.

    The checkpoint is loaded only from local files. A model and tokenizer may
    be injected for unit testing; release use should pass ``model_path``.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        device: str = "auto",
        inference_profile: str = "portable",
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if (model is None) != (tokenizer is None):
            raise ValueError("model and tokenizer must be supplied together")
        if model is None and model_path is None:
            raise ValueError("model_path is required unless model and tokenizer are injected")
        if inference_profile not in INFERENCE_PROFILES:
            raise ValueError(f"inference_profile must be one of {', '.join(INFERENCE_PROFILES)}")

        self.model_path = Path(model_path).expanduser().resolve() if model_path else None
        self.max_sequence_length = max_sequence_length
        self.pretokenization_character_limit = (
            max_sequence_length * MAX_CHARACTERS_PER_TOKEN
        )
        self.device = self._resolve_device(device)
        self.inference_profile = inference_profile
        self.fixed_length_padding = inference_profile == "paper"
        self.reference_compile = inference_profile == "paper" or self.device == "cuda"
        self.precision = "bfloat16" if inference_profile == "paper" else "float32"
        self.attention_implementation = (
            "flash_attention_2" if inference_profile == "paper" else "default"
        )
        if inference_profile == "paper":
            if self.device != "cuda":
                raise CheckpointContractError(
                    "the paper inference profile requires an available CUDA device"
                )
            if max_sequence_length != DEFAULT_MAX_SEQUENCE_LENGTH:
                raise CheckpointContractError(
                    "the paper inference profile requires max_sequence_length=512"
                )
        self.mapping = load_label_mapping()
        if self.mapping.get(FALLBACK_LABEL_ID) != ("Other", "Compound_Complex"):
            raise CheckpointContractError(
                "label ID 23 must map to Other/Compound_Complex for serving fallback"
            )
        self.checkpoint_inventory_sha256 = checkpoint_inventory_sha256()

        if model is None:
            model, tokenizer = self._load_local_checkpoint(
                self.model_path,
                inference_profile=self.inference_profile,
                reference_compile=self.reference_compile,
            )
        self.model = model
        self.tokenizer = tokenizer
        self._validate_contract()
        if self.model_path is not None:
            parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
            if parameter_count != EXPECTED_PARAMETER_COUNT:
                raise CheckpointContractError(
                    f"checkpoint has {parameter_count} parameters; "
                    f"expected {EXPECTED_PARAMETER_COUNT}"
                )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(requested: str) -> str:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - installation failure
            raise CheckpointContractError("PyTorch is required for inference") from error
        if requested not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of auto, cpu, cuda, or mps")
        if requested == "cpu":
            return "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise CheckpointContractError("CUDA was requested but is not available")
            return "cuda"
        if requested == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise CheckpointContractError("MPS was requested but is not available")
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_local_checkpoint(
        model_path: Path | None,
        *,
        inference_profile: str,
        reference_compile: bool,
    ) -> tuple[Any, Any]:
        assert model_path is not None
        if not model_path.is_dir():
            raise CheckpointContractError(f"checkpoint directory does not exist: {model_path}")
        unsafe_weights = sorted(model_path.glob("*.bin"))
        if unsafe_weights:
            names = ", ".join(path.name for path in unsafe_weights)
            raise CheckpointContractError(
                f"pickle-based checkpoint files are not accepted; use safetensors: {names}"
            )
        if not list(model_path.glob("*.safetensors")):
            raise CheckpointContractError("checkpoint contains no safetensors weights")
        for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (model_path / required).is_file():
                if required.startswith("tokenizer"):
                    raise CheckpointContractError(
                        f"checkpoint is missing {required}; run "
                        f"`metc prepare-tokenizer --model {model_path}` first"
                    )
                raise CheckpointContractError(f"checkpoint is missing {required}")
        verify_checkpoint_files(model_path)
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as error:  # pragma: no cover - installation failure
            raise CheckpointContractError("transformers is required for inference") from error

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "reference_compile": reference_compile,
        }
        if inference_profile == "paper":
            import torch

            model_kwargs.update(
                {
                    "device_map": "cuda",
                    "dtype": torch.bfloat16,
                    "attn_implementation": "flash_attention_2",
                }
            )
        try:
            model = AutoModelForTokenClassification.from_pretrained(str(model_path), **model_kwargs)
        except ImportError as error:
            if inference_profile == "paper":
                raise CheckpointContractError(
                    "the paper inference profile requires the optional CUDA, "
                    "Accelerate, and FlashAttention dependencies"
                ) from error
            raise
        return model, tokenizer

    def _marker_id(self, marker: str) -> int:
        encoded = self.tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded) != 1:
            raise CheckpointContractError(f"{marker} must tokenize as one token, got {encoded}")
        marker_id = int(encoded[0])
        unknown_id = getattr(self.tokenizer, "unk_token_id", None)
        if unknown_id is not None and marker_id == unknown_id:
            raise CheckpointContractError(f"{marker} resolves to the unknown token")
        return marker_id

    def _validate_contract(self) -> None:
        config = getattr(self.model, "config", None)
        if config is None:
            raise CheckpointContractError("checkpoint model has no config")
        num_labels = int(getattr(config, "num_labels", -1))
        if num_labels != EXPECTED_LABEL_COUNT:
            raise CheckpointContractError(
                f"checkpoint must expose {EXPECTED_LABEL_COUNT} labels, got {num_labels}"
            )
        architectures = getattr(config, "architectures", None)
        if architectures and "ModernBertForTokenClassification" not in architectures:
            raise CheckpointContractError(
                "checkpoint architecture must be ModernBertForTokenClassification, got "
                f"{architectures}"
            )

        self.edit_open_token_id = self._marker_id(EDIT_OPEN)
        self.edit_close_token_id = self._marker_id(EDIT_CLOSE)
        if self.edit_open_token_id == self.edit_close_token_id:
            raise CheckpointContractError("opening and closing edit markers share a token ID")
        if self.edit_open_token_id != EXPECTED_EDIT_OPEN_TOKEN_ID:
            raise CheckpointContractError(
                f"[EDIT] token ID is {self.edit_open_token_id}; expected "
                f"{EXPECTED_EDIT_OPEN_TOKEN_ID} from the released checkpoint"
            )
        if self.edit_close_token_id != EXPECTED_EDIT_CLOSE_TOKEN_ID:
            raise CheckpointContractError(
                f"[/EDIT] token ID is {self.edit_close_token_id}; expected "
                f"{EXPECTED_EDIT_CLOSE_TOKEN_ID} from the released checkpoint"
            )

        max_positions = getattr(config, "max_position_embeddings", None)
        if max_positions is not None and int(max_positions) < self.max_sequence_length:
            raise CheckpointContractError(
                f"checkpoint capacity {max_positions} is below requested inference limit "
                f"{self.max_sequence_length}"
            )
        pad_id = getattr(config, "pad_token_id", None)
        tokenizer_pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None and tokenizer_pad_id is None:
            raise CheckpointContractError("checkpoint and tokenizer have no pad token ID")
        if pad_id is not None and tokenizer_pad_id is not None and pad_id != tokenizer_pad_id:
            raise CheckpointContractError(
                f"model pad token ID {pad_id} differs from tokenizer ID {tokenizer_pad_id}"
            )
        if getattr(self.tokenizer, "truncation_side", None) != "right":
            raise CheckpointContractError(
                "the evaluated tokenizer contract requires right-side truncation"
            )
        if getattr(self.tokenizer, "padding_side", None) != "right":
            raise CheckpointContractError(
                "the evaluated tokenizer contract requires right-side padding"
            )

    def inspect_checkpoint(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "model_path": str(self.model_path) if self.model_path else None,
            "architecture": type(self.model).__name__,
            "num_labels": int(config.num_labels),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "inference_max_sequence_length": self.max_sequence_length,
            "pretokenization_character_limit": self.pretokenization_character_limit,
            "edit_open_token_id": self.edit_open_token_id,
            "edit_close_token_id": self.edit_close_token_id,
            "device": self.device,
            "inference_profile": self.inference_profile,
            "precision": self.precision,
            "attention_implementation": self.attention_implementation,
            "padding": "max_length" if self.fixed_length_padding else "longest_in_batch",
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
            "truncated_edit_fallback": "Other/Compound_Complex",
            "reference_compile": self.reference_compile,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
        }

    def predict_marked(
        self, annotated: str, expected_edit_count: int | None = None
    ) -> list[EditPrediction]:
        return self.predict_marked_batch(
            [annotated],
            expected_edit_counts=(
                [expected_edit_count] if expected_edit_count is not None else None
            ),
        )[0]

    def predict_marked_batch(
        self,
        annotated_texts: Sequence[str],
        expected_edit_counts: Sequence[int] | None = None,
    ) -> list[list[EditPrediction]]:
        if not annotated_texts:
            return []

        parsed_counts = [count_well_formed_edits(text) for text in annotated_texts]
        if expected_edit_counts is None:
            expected = parsed_counts
        else:
            expected = list(expected_edit_counts)
            if len(expected) != len(annotated_texts):
                raise PredictionCountError(
                    "expected_edit_counts length does not match the input batch"
                )
            for index, (parsed, declared) in enumerate(zip(parsed_counts, expected, strict=True)):
                if parsed != declared:
                    raise PredictionCountError(
                        f"input {index} contains {parsed} marked edits, expected {declared}"
                    )

        # The evaluated endpoint bounded request work before tokenization at
        # seven Unicode code points per configured token. The evaluated serving
        # path retained the model predictions for surviving [EDIT] markers and
        # assigned Other/Compound_Complex to any trailing edits whose opening
        # marker was truncated.
        serving_texts = [
            text[: self.pretokenization_character_limit] for text in annotated_texts
        ]
        encoded = self.tokenizer(
            serving_texts,
            max_length=self.max_sequence_length,
            truncation=True,
            padding="max_length" if self.fixed_length_padding else True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = input_ids.ne(int(self.tokenizer.pad_token_id)).long()

        import torch
        import torch.nn.functional as functional

        input_ids_device = input_ids.to(self.device)
        model_inputs: dict[str, Any] = {"input_ids": input_ids_device}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask.to(self.device)
        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
        expected_shape = (*input_ids_device.shape, EXPECTED_LABEL_COUNT)
        if tuple(logits.shape) != expected_shape:
            raise CheckpointContractError(
                f"token-classification logits have shape {tuple(logits.shape)}, expected "
                f"{expected_shape}"
            )

        edit_mask = input_ids_device == self.edit_open_token_id
        batch_index, sequence_index = edit_mask.nonzero(as_tuple=True)
        edit_logits = logits[batch_index, sequence_index]
        probabilities = functional.softmax(edit_logits, dim=-1)
        top_score, top_id = probabilities.max(dim=-1)

        counts = torch.bincount(batch_index, minlength=len(annotated_texts)).tolist()
        all_ids = top_id.to("cpu").tolist()
        all_scores = top_score.float().to("cpu").tolist()
        results: list[list[EditPrediction]] = []
        cursor = 0
        for index, retained_count in enumerate(counts):
            if retained_count > expected[index]:
                raise PredictionCountError(
                    f"input {index} produced {retained_count} predictions, "
                    f"expected at most {expected[index]}"
                )
            predictions: list[EditPrediction] = []
            for label_id, score in zip(
                all_ids[cursor : cursor + retained_count],
                all_scores[cursor : cursor + retained_count],
                strict=True,
            ):
                try:
                    label, sublabel = self.mapping[int(label_id)]
                except KeyError as error:
                    raise CheckpointContractError(
                        f"model emitted unknown label ID {label_id}"
                    ) from error
                predictions.append(
                    EditPrediction(
                        label_id=int(label_id),
                        label=label,
                        sublabel=sublabel,
                        score=float(score),
                    )
                )
            cursor += retained_count
            fallback_label, fallback_sublabel = self.mapping[FALLBACK_LABEL_ID]
            predictions.extend(
                EditPrediction(
                    label_id=FALLBACK_LABEL_ID,
                    label=fallback_label,
                    sublabel=fallback_sublabel,
                    # The serving path's missing-prediction fallback has no model
                    # confidence. Zero is a synthetic sentinel and is not
                    # included in the semantic evaluation artifacts.
                    score=0.0,
                )
                for _ in range(expected[index] - retained_count)
            )
            results.append(predictions)
        if cursor != len(all_ids):
            raise PredictionCountError("not all model predictions were assigned to inputs")
        return results
