import json
import os
import torchvision.transforms.transforms as T

from pathlib import Path
from PIL import Image
from safetensors.torch import load_file
from torch import Tensor
from torch.utils.data import Dataset
from typing import Optional


class SmoothStyleDataset(Dataset):
    r"""Load a split from the standalone SmoothStyle Hugging Face repository.

    Expected layout::

        <dataset_dir>/
        └── data/
            ├── train/
            │   ├── metadata.jsonl
            │   ├── content/
            │   ├── style/
            │   └── target/s01 ... target/s10/
            └── test/
                └── ...

    Every JSONL row describes one training target and contains
    ``content_file_name``, ``style_file_name``, ``target_file_name``,
    ``pair_id``, ``strength_id``, and ``strength``. Image paths are resolved
    relative to the split directory, matching the Hugging Face ImageFolder
    metadata convention.

    The returned tuple is compatible with the existing StyCtrl finetuners::

        content, style, target, strength

    If ``txt_latent_dir`` is provided, ``<pair_id>.safetensors`` is loaded and
    the return value additionally contains ``txt_latent`` and
    ``txt_latent_mask``.

    Args:
        dataset_dir: Root of a local SmoothStyle repository checkout.
        split: Dataset split below ``data/``, normally ``train`` or ``test``.
        image_height: Output tensor height.
        image_width: Output tensor width.
        txt_latent_dir: Optional directory containing pair-level safetensors.
        target_strength: Optional subset of integer strength IDs in ``0..10``.
            Strength ``0`` uses the original content image as its target when
            the metadata does not provide an explicit zero-strength row.
            Defaults to IDs ``1..5`` for compatibility with
            :class:`StyleTransferDataset`.
    """

    MAX_STYLIZED_STRENGTH_ID = 10
    DEFAULT_TARGET_STRENGTHS = (1, 2, 3, 4, 5)
    VALID_STRENGTH_IDS = frozenset(range(MAX_STYLIZED_STRENGTH_ID + 1))
    REQUIRED_METADATA_FIELDS = frozenset(
        {
            "content_file_name",
            "style_file_name",
            "target_file_name",
            "pair_id",
            "strength_id",
            "strength",
        }
    )

    def __init__(
        self,
        dataset_dir: str,
        split: str,
        image_height: int,
        image_width: int,
        txt_latent_dir: Optional[str] = None,
        target_strength: Optional[list[int]] = None,
    ):
        super().__init__()

        if not split or Path(split).name != split:
            raise ValueError(
                f"split must be a single directory name below data/; got {split!r}."
            )
        if image_height <= 0 or image_width <= 0:
            raise ValueError(
                "image_height and image_width must both be positive; "
                f"got {(image_height, image_width)}."
            )

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.split = split
        self.split_dir = (self.dataset_dir / "data" / split).resolve()
        self.metadata_path = self.split_dir / "metadata.jsonl"

        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"SmoothStyle repository not found: {self.dataset_dir}")
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"SmoothStyle split not found: {self.split_dir}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"SmoothStyle metadata file not found: {self.metadata_path}"
            )

        if target_strength is None:
            target_strength = list(self.DEFAULT_TARGET_STRENGTHS)
        non_integer_strengths = [
            strength_id
            for strength_id in target_strength
            if isinstance(strength_id, bool) or not isinstance(strength_id, int)
        ]
        if non_integer_strengths:
            raise ValueError(
                "target_strength must contain only integer strength IDs; got "
                f"{non_integer_strengths}."
            )
        selected_strengths = set(target_strength)
        invalid_strengths = selected_strengths - self.VALID_STRENGTH_IDS
        if invalid_strengths:
            raise ValueError(
                "target_strength must contain only integer strength IDs from 0 to "
                f"{self.MAX_STYLIZED_STRENGTH_ID}; got invalid IDs: "
                f"{sorted(invalid_strengths)}."
            )
        if not selected_strengths:
            raise ValueError("target_strength must contain at least one strength ID.")

        self.txt_latent_dir = None
        if txt_latent_dir:
            self.txt_latent_dir = Path(txt_latent_dir).expanduser().resolve()
            if not self.txt_latent_dir.is_dir():
                raise FileNotFoundError(
                    f"Text latent directory not found: {self.txt_latent_dir}"
                )

        self.samples = []
        zero_strength_templates: dict[str, dict[str, object]] = {}
        explicit_zero_pairs: set[str] = set()
        with self.metadata_path.open("r", encoding="utf-8") as metadata_file:
            for line_number, line in enumerate(metadata_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {self.metadata_path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise ValueError(
                        f"Expected a JSON object at {self.metadata_path}:{line_number}."
                    )

                missing_fields = self.REQUIRED_METADATA_FIELDS - row.keys()
                if missing_fields:
                    raise ValueError(
                        f"Missing fields at {self.metadata_path}:{line_number}: "
                        f"{sorted(missing_fields)}."
                    )

                strength_id = row["strength_id"]
                if isinstance(strength_id, bool) or not isinstance(strength_id, int):
                    raise ValueError(
                        f"strength_id must be an integer at "
                        f"{self.metadata_path}:{line_number}; got {strength_id!r}."
                    )
                if strength_id not in self.VALID_STRENGTH_IDS:
                    raise ValueError(
                        f"strength_id must be within 0..10 at "
                        f"{self.metadata_path}:{line_number}; got {strength_id}."
                    )

                try:
                    strength = float(row["strength"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"strength must be numeric at "
                        f"{self.metadata_path}:{line_number}; got {row['strength']!r}."
                    ) from error
                expected_strength = strength_id / self.MAX_STYLIZED_STRENGTH_ID
                if abs(strength - expected_strength) > 1e-6:
                    raise ValueError(
                        f"strength mismatch at {self.metadata_path}:{line_number}: "
                        f"ID {strength_id} requires {expected_strength}, got {strength}."
                    )

                pair_id = row["pair_id"]
                if not isinstance(pair_id, str) or not pair_id:
                    raise ValueError(
                        f"pair_id must be a non-empty string at "
                        f"{self.metadata_path}:{line_number}."
                    )

                content_path = self._resolve_image_path(
                    row["content_file_name"], line_number
                )
                style_path = self._resolve_image_path(
                    row["style_file_name"], line_number
                )
                txt_latent_path = None
                if self.txt_latent_dir is not None:
                    txt_latent_path = self.txt_latent_dir / f"{pair_id}.safetensors"
                    if not txt_latent_path.is_file():
                        raise FileNotFoundError(
                            f"Text latent for pair {pair_id!r} not found: "
                            f"{txt_latent_path}"
                        )
                zero_strength_templates.setdefault(
                    pair_id,
                    {
                        "cnt": content_path,
                        "sty": style_path,
                        "res": content_path,
                        "strength": 0.0,
                        "txt_latent": txt_latent_path,
                    },
                )

                if strength_id == 0:
                    explicit_zero_pairs.add(pair_id)

                if strength_id in selected_strengths:
                    self.samples.append(
                        {
                            "cnt": content_path,
                            "sty": style_path,
                            "res": self._resolve_image_path(
                                row["target_file_name"], line_number
                            ),
                            "strength": strength,
                            "txt_latent": txt_latent_path,
                        }
                    )

        if 0 in selected_strengths:
            self.samples.extend(
                sample
                for pair_id, sample in zero_strength_templates.items()
                if pair_id not in explicit_zero_pairs
            )

        if not self.samples:
            raise ValueError(
                f"No samples from {self.metadata_path} match target_strength="
                f"{sorted(selected_strengths)}."
            )

        self.num_samples = len(self.samples)
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (image_height, image_width), T.InterpolationMode.BILINEAR
                ),
            ]
        )

    def _resolve_image_path(self, relative_name: object, line_number: int) -> Path:
        if not isinstance(relative_name, str) or not relative_name:
            raise ValueError(
                f"Image paths must be non-empty strings at "
                f"{self.metadata_path}:{line_number}; got {relative_name!r}."
            )
        relative_path = Path(relative_name)
        if relative_path.is_absolute():
            raise ValueError(
                f"Image paths must be relative at {self.metadata_path}:{line_number}; "
                f"got {relative_name!r}."
            )

        image_path = (self.split_dir / relative_path).resolve()
        try:
            image_path.relative_to(self.split_dir)
        except ValueError as error:
            raise ValueError(
                f"Image path escapes split directory at "
                f"{self.metadata_path}:{line_number}: {relative_name!r}."
            ) from error
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image referenced at {self.metadata_path}:{line_number} not found: "
                f"{image_path}"
            )
        return image_path

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        index = index % self.num_samples
        sample = self.samples[index]

        cnt_image = self._load_image(sample["cnt"])
        sty_image = self._load_image(sample["sty"])
        res_image = self._load_image(sample["res"])
        strength = sample["strength"]
        txt_latent = sample["txt_latent"]

        if txt_latent is not None:
            latent_and_mask = load_file(str(txt_latent))
            missing_keys = {"txt_latent", "txt_latent_mask"} - latent_and_mask.keys()
            if missing_keys:
                raise KeyError(
                    f"Missing tensors in {txt_latent}: {sorted(missing_keys)}."
                )
            return (
                cnt_image,
                sty_image,
                res_image,
                strength,
                latent_and_mask["txt_latent"],
                latent_and_mask["txt_latent_mask"].long(),
            )

        return cnt_image, sty_image, res_image, strength

    def _load_image(self, path: Path) -> Tensor:
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))
