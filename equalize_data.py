import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from skimage import exposure
from tqdm import tqdm


DEFAULT_DATA_DIR = Path("/grand/insitu/cohanlon/datasets/raw")


@dataclass(frozen=True)
class VolumeMetadata:
    """Parsed information extracted from a dataset filename."""

    dataset: str
    dims: Tuple[int, int, int]
    dtype_str: str
    size_token: str

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)


def parse_metadata(file_path: Path) -> VolumeMetadata:
    """Split ``<name>_<dims>_<dtype>.raw`` into its components."""

    try:
        base, size_token, dtype_token = file_path.stem.rsplit("_", maxsplit=2)
    except ValueError as exc:  # pragma: no cover - defensive parsing
        raise ValueError(
            f"Cannot parse dimensions/dtype from {file_path.name}; expected '<name>_<dims>_<dtype>.raw'."
        ) from exc

    dims = tuple(int(part) for part in size_token.split("x"))
    if len(dims) != 3:
        raise ValueError(
            f"Expected 3 dimensions in '{size_token}' extracted from {file_path.name}."
        )

    return VolumeMetadata(
        dataset=base,
        dims=dims,  # type: ignore[arg-type]
        dtype_str=dtype_token,
        size_token=size_token,
    )


def load_volume(file_path: Path, metadata: VolumeMetadata) -> np.ndarray:
    """Load the raw volume and reshape it according to the parsed metadata."""

    np_dtype = metadata.np_dtype
    data = np.fromfile(file_path, dtype=np_dtype)
    expected_elems = np.prod(metadata.dims)
    if data.size != expected_elems:
        raise ValueError(
            f"File {file_path} contains {data.size} elements but {expected_elems} were expected for dims {metadata.dims}."
        )

    return data.reshape(metadata.dims)


def equalize_volume(volume: np.ndarray) -> np.ndarray:
    """Apply histogram equalization in-place-safe fashion."""

    volume = volume.astype(np.float32, copy=False)
    v_min = float(volume.min())
    v_max = float(volume.max())
    if np.isclose(v_max, v_min):
        return np.full_like(volume, fill_value=v_min, dtype=np.float32)

    normalized = (volume - v_min) / (v_max - v_min)
    equalized = exposure.equalize_hist(normalized)
    return equalized * (v_max - v_min) + v_min


def cast_to_dtype(volume: np.ndarray, metadata: VolumeMetadata) -> np.ndarray:
    """Project the equalized float data back to the original dtype."""

    target_dtype = metadata.np_dtype
    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        clipped = np.clip(volume, info.min, info.max)
        return np.rint(clipped).astype(target_dtype)

    if np.issubdtype(target_dtype, np.floating):
        return volume.astype(target_dtype)

    raise TypeError(f"Unsupported dtype '{metadata.dtype_str}' in {metadata.dataset}.")


def build_output_path(
    metadata: VolumeMetadata, output_dir: Path, original_suffix: str
) -> Path:
    stem = f"equalized_{metadata.dataset}_{metadata.size_token}_{metadata.dtype_str}"
    return output_dir / f"{stem}{original_suffix}"


def iter_target_files(
    data_dir: Path, datasets: Optional[Sequence[str]]
) -> Iterable[Path]:
    files = sorted(data_dir.glob("*.raw"))
    wanted = set(datasets)
    return [fp for fp in files if parse_metadata(fp).dataset in wanted]


def process_file(
    file_path: Path,
    metadata: VolumeMetadata,
    output_dir: Path,
) -> Optional[Path]:
    output_path = build_output_path(metadata, output_dir, file_path.suffix)
    volume = load_volume(file_path, metadata)
    equalized = equalize_volume(volume)
    equalized = cast_to_dtype(equalized, metadata)
    equalized.tofile(output_path)
    
    return output_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing *.raw volumes.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional dataset basenames to process (matches the leading token before dimensions).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory {args.data_dir} does not exist.")

    target_files = iter_target_files(args.data_dir, args.datasets)

    for file_path in tqdm(target_files, desc="Equalizing volumes", unit="file"):
        metadata = parse_metadata(file_path)
        process_file(
            file_path=file_path,
            metadata=metadata,
            output_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
