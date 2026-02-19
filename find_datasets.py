import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from samplers import VolumeSampler
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

data_dir = Path("/grand/insitu/cohanlon/datasets/raw")
output_dir = Path("./dataset_hists")
output_dir.mkdir(exist_ok=True)

datasets_to_consider = [
    "chameleon",
    "kingsnake",
    "pawpawsaurus",
    "beechnut",
    "hcci_oh",
    "magnetic_reconnection",
    "miranda",
    "prone",
    "vertebra",
    "mrt_angio",
]

datasets_to_consider = {}
for potential_dataset in tqdm(list(data_dir.glob("*.raw"))):
    if any(potential_dataset.stem.startswith(d) for d in datasets_to_consider):
        continue  # skip datasets we know

    size_str, type_str = potential_dataset.stem.split("_")[-2:]
    sampler = VolumeSampler(
        dims=tuple(map(int, size_str.split("x"))), dtype=type_str, device="cpu"
    )
    try:
        sampler.load_from_file(filename=str(potential_dataset))
    except Exception as e:
        print(f"Error loading {potential_dataset.stem}: {e}")
        continue
    data = sampler.gt.data.numpy()

    # Make histogram of dataset
    hist, bin_edges = np.histogram(data.flatten(), bins=100)

    # If the histogram has a spike, skip this dataset
    if np.max(hist) > 0.3 * np.sum(hist):
        continue

    print("Potential dataset:", potential_dataset.stem)
