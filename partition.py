from pathlib import Path
import numpy as np
from tqdm import tqdm

datasets = Path("/grand/insitu/cohanlon/datasets/")
raw_datasets = datasets / "raw"

partition_size = 64
partitions = datasets / "partitions"

# Make volumetric dataset partitions
for filename in tqdm(raw_datasets.iterdir()):
    if not filename.name.endswith(".raw"):
        raise ValueError(f"Unexpected file format: {filename}")
    metadata = filename.name[: -len(".raw")].split("_")
    name = "_".join(metadata[:-2])
    shape = tuple(int(dim) for dim in metadata[-2].split("x"))
    dtype = np.dtype(metadata[-1])
    arr = np.fromfile(filename, dtype=dtype).reshape(shape, order="F")
    partition_id = 0
    for i in range(0, shape[0], partition_size):
        for j in range(0, shape[1], partition_size):
            for k in range(0, shape[2], partition_size):
                partition_shape = (
                    min(partition_size, shape[0] - i),
                    min(partition_size, shape[1] - j),
                    min(partition_size, shape[2] - k),
                )
                partition = arr[
                    i : i + partition_shape[0],
                    j : j + partition_shape[1],
                    k : k + partition_shape[2],
                ]
                fname = f"{name}_{'x'.join([str(s) for s in partition_shape])}_{dtype}_id{partition_id}.raw"
                partition_id += 1
                partition_filename = partitions / fname
                partition.tofile(partition_filename)
