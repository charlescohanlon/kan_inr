# Kolmogorov‑Arnold Neural Networks (KANs) for Implicit Neural Representations (INRs) of Volumetric Data

Reconstruct and analyze 3‑D volumes (e.g., CT/MRI scans, scientific simulations) with lightweight, spline‑activated Kolmogorov‑Arnold Networks.

---

## Features

- **Compact implicit representation** of 3‑D scalar fields using KANs.
- **Spline activations** for improved expressivity and interpretability.
- **Hydra‑driven experiment configuration**.
- **Jupyter demo notebook** illustrating end‑to‑end reconstruction & visualization workflow.
- **PyVista & VTK** for interactive volume rendering and iso‑surface extraction.
- **Modular training loop** built on PyTorch‑Lightning (optional).

---

## Quick Start

### 1. Clone & install

```bash
git clone git@github.com:charlescohanlon/alcf_kan_inr.git
cd alcf_kan_inr
conda env create -f environment.yml   # or micromamba
conda activate alcf_kan_inr
pip install -e .
```

> **GPU:** A CUDA‑enabled GPU is recommended for training, but the notebook can run on CPU for small volumes.

### 2. Obtain or generate a volume

You can start with any 3‑D numpy array. The repository ships with a tiny `data/toy_ct.npy` example.

```python
import numpy as np
volume = np.load("data/toy_ct.npy")   # shape (D, H, W)
```

---

## Using the `demo.ipynb` Notebook

1. Launch Jupyter:

    ```bash
    jupyter lab
    ```

2. Open `notebooks/demo.ipynb`.

3. Run the cells sequentially:

| Cell range | Purpose                                 |
|------------|-----------------------------------------|
| 0‑2        | Imports, environment checks             |
| 3          | Create model, load datasets             |
| 4-5        | Fit model: train for N epochs           |
| 6          | Reconstruct grid, evaluate PSNR/SSIM    |
| 7          | Visualize slices & iso‑surfaces         |

- Optional: Tweak hyper‑parameters in the config section (epochs, spline knots, latent size).

---

## Troubleshooting

| Symptom                        | Fix                                                         |
|---------------------------------|-------------------------------------------------------------|
| CUDA out of memory              | Reduce `batch_size` or volume crop size                     |
| vtkOpenGL errors in remote notebooks | Start Jupyter with `jupyter lab --no-browser` and use `pyvistaqt` instead of default backend |

---

## Training from the Command Line

WIP

Hydra logs configs & checkpoints under `outputs/`.

---

## Repository Structure

```
alcf_kan_inr/
├── data/              
│   └── nucleon_41x41x41_unint8.raw     # demo dataset
├── notebooks/
│   └── demo.ipynb
├── conf/           # Hydra configs
├── environment.yml
├──  README.md
|
...
```

---

## Citation

WIP

---

## License

MIT
