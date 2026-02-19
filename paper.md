# KAN-INR: Kolmogorov–Arnold Networks as Implicit Neural Representations for Scientific Volumetric Data

**Authors:** First A. Author, Second B. Author, Third C. Author  
**Affiliations:** Dept./Org, City, Country  
**Emails:** {first,last}@your.org


---

## Abstract

Implicit neural representations (INRs), also known as neural fields, are a form of lossy compression whereby a neural network is used as a continuous approximation of a function from coordinate to data value.
For volumetric data specifically, INRs encode a signal in a map from 3-space to voxel value.
Recently, the Kolmogorov-Arnold Network (KAN), which leverages the Kolmogorov-Arnold approximation theorem and univariate b-splines to fit curves, has garnered attention for its bold claims.
State-of-the-art INRs typically utilize a learned multiresolution hash encoding to emphasize frequencies at relevant scales.
KANs may be able to mimic this behaviour organically with their cubic spline basis functions, resulting in superior performance under specific conditions.
To test their effectiveness we benchmarked the KAN against state of the art SIREN and MLP baselines on open-source scientific volumetric datasets.
We report PSNR, SSIM, and MSE with mean and standard deviations for each.
We also showcase several examples reconstructed from their compressed forms.
We find that KANs match or exceed baselines, when no multiresolution hash encoding is used, for various shallow network depths.

---

**Index Terms**—Kolmogorov–Arnold Networks; Implicit Neural Representations; Scientific Visualization; Volume Rendering; Multiresolution Hash Encoding; SIREN; PSNR; SSIM. :contentReference[oaicite:7]{index=7}

---

## I. Introduction
- **Context:** INRs model continuous signals with coordinate‑based neural networks and have become a unifying representation across imaging and graphics. Position this work for **scientific** volumetric data (simulation & microscopy). :contentReference[oaicite:8]{index=8}
- **Motivation:** KANs replace linear weights with learnable univariate functions on edges (splines), potentially improving accuracy/interpretability and scaling for function approximation and PDEs—properties attractive for INRs. :contentReference[oaicite:9]{index=9}
- **Challenge:** Modern encodings (e.g., multiresolution hash grids) inject powerful learned features. However, the learned hashmap encoders are bulky, making up the majority of an INR's parameters. In scenarios such as in-situ scientific visualization, where high compression ratios are desirable, KAN becomes appealing; how do KAN INRs compare with/without such encoders on scientific volumes? :contentReference[oaicite:10]{index=10}
- **Contributions:**
  - (**C1**) A controlled INR benchmark on **Open SciVis** volumetric datasets.
  - (**C2**) KAN as an **scientific INR architecture** vs. SIREN/MLP baselines under small hidden‑width budgets and feature encoder budgets.
  - (**C3**) Analysis of **multiresolution hash encoding** (log2 sizes \(2^0\)–\(2^9\)) and its interaction with KANs. :contentReference[oaicite:11]{index=11}
  - (**C4**) Robust evaluation: **PSNR/SSIM**, **10× runs** over 5 seeds with mean±std.
  - (**C5**) **Finding:** **Without** a hash encoding, **KAN outperforms** SIREN/MLP across all swept small hidden sizes on all datasets. If we gradually increase the hash encoding size, the baseline INRs catch up to KAN INR peformance at sizes around 2^13 - 2^16 on average. This is significant because INR compression ratio is key in certain contexts, and the hash encoding makes up INR parameters
---

## II. Background & Related Work
### A. Implicit Neural Representations (INRs)
- Coordinate‑based networks that map \(\mathbf{x}\!\in\!\mathbb{R}^3\to\) density/intensity, widely used in visual computing and beyond (survey). :contentReference[oaicite:12]{index=12}

### B. SIREN and periodic activations
- SIREN uses sinusoidal activations and specialized initialization for representing signals and their derivatives; established INR baseline. :contentReference[oaicite:13]{index=13}

### C. Kolmogorov–Arnold Networks (KANs)
- KANs are inspired by the Kolmogorov–Arnold representation theorem; parameters are learnable univariate functions (e.g., splines) on edges; shown to be competitive in function fitting and PDE solving. :contentReference[oaicite:14]{index=14}
- Follow‑ups explore KANs in scientific discovery (KAN 2.0) and also provide critical assessments—cite to acknowledge the landscape. :contentReference[oaicite:15]{index=15}

### D. Learned input encodings for INRs
- Multiresolution hash encoding (Instant‑NGP) attaches trainable features in a hierarchy of grids, enabling much smaller MLP backbones without sacrificing quality and is now common in neural fields. :contentReference[oaicite:16]{index=16}

---

## III. Data: Scientific Volumetric Datasets
*State voxel dims/type only if you used the exact Open SciVis versions below; otherwise adapt.*

- **beechnut** — micro‑CT of a dried beechnut; 1024×1024×1546, uint16. :contentReference[oaicite:17]{index=17}  
- **chameleon** — CT of a chameleon; 1024×1024×1080, uint16. :contentReference[oaicite:18]{index=18}  
- **hcci_oh** — combustion simulation scalar (OH); 560³, float32. :contentReference[oaicite:19]{index=19}  
- **kingsnake** — CT of a kingsnake; 1024×1024×795, uint8. :contentReference[oaicite:20]{index=20}  
- **magnetic_reconnection** — MHD/space‑plasma simulation; 512³, float32. :contentReference[oaicite:21]{index=21}  
- **marmoset_neurons** — 2‑photon microscopy; 1024×1024×314, uint8. :contentReference[oaicite:22]{index=22}  
- **pawpawsaurus** — CT fossil skull (UTCT); 958×646×1088, uint16. :contentReference[oaicite:23]{index=23}  
- **tacc_turbulence** — turbulence simulation; 256³, float32. :contentReference[oaicite:24]{index=24}

> **Note:** These are part of the Open SciVis collection curated by Klacansky; OME‑Zarr mirrors also exist on AWS for web‑native access. Cite appropriately if using those distributions. :contentReference[oaicite:25]{index=25}

---

## IV. Methods
### A. Problem Formulation
- INR learns \(f_\theta:\mathbb{R}^3\!\to\!\mathbb{R}\) (voxel intensity) from coordinates \(\mathbf{x}\in[0,1]^3\).
- **Loss:** per‑voxel MSE; report PSNR (dB) and SSIM; define MAX\(_I\) for each dataset/normalization. :contentReference[oaicite:26]{index=26}
- PSNR, MSE calculated on entire volume, SSIM computed as average of slices (as is common in medical imaging studies, CITATION)

### B. Architectures
- **KAN (ours as INR):** brief diagram/description—depth, small hidden‑width sweep; spline order; grid of knots; regularization (if any). Cite KAN paper. :contentReference[oaicite:27]{index=27}
- **Baselines:**  
  - **SIREN:** depth/width; \(\omega_0\) initialization. :contentReference[oaicite:28]{index=28}  
  - **ReLU/MLP:** vanilla coordinate MLP with same depth/width.
- **Input encodings:** None vs. **Multiresolution Hash Encoding** (levels, per‑level feature dim, interpolation, and **log2(hash size)\(\in\{0,\dots,9\}\)** i.e., sizes \(2^0\)–\(2^9\). Put exact values you used.) :contentReference[oaicite:29]{index=29}

### C. Training Protocol
- **Sampling:** uniform/random voxel sampling per step; train/val/test slicing or masked splits (specify).
  - Input coordinates minmax normalized to [0, 1]
  - When loading, uniform sample [0, 1] then linear interpolate to normalized coordinate space
  - Voxel centered sampling
- **Optimization:** optimizer, LR schedule, batch size, epochs/steps, early stopping.
- **Compute:** GPU/CPU, memory cap, wall‑clock/time‑to‑target?
- **Stochasticity:** **10 independent runs** per setting; report mean±std, 3 seeds

### D. Evaluation Metrics
- **PSNR:** \(10\log_{10}\left(\frac{\mathrm{MAX}_I^2}{\mathrm{MSE}}\right)\).  
  - Computed over entire output volume
- **SSIM:** structural similarity; window size and implementation details.  
  - Computed as average of cross-section slices
- **MSE:** per‑voxel average squared error. :contentReference[oaicite:30]{index=30}
  - Entire output volume

---

## V. Experiments
### A. Experimental Matrix
- **Model families:** {KAN, SIREN, MLP}  
- **Hidden widths (small):** {1, 5}  *(or clarify as number of units/layers—match your setup)*  
- **Encoding:** {None, Hash encoding \(2^0 \ldots 2^9\)}  
- **Datasets:** 8 volumes listed in §III  
- **Runs:** 10 per configuration per seed --> aggregate mean±std

### B. Figures & Tables
- **Fig. 1:** Schematic: KAN (no encoder) vs. SIREN/MLP (w/ increasing hashmap size until KAN performance matched).
  - "Zero hashmap KAN"
- **Fig. 2:** Side-by-side GT, SIREN, MLP, KAN dataset reconstructions
- **Table I:** Dataset summary (dims, type, scientific domain).  
- **Fig. 3:** **Average PSNR** across datasets (bars with error bars), per model & encoding.  
- **Fig. 4:** **Average SSIM** across datasets.  
- **Fig. 5:** **Average MSE** across datasets.  
- **Table II:** Per‑dataset PSNR (mean±std) @ best validation for each method.  

---

## VI. Results
**Headline observations (TODO: add numbers):**
- **Without hash encoding, KAN > SIREN and MLP** for a hidden width of 4 layers KAN w/ no encoder able to achieve PSNR in the low-to-mid 30s (dB) across all datasets. We select a hidden width of 4 as we found it's the minimum width for SIREN to be a competitive baseline. Baseline SIREN and MLP networks of same depth achieve PSNRS of 
- **With hash encoding,** summarize whether performance gaps narrow/flip; discuss how encoding capacity interacts with backbone capacity. :contentReference[oaicite:31]{index=31}
- **Stability:** comment on run‑to‑run std; any sensitivity to initialization (esp. SIREN’s \(\omega_0\)). :contentReference[oaicite:32]{index=32}
- **Qualitative slices:** add 2–3 slice visuals per dataset (optional).

---

## VII. Discussion
- **Why might KANs help as INRs (no encoder)?** Hypothesize about edge‑function flexibility vs. node activations; approximation bias for smooth/structured scientific fields; relate to KAN claims **and** critical assessments. :contentReference[oaicite:33]{index=33}
  - Suspect cubic splines able to fit periodic signal with high granularity?
- **Effect of multiresolution hash encoding:** learned features can dominate representational capacity, reducing backbone differences; discuss compute/parameter trade‑offs. :contentReference[oaicite:34]{index=34}
- **Limitations:** small‑width regime; limited hyperparameter search; single loss; static volumes only; no compression rate analysis; potential generalization to time‑varying data.
  - Very slow

---

## VIII. Conclusion
- 1–2 sentences: recap problem & setup.  
- 1–2 sentences: decisive results (KAN advantage without encoder; interaction with hash encoding).  
- 1 sentence: outlook—KANs with other encodings/regularizers; dynamic fields; compression metrics.

---

## Acknowledgments
**TODO:** Funding, compute resources, dataset providers (Klacansky/Open SciVis; UTCT for pawpawsaurus; TACC turbulence, etc.). :contentReference[oaicite:35]{index=35}

---

## References
[1] Z. Liu *et al.*, “KAN: Kolmogorov–Arnold Networks,” *arXiv:2404.19756*, 2024. :contentReference[oaicite:36]{index=36}  
[2] V. Sitzmann *et al.*, “Implicit Neural Representations with Periodic Activation Functions,” *NeurIPS*, 2020. :contentReference[oaicite:37]{index=37}  
[3] T. Müller *et al.*, “Instant Neural Graphics Primitives with a Multiresolution Hash Encoding,” *arXiv:2201.05989*, 2022; project page. :contentReference[oaicite:38]{index=38}  
[4] B. Xie *et al.*, “Neural Fields in Visual Computing and Beyond,” *arXiv:2111.11426*, 2022. :contentReference[oaicite:39]{index=39}  
[5] Z. Wang *et al.*, “Image Quality Assessment: From Error Visibility to Structural Similarity,” *IEEE TIP*, 2004. (SSIM) :contentReference[oaicite:40]{index=40}  
[6] PSNR: standard definition (overview). :contentReference[oaicite:41]{index=41}  
[7] Mean Squared Error (MSE): definition (overview). :contentReference[oaicite:42]{index=42}  
[8] **Open SciVis datasets (examples used):** beechnut, chameleon, hcci_oh, kingsnake, magnetic_reconnection, marmoset_neurons, pawpawsaurus, tacc_turbulence (use specific pages if accessed):  
 • beechnut (metadata). :contentReference[oaicite:43]{index=43}  
 • chameleon (metadata). :contentReference[oaicite:44]{index=44}  
 • hcci_oh (metadata). :contentReference[oaicite:45]{index=45}  
 • kingsnake (metadata). :contentReference[oaicite:46]{index=46}  
 • magnetic_reconnection (metadata). :contentReference[oaicite:47]{index=47}  
 • marmoset_neurons (metadata). :contentReference[oaicite:48]{index=48}  
 • pawpawsaurus (metadata). :contentReference[oaicite:49]{index=49}  
 • tacc_turbulence (metadata). :contentReference[oaicite:50]{index=50}  
[9] Z. Liu *et al.*, “KAN 2.0: Kolmogorov–Arnold Networks Meet Science,” *arXiv:2408.10205*, 2024. :contentReference[oaicite:51]{index=51}  
[10] A. Stroev *et al.*, “Kolmogorov–Arnold Networks: A Critical Assessment of Claims,” *arXiv:2407.11075*, 2024. :contentReference[oaicite:52]{index=52}