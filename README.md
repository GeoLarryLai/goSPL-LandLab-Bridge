# goSPL–Landlab bridge

![PyPI version](https://img.shields.io/pypi/v/gospl-landlab.svg) ![PyPI downloads](https://img.shields.io/pypi/dm/gospl-landlab.svg)

v0.0.1

This package lets you run Landlab erosion tools inside a goSPL model, without changing goSPL or Landlab themselves.

goSPL still owns the mesh, time loop, rivers, sediment routing, tectonics, and outputs. The demo uses an imposed mountain-uplift and basin-subsidence map. For river cutting, hillslope creep, or landslides, a slot can point at a built-in goSPL law or a Landlab component.

> [!NOTE]
> This work resulted from an ongoing collaboration with **Dr. Xuesong Ding** at the Bureau of Economic Geology, The University of Texas at Austin.

## Comparison video

https://github.com/user-attachments/assets/fc4a3a02-10cc-48bb-a6fb-de4a25fe8593

The same clip is also in [`docs/media/compare_combo.mp4`](docs/media/compare_combo.mp4).

The question behind the clip is whether basin stratigraphy can record a change in mountain topography and in the surface processes that shape it. The two runs share the same uplift and subsidence. The domain is 60×40 km over 4 Myr.

Run A uses goSPL’s river incision and linear hillslope creep. Run B keeps that river incision, and adds Landlab’s soil-depth-dependent hillslope diffusion and the HyLands bedrock landslide model. Sediment eroded in the mountains is passed into the basin fill. The landslides arrive as pulses, so the mountain and the deep-water floor change in steps that Run A does not show.

## Installation

```bash
conda env create -f environment.yml
conda activate gospl-landlab
pip install -e .
```

On a Mac, call this before `from gospl.model import Model`:

```python
from gospl_landlab import install_vtk_shim
install_vtk_shim(force=True)
```

goSPL is pinned to 2026.7.14 from the [geodels channel](https://gospl.readthedocs.io/en/latest/getting_started/installConda.html). A goSPL-only env is in `environment_gospl.yml`.

## Citation

Lai, L.S.-H. (2026) goSPL-LandLab-Bridge.

Contact: larrysyuhenglai@gmail.com

- goSPL: Salles et al. (2020), JOSS
- Landlab: Hobley et al. (2017); Barnhart et al. (2020)
- SPACE / HyLands: Shobe et al. (2017); Campforts et al. (2020)

## Key Features

- Three slots, `fluvial`, `hillslope`, and `mass_wasting`, each set to `gospl.*` or `landlab.<Component>`.
- Push the goSPL mesh to a Landlab raster and pull Δz and Δsoil back.
- Sediment from that change goes into goSPL stratigraphy.
- Landlab work stays on MPI rank 0.

## Example notebooks

[`goSPL-LandLab-Bridge_1_compare_SPL_vs_HyLands.ipynb`](goSPL-LandLab-Bridge_1_compare_SPL_vs_HyLands.ipynb)

Saved model files under `examples/outputs/` are local only. They are not in the repository.

```bash
python -m gospl_landlab.mesh_builder
python examples/run_compare.py both
python examples/make_compare_plots.py
```

## Package structure

```
gospl_landlab/
    config.py runner.py mirror.py flow_adapter.py
    loader.py capability.py sediment.py mpi_io.py
    gospl_patches.py vtk_shim.py landlab_dtype.py
    progress.py threads.py postprocess.py
examples/
docs/media/compare_combo.mp4
tests/test_bridge_unit.py
```

Function list: `FUNCTION_DOCUMENTATION.md`.

## Choosing pieces

| Slot | What it controls | Examples |
|------|------------------|----------|
| `fluvial` | River erosion | `gospl.spl`, `landlab.SpaceLargeScaleEroder` |
| `hillslope` | Creep on slopes | `gospl.linear_diffusion`, `landlab.LinearDiffuser` |
| `mass_wasting` | Landslides | `landlab.BedrockLandslider` |

| Setting | Meaning |
|---------|---------|
| `flow: gospl` | Reuse goSPL’s river network |
| `flow: landlab` | Landlab flow on the raster. Demo Run B does this for DDTD and the landslider |

Demo Run B keeps rivers on goSPL stream power. After that, elevation is pushed to a regular raster. Landlab DDTD, a D8 flow accumulator, and BedrockLandslider run there. The bridge writes back Δz and Δsoil.

```python
from gospl_landlab import probe_component
print(probe_component("BedrockLandslider"))
```

`supported` means the bridge expects it to work. `needs_landlab_flow` means set `flow: landlab`.

## Dependencies

Requires Python >=3.10.

Installed with the package: numpy, scipy, landlab >=2.6, joblib, imageio, imageio-ffmpeg, h5py, pandas, matplotlib, tqdm, pyyaml, threadpoolctl, pillow.

Optional: `gospl`, `mpi4py`, `petsc4py`.

## Common problems

**goSPL import crashes.** Call `install_vtk_shim(force=True)` first.

**PriorityFlood crashes on some Macs.** The demo does not use it. Rivers use goSPL stream power. The hillslope and landslide stack uses Landlab `FlowAccumulator`.

**Output folders named `runA_gospl_spl_0`, `_1`, …** goSPL does that when `makedir: True`. The demo files use `makedir: False`.

**Large meshes.** Landlab runs on MPI rank 0 after the surface is gathered.

## License

goSPL–Landlab bridge is licensed under the GNU Affero General Public License 3.0 or later. See [LICENSE](LICENSE).

goSPL is GPL-3.0. Landlab is MIT. The pit-fill change in `gospl_patches.py` is adapted from goSPL, and that copyright stays with the goSPL authors.
