# goSPL–Landlab bridge

[![PyPI version](https://img.shields.io/pypi/v/gospl-landlab-bridge.svg)](https://pypi.org/project/gospl-landlab-bridge/) [![PyPI downloads](https://static.pepy.tech/badge/gospl-landlab-bridge/month)](https://pepy.tech/project/gospl-landlab-bridge)

**gospl-landlab-bridge** runs Landlab mountain-erosion components inside a goSPL timestep, without changing goSPL or Landlab. goSPL still owns the mesh, the time loop, rivers, sediment routing, and the basin outputs.

This work resulted from an ongoing collaboration with **Dr. Xuesong Ding** at the Bureau of Economic Geology, The University of Texas at Austin.

![goSPL SPL+LD vs goSPL SPL + Landlab nonlinear diffusion + HyLands](https://raw.githubusercontent.com/GeoLarryLai/goSPL-LandLab-Bridge/main/docs/media/compare_combo.gif)

Comparison of goSPL native surface processes against goSPL with Landlab's depth-dependent nonlinear diffuser and HyLands landslider (60×40 km domain, 4 Myr). Full video: [docs/media/compare_combo.mp4](https://github.com/GeoLarryLai/goSPL-LandLab-Bridge/blob/main/docs/media/compare_combo.mp4).

## Requirements

- Python >= 3.10
- [Landlab](https://landlab.readthedocs.io/en/latest/installation.html) (`pip install landlab`)
- [goSPL](https://gospl.readthedocs.io/en/v2026.7.14/getting_started/installPip.html) when you want to run a model. The bridge itself imports without it.

## Installation

```bash
pip install gospl-landlab-bridge
```

```python
import gospl_landlab_bridge
```

Python cannot import a hyphen, so the install name `gospl-landlab-bridge` is imported as `gospl_landlab_bridge`.

## Citation

Lai, L.S.-H. (2026) goSPL-LandLab-Bridge.

## Key Features

- Three slots (`fluvial`, `hillslope`, `mass_wasting`) set to a goSPL law or a Landlab component
- Push a goSPL mesh to a Landlab raster and pull elevation and soil change back
- Pass that sediment into goSPL stratigraphy

The comparison notebook is on [GitHub](https://github.com/GeoLarryLai/goSPL-LandLab-Bridge/blob/main/goSPL-LandLab-Bridge_1_compare_SPL_vs_HyLands.ipynb).

## License

GNU Affero General Public License 3.0 or later. See [LICENSE](LICENSE).

goSPL is GPL-3.0. Landlab is MIT. A small pit-fill patch in `gospl_patches.py` is adapted from goSPL.
