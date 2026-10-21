# goSPL–Landlab bridge

**gospl-landlab-bridge** runs Landlab mountain-erosion components inside a goSPL timestep, without changing goSPL or Landlab. goSPL still owns the mesh, the time loop, rivers, sediment routing, and the basin outputs.

This work resulted from an ongoing collaboration with **Dr. Xuesong Ding** at the Bureau of Economic Geology, The University of Texas at Austin.

## Requirements

- Python >= 3.10
- [Landlab](https://landlab.readthedocs.io/) (`pip install "landlab>=2.6"`)
- [goSPL](https://gospl.readthedocs.io/en/latest/getting_started/installConda.html) when you want to run a model. The bridge itself imports without it.

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
