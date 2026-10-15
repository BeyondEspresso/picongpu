# Fix TWTS centroid validation and polarization metadata

## Problem and behavior

`TWTSLaser` uses its centroid to set an analytic pulse time delay, and a separate
incident-field switching window controls when the field is injected. The common
incoming-pulse validator rejected a positive centroid y coordinate as if it
were an initially populated Gaussian pulse. A TWTS pulse with its centroid at
the focus should instead be valid and render a zero manual delay.

Validate TWTS directions locally while retaining the common normalization and
inward-propagation restrictions. Return zero for the backend's unused
`pulse_init` compatibility field, and correct the reported polarization vector
so it is transverse to the TWTS propagation direction. The C++ field functor
still receives the original scalar polarization angle. Other laser classes
continue to use their existing centroid checks.

This changes the PIConGPU TWTS implementation; it does not add a PICMI-standard
laser parameter. Callers still choose the incident-field switching window.

## Validation

From the repository root with its Python test dependencies installed:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/lib/python" \
python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_twts_laser.py \
  lib/python/test/picongpu/quick/picmi/test_gaussian_laser.py --tb=short
```

Result: **62 passed**. Tests cover positive and negative centroids, the
switching-window rendering, transverse polarization for both incidence signs,
and unchanged Gaussian validation. The setup-generation test only writes Python
input files; it does not compile or run PIConGPU.
