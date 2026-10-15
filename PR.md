# Expose PIConGPU composite particle pushers through PICMI

## Problem and behavior

PIConGPU can switch a species between two particle pushers at a specified
timestep, but PICMI could select only one pusher with `Species.method`. This
adds `picongpu_composite_pusher=picmi.CompositePusher(first=..., second=...,
switch_step=...)`. The switch step is an absolute unsigned 32-bit timestep;
the two supported pushers must differ. Supplying both this option and
`method` is rejected. Ordinary species still use Boris when `method` is omitted.

The same change corrects the generated C++ spelling of HigueraCary and removes
the unsupported Axel pusher from the accepted list. These fixes ensure the
validated names correspond to C++ types that PIConGPU actually provides.

`picongpu_composite_pusher` is a PIConGPU extension; it does not add a
PICMI-standard pusher method. The public model, backend model, species template
and documentation are included together.

## Validation

From the repository root with Python test dependencies installed:

```bash
PIC_RC=/tmp/picmi-rebundle-no-config.toml \
MPLCONFIGDIR=/tmp/picmi-rebundle-matplotlib \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/lib/python" \
python -m pytest -q -p no:cacheprovider \
  lib/python/test/picongpu/quick/picmi/test_composite_pusher.py \
  lib/python/test/picongpu/quick/picmi/test_species.py \
  lib/python/test/picongpu/quick/picmi/test_simulation.py --tb=short
```

Result: **161 passed, 24 subtests passed**. Coverage includes valid and invalid
pushers, switch boundaries, `method` conflicts, mutated/copied models, ordinary
species and generated definitions. Tests use Python input generation; no
PIConGPU compilation or simulation was performed.
