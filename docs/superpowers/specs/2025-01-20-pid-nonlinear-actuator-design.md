# PID Controller Verification ¡ª Nonlinear Actuator Extension

**Date:** 2025-01-20 | **Status:** approved

## Goal

Extend `pid_verify.py` with composable actuator nonlinearities: dead zone, saturation, rate limiter, backlash.

## Architecture

```
PID ¡ú NonlinearActuator ¡ú Plant
        ©À©¤ Dead Zone
        ©À©¤ Saturation
        ©À©¤ Rate Limiter
        ©¸©¤ Backlash
```

Each block independently toggled. All-off = identical to current script.

## New Class: NonlinearActuator

| Param | Default | Meaning |
|-------|---------|---------|
| deadzone | 0.0 | `|u| < dz` ¡ú output 0 |
| u_min/max | ¡À10 | Saturation clamp |
| rate_limit | inf | Max ¦¤u per second |
| backlash_width | 0.0 | Gap width |

Processing: Dead Zone ¡ú Saturation ¡ú Rate Limiter ¡ú Backlash.

## Changes

- `simulate()`: insert `actuator.step()` between controller and plant
- `main()`: 4 scenarios ¡ª linear, dead-zone, rate-limited, combined
- Plots: add actuator I/O comparison subplot
- Backward compatible: default params ¡ú behavior unchanged

## Scenarios

1. **Linear** ¡ª all off (baseline)
2. **Dead-Zone** ¡ª `deadzone=0.1`
3. **Rate-Limited** ¡ª `rate_limit=5.0`
4. **Combined** ¡ª `deadzone=0.1, rate_limit=3.0, backlash=0.05`

## Acceptance

- Linear scenario metrics match original
- Nonlinear scenarios show plausible degradation
- Plots render without error
