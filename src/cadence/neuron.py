"""The neuron model: what one neuron does with what arrives over its synapses.

The graded rate model is the one every connectome lane uses. A neuron
holds a potential ``v`` and publishes an activation ``s(v)``; each step it
moves ``v`` toward the sum of its synaptic input, its stimulus, and its bias:

    v <- v + dt * ( -v + synaptic input + stimulus + bias - strength * a )
    synaptic input = sum over inbound synapses of  gain * count * sign * exp(log_gain[pre]) * s[pre]
    s(v) = rectified sigmoid, exactly zero at rest

With ``leak`` above zero a neuron below rest publishes a small negative
activation, ``-leak`` at most, so the learning rule can still tell a
silent neuron from a dead one; rest stays an exact zero. Connectome lanes
leave it at zero. ``a`` is an optional adaptation variable, one per neuron, following its own
activation with a slow time constant. It is what lets a connectome with mutual
inhibition produce rhythm instead of a fixed point. It is off by default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

__all__ = ["Adaptation", "NeuronModel", "evenodd_lipschitz"]

FAMILIES = ("sigmoid", "evenodd")


def evenodd_lipschitz(a: float, b: float) -> float:
    """``sup_v |d/dv (a tanh(v/2) + b tanh(v/2)^2)|`` in closed form.

    With ``t = tanh(v/2)`` the slope is ``(1 - t^2)(a + 2 b t) / 2``; it vanishes at
    ``t = +-1``, so the supremum sits at a root of ``3 b t^2 + a t - b = 0``. The roots
    are written without cancellation for small ``|b|``. Depends on ``|b|`` only.
    """
    a, b = abs(float(a)), abs(float(b))  # |s'| is unchanged by flipping the sign of a or b
    if b == 0.0:
        return 0.5 * a
    r = float(np.sqrt(a * a + 12.0 * b * b))
    roots = (2.0 * b / (a + r), -(a + r) / (6.0 * b))
    best = 0.0
    for t in roots:
        if -1.0 < t < 1.0:
            best = max(best, abs((1.0 - t * t) * (a + 2.0 * b * t)))
    return 0.5 * best


@lru_cache(maxsize=64)
def _per_neuron(odd: Any, even: Any, linear: tuple[int, ...], n: int) -> tuple[Any, Any, Any]:
    """Arrays (or scalars) of the even/odd parameters and the linear mask for ``n`` neurons."""
    a = np.asarray(odd, dtype=float) if isinstance(odd, tuple) else float(odd)
    b = np.asarray(even, dtype=float) if isinstance(even, tuple) else float(even)
    for x in (a, b):
        if isinstance(x, np.ndarray) and x.shape != (n,):
            raise ValueError("per-neuron odd/even parameters need one entry per neuron")
    mask = None
    if linear:
        mask = np.zeros(n, dtype=bool)
        idx = np.asarray(linear, dtype=np.int64)
        if (idx < 0).any() or (idx >= n).any():
            raise ValueError("linear neuron index out of range")
        mask[idx] = True
    return a, b, mask


@dataclass(frozen=True, slots=True)
class Adaptation:
    """Spike-frequency adaptation: ``a <- a + (s - a) / tau_steps``; drive loses ``strength * a``.

    Neuron-local: a neuron reads only its own ``a``.
    """

    tau_steps: float = 50.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite([self.tau_steps, self.strength]).all():
            raise ValueError("adaptation parameters must be finite")
        if self.tau_steps <= 0 or self.strength < 0:
            raise ValueError("tau_steps must be positive and strength nonnegative")


@dataclass(frozen=True, slots=True)
class NeuronModel:
    dt: float = 0.2
    slope: float = 4.0
    threshold: float = 1.5
    gain: float = 0.02
    stimulus_amplitude: float = 3.0
    adaptation: Adaptation | None = None
    leak: float = 0.0
    # The activation family. "sigmoid": the re-based rectified sigmoid above (default).
    # "evenodd": s(v) = odd * tanh(v/2) + even * tanh(v/2)^2, smooth, zero at rest, with an
    # even component; ``odd`` and ``even`` are scalars or one value per neuron, and the
    # neurons listed in ``linear`` publish their potential unchanged (s = v), for clamped
    # input populations that must not pass through the nonlinearity.
    family: str = "sigmoid"
    odd: float | tuple[float, ...] = 1.0
    even: float | tuple[float, ...] = 0.0
    linear: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        for name in ("odd", "even"):
            value = getattr(self, name)
            if isinstance(value, (list, np.ndarray)):
                value = tuple(float(x) for x in np.asarray(value, float).ravel())
                object.__setattr__(self, name, value)
            elif not isinstance(value, tuple):
                object.__setattr__(self, name, float(value))
            else:
                object.__setattr__(self, name, tuple(float(x) for x in value))
        lin = self.linear
        if not isinstance(lin, tuple):
            object.__setattr__(self, "linear", tuple(int(i) for i in np.asarray(lin).ravel()))
        else:
            object.__setattr__(self, "linear", tuple(int(i) for i in lin))
        if len(set(self.linear)) != len(self.linear):
            raise ValueError("linear neurons must be distinct")
        odd_values = np.atleast_1d(np.asarray(self.odd, float))
        even_values = np.atleast_1d(np.asarray(self.even, float))
        if not (np.isfinite(odd_values).all() and np.isfinite(even_values).all()):
            raise ValueError("odd and even must be finite")
        if self.family == "sigmoid":
            if self.odd != 1.0 or self.even != 0.0 or self.linear:
                raise ValueError("odd, even and linear apply to the evenodd family only")
        elif (odd_values <= 0).any():
            raise ValueError("the odd coefficient must be positive")
        if not np.isfinite(
            [self.dt, self.slope, self.threshold, self.gain, self.stimulus_amplitude, self.leak]
        ).all():
            raise ValueError("rule parameters must be finite")
        if not 0 < self.dt <= 1:
            raise ValueError("dt must lie in (0, 1]")
        if self.slope <= 0 or self.gain <= 0:
            raise ValueError("slope and gain must be positive")
        if not 0 <= self.leak <= 1:
            raise ValueError("leak must lie in [0, 1]")
        with np.errstate(over="ignore"):
            rest = self.rest_emission
        if not 0 < rest < 1:
            raise ValueError(
                "slope and threshold saturate the resting sigmoid; reduce their product's magnitude"
            )

    @property
    def rest_emission(self) -> float:
        """Activation the raw sigmoid would emit at v = 0; subtracted so rest emits nothing."""
        return float(1.0 / (1.0 + np.exp(self.slope * self.threshold)))

    def _validate_precision(self, dtype: str) -> None:
        """Reject a rule whose rebasing constants fail in a backend's precision."""
        rest = self.rest_emission
        values = [self.dt, self.slope, self.threshold, rest,
                  1.0 / (1.0 - rest), self.leak / rest]
        if self.adaptation is not None:
            values.extend([self.adaptation.tau_steps, self.adaptation.strength,
                           1.0 / self.adaptation.tau_steps])
        with np.errstate(over="ignore", under="ignore"):
            cast = np.asarray(values, dtype=dtype)
        if not np.isfinite(cast).all() or cast[0] <= 0 or cast[1] <= 0 or not 0 < cast[3] < 1:
            raise ValueError("neuron rule is not representable at runtime precision")

    def evenodd_parameters(self, n: int) -> tuple[Any, Any, Any]:
        """Per-neuron ``odd``, ``even`` (array or scalar) and the linear mask (or None)."""
        return _per_neuron(self.odd, self.even, self.linear, int(n))

    def lipschitz_per_neuron(self, n: int) -> np.ndarray:
        """Each neuron's activation Lipschitz constant (one for a linear neuron)."""
        if self.family == "sigmoid":
            rest = self.rest_emission
            return np.full(n, self.slope / 4.0 * max(1.0 / (1.0 - rest), self.leak / rest))
        a, b, mask = self.evenodd_parameters(n)
        a_arr = np.broadcast_to(np.asarray(a, float), (n,))
        b_arr = np.broadcast_to(np.asarray(b, float), (n,))
        cache: dict[tuple[float, float], float] = {}
        out = np.empty(n)
        for i in range(n):
            key = (float(a_arr[i]), float(b_arr[i]))
            if key not in cache:
                cache[key] = evenodd_lipschitz(*key)
            out[i] = cache[key]
        if mask is not None:
            out[mask] = 1.0
        return out

    def _evenodd(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        a, b, mask = self.evenodd_parameters(v.shape[-1])
        t = np.tanh(0.5 * v)
        s = t * (a + b * t)
        if mask is not None:
            s = np.where(mask, v, s)
        return s

    def _evenodd_slope(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        a, b, mask = self.evenodd_parameters(v.shape[-1])
        t = np.tanh(0.5 * v)
        d = 0.5 * (1.0 - t * t) * (a + 2.0 * b * t)
        if mask is not None:
            d = np.where(mask, 1.0, d)
        return d

    def activation(self, v: np.ndarray) -> np.ndarray:
        if self.family == "evenodd":
            return self._evenodd(v)
        rest = self.rest_emission
        r = np.exp((-self.slope) * (v - self.threshold))
        r += 1.0
        np.reciprocal(r, out=r)
        r -= rest  # zero at rest, in the array's own precision
        if self.leak == 0.0:
            np.maximum(r, 0.0, out=r)
            r *= 1.0 / (1.0 - rest)
            return r
        return np.where(r > 0.0, r * (1.0 / (1.0 - rest)), r * (self.leak / rest))

    def slope_at(self, v: np.ndarray) -> np.ndarray:
        """Derivative of the activation, for reports on how responsive neurons are."""
        if self.family == "evenodd":
            return self._evenodd_slope(v)
        rest = self.rest_emission
        sig = 1.0 / (1.0 + np.exp(-self.slope * (v - self.threshold)))
        d = self.slope * sig * (1.0 - sig)
        return np.where(sig > rest, d / (1.0 - rest), self.leak * d / rest)

    def replace(self, **changes: Any) -> NeuronModel:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["adaptation"] = None if self.adaptation is None else asdict(self.adaptation)
        if self.family == "sigmoid":  # unchanged records for the default family
            for key in ("family", "odd", "even", "linear"):
                out.pop(key, None)
        else:
            for key in ("odd", "even"):
                if isinstance(out[key], tuple):
                    out[key] = list(out[key])
            out["linear"] = list(out["linear"])
        return out
