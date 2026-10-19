"""Optional classical energy refinement for a narrow reciprocal CPU model.

All neurons remain in the equilibrium. Independent, unanchored input neurons
are eliminated conditionally and reconstructed at every retained-state evaluation.
This is a global dense Newton method, not a neuron-local repair schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .brain import Brain, BrainState, Equilibrium, Nudge, RefinementReport


def _geometry(brain: Brain, input_index: np.ndarray) -> tuple[np.ndarray, ...]:
    rule = brain.neuron_model
    if brain.backend != "cpu":
        raise ValueError("hybrid refinement requires the float64 CPU backend")
    if rule.family != "sigmoid" or (rule.slope, rule.threshold, rule.leak) != (1.0, 0.0, 1.0) or rule.adaptation:
        raise ValueError("hybrid refinement requires smooth tanh(v/2), without adaptation")
    n = brain.connectome.n
    inputs = np.asarray(input_index)
    if (
        inputs.ndim != 1
        or inputs.dtype.kind not in "iu"
        or np.any(inputs < 0)
        or np.any(inputs >= n)
        or len(np.unique(inputs)) != len(inputs)
    ):
        raise ValueError("hybrid input indices must be distinct valid integers")
    inputs = inputs.astype(np.int64, copy=False)
    keep = np.ones(n, dtype=bool)
    keep[inputs] = False
    retained = np.flatnonzero(keep)
    ni, nr = len(inputs), len(retained)
    ii, rr = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int)
    ii[inputs], rr[retained] = np.arange(ni), np.arange(nr)
    pre, post, weights = brain.connectome.pre, brain.connectome.post, brain.weights
    if not np.isfinite(weights).all() or not np.isfinite(brain.bias).all():
        raise ValueError("hybrid parameters must be finite")
    within = (~keep[pre]) & (~keep[post])
    if within.any():
        _, inverse = np.unique(pre[within] * n + post[within], return_inverse=True)
        totals = np.bincount(inverse, weights=weights[within])
        if np.any(totals != 0):
            raise ValueError("hybrid eliminated inputs must have zero mutual weights")
    a, reverse, b = np.zeros((ni, nr)), np.zeros((ni, nr)), np.zeros((nr, nr))
    forward = (~keep[pre]) & keep[post]
    backward = keep[pre] & (~keep[post])
    core = keep[pre] & keep[post]
    np.add.at(a, (ii[pre[forward]], rr[post[forward]]), weights[forward])
    np.add.at(reverse, (ii[post[backward]], rr[pre[backward]]), weights[backward])
    np.add.at(b, (rr[post[core]], rr[pre[core]]), weights[core])
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("hybrid summed effective weights must be finite")
    if not np.array_equal(a, reverse) or not np.array_equal(b, b.T):
        raise ValueError("hybrid refinement requires exact reciprocal effective weights")
    return inputs, retained, a, b


def validate_hybrid(brain: Brain, input_index: np.ndarray) -> None:
    """Reject unsupported equations without changing state or parameters."""
    _geometry(brain, input_index)


def _quadratic(
    nudge: Nudge | None, shape: tuple[int, int], inputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return linear and quadratic coefficients; anchors are not beta-scaled."""
    linear, coefficient = np.zeros(shape), np.zeros(shape)
    if nudge is None:
        return linear, coefficient
    if nudge.softmax_temperature is not None:
        raise ValueError("hybrid refinement supports quadratic nudges only")
    if nudge.mask.shape != (shape[1],):
        raise ValueError("nudge mask must match the brain")
    if np.any(nudge.mask[inputs] != 0) or (
        nudge.anchor_gain is not None and np.any(nudge.anchor_gain[inputs] != 0)
    ):
        raise ValueError("hybrid nudges and anchors cannot act on eliminated inputs")
    weight = np.ones(shape[0]) if nudge.weight is None else nudge.weight
    if weight.shape != (shape[0],):
        raise ValueError("nudge weight must match the batch")
    with np.errstate(over="ignore", invalid="ignore"):
        coefficient[:] = nudge.beta * weight[:, None] * nudge.mask
        linear[:] = coefficient * np.broadcast_to(nudge.target, shape)
        if nudge.anchor is not None:
            assert nudge.anchor_gain is not None
            linear += nudge.anchor_gain * np.broadcast_to(nudge.anchor, shape)
            coefficient += nudge.anchor_gain
    if not np.isfinite(linear).all() or not np.isfinite(coefficient).all():
        raise ValueError("hybrid quadratic coefficients must be finite")
    return linear, coefficient


@dataclass(frozen=True)
class _Reduced:
    a: np.ndarray
    b: np.ndarray
    hi: np.ndarray
    hc: np.ndarray
    quadratic: np.ndarray

    def evaluate(
        self, x: np.ndarray, *, hessian: bool = True
    ) -> tuple[float, np.ndarray, np.ndarray | None]:
        if not np.isfinite(x).all() or np.any(np.abs(x) >= 1):
            raise ValueError("retained activity must be finite and strictly inside (-1, 1)")
        vi = self.hi + self.a @ x
        si = np.tanh(vi / 2)
        bx = self.b @ x
        primitive = (1 + x) * np.log1p(x) + (1 - x) * np.log1p(-x)
        energy = float(
            primitive.sum()
            - self.hc @ x
            - 0.5 * x @ bx
            + 0.5 * (self.quadratic * x) @ x
            - 2 * (np.logaddexp(vi / 2, -vi / 2) - np.log(2)).sum()
        )
        gradient = 2 * np.arctanh(x) - self.hc - bx - self.a.T @ si + self.quadratic * x
        curvature = None
        if hessian:
            curvature = (
                np.diag(2 / (1 - x * x) + self.quadratic)
                - self.b
                - self.a.T @ (0.5 * (1 - si * si)[:, None] * self.a)
            )
        return energy, gradient, curvature


def _minimum(hessian: np.ndarray) -> float:
    if not np.isfinite(hessian).all():
        return float("nan")
    if not len(hessian):
        # No retained variables: the independent input energy is strictly convex.
        return 2.0
    try:
        return float(np.linalg.eigvalsh(hessian)[0])
    except np.linalg.LinAlgError:
        return float("nan")


def _curvature(
    brain: Brain,
    v: np.ndarray,
    inputs: np.ndarray,
    retained: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    coefficient: np.ndarray,
) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        s = brain.neuron_model.activation(v)
        derivative = brain.neuron_model.slope_at(v)
    if (
        not np.isfinite(v).all()
        or not np.isfinite(s).all()
        or not np.isfinite(derivative).all()
        or np.any(derivative <= 0)
        or np.any(np.abs(s) >= 1)
    ):
        return float("nan")
    if not len(retained):
        return float(np.min(1 / derivative[inputs], initial=np.inf))
    return _minimum(
        np.diag(1 / derivative[retained] + coefficient[retained])
        - b
        - a.T @ (derivative[inputs, None] * a)
    )


def _residual(
    brain: Brain,
    v: np.ndarray,
    drive: np.ndarray,
    nudge: Nudge | None,
) -> float:
    """Audit every original neuron equation using the actual activation rule."""
    with np.errstate(over="ignore", invalid="ignore"):
        s = brain.neuron_model.activation(v)
        error = np.zeros_like(v)
        np.add.at(error, brain.connectome.post, brain.weights * s[brain.connectome.pre])
        error = error + drive + brain.bias - v
        if nudge is not None:
            error += nudge.drive(s[None, :])[0]
        value = float(np.max(np.abs(error), initial=0))
    return value if np.isfinite(value) else float("inf")


def refine_equilibrium(
    brain: Brain,
    drive: np.ndarray,
    local: Equilibrium,
    input_index: np.ndarray,
    *,
    nudge: Nudge | None = None,
    max_steps: int = 64,
) -> Equilibrium:
    """Audit local curvature and optionally refine capped rows without changing equations.

    Only positive-curvature Newton directions are attempted. Line-search failure,
    saturation, nonfinite arithmetic and exhausted budgets return unqualified
    diagnostics; no saddle escape, damping of the Hessian or tolerance relaxation
    is performed. A zero budget audits the unchanged local state only.
    """
    if (
        isinstance(max_steps, (bool, np.bool_))
        or not isinstance(max_steps, (int, np.integer))
        or max_steps < 0
    ):
        raise ValueError("refinement max_steps must be a nonnegative integer")
    if not np.isfinite(local.tolerance) or local.tolerance < 0:
        raise ValueError("refinement tolerance must be finite and nonnegative")
    inputs, retained, a, b = _geometry(brain, input_index)
    v_original = np.asarray(local.state.v)
    if any(
        np.asarray(array).dtype != np.float64
        for array in (
            local.state.v,
            local.state.activation,
            local.state.adaptation,
        )
    ):
        raise ValueError("hybrid refinement requires float64 state")
    d = np.atleast_2d(np.asarray(drive, dtype=float))
    v = np.atleast_2d(v_original).copy()
    if v_original.ndim not in (1, 2) or d.shape != v.shape or d.shape[1] != brain.connectome.n:
        raise ValueError("state and drive must have matching neuron and batch dimensions")
    if not np.isfinite(d).all():
        raise ValueError("hybrid drive must be finite")
    if np.asarray(local.residual).shape != (len(v),):
        raise ValueError("local residual must have one entry per batch row")
    linear, coefficient = _quadratic(nudge, d.shape, inputs)
    steps = np.zeros(len(v), dtype=np.int64)
    curvature = np.full(len(v), np.nan)
    residual = np.full(len(v), np.inf)
    statuses: list[str] = []
    activation = np.atleast_2d(local.state.activation).copy()
    if activation.shape != v.shape or np.asarray(local.state.adaptation).shape != v_original.shape:
        raise ValueError("state arrays must have matching dimensions")
    for row in range(len(v)):
        current = v[row]
        row_nudge = (
            None
            if nudge is None
            else replace(
                nudge,
                target=np.broadcast_to(nudge.target, d.shape)[row],
                weight=None if nudge.weight is None else nudge.weight[row : row + 1],
                anchor=None
                if nudge.anchor is None
                else np.broadcast_to(nudge.anchor, d.shape)[row],
            )
        )
        status = "step_cap"
        residual[row] = _residual(brain, current, d[row], row_nudge)
        curvature[row] = _curvature(brain, current, inputs, retained, a, b, coefficient[row])
        if (
            not np.isfinite(current).all()
            or not np.isfinite(activation[row]).all()
            or not np.isfinite(np.atleast_2d(local.state.adaptation)[row]).all()
        ):
            statuses.append("nonfinite_state")
            continue
        actual_activation = brain.neuron_model.activation(current)
        if not np.allclose(activation[row], actual_activation, rtol=2e-15, atol=2e-15):
            statuses.append("inconsistent_activation")
            continue
        if not np.isfinite(curvature[row]):
            statuses.append("nonfinite_curvature_or_saturation")
            continue
        if curvature[row] <= 0:
            statuses.append("nonpositive_curvature")
            continue
        if residual[row] <= local.tolerance:
            statuses.append("local")
            continue
        if max_steps == 0:
            statuses.append(status)
            continue
        system = _Reduced(
            a,
            b,
            d[row, inputs] + brain.bias[inputs] + linear[row, inputs],
            d[row, retained] + brain.bias[retained] + linear[row, retained],
            coefficient[row, retained],
        )
        x = np.tanh(current[retained] / 2)
        for iteration in range(max_steps + 1):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                energy, gradient, hessian = system.evaluate(x)
            assert hessian is not None
            smallest = _minimum(hessian)
            error = float(np.max(np.abs(gradient), initial=0))
            if (
                not np.isfinite(energy)
                or not np.isfinite(gradient).all()
                or not np.isfinite(smallest)
            ):
                status = "nonfinite_refinement"
                break
            if smallest <= 0:
                status = "nonpositive_curvature"
                break
            if error <= local.tolerance:
                status = "refined"
                break
            if iteration == max_steps:
                break
            try:
                # Both checks are explicit: NumPy has no triangular-solve API.
                np.linalg.cholesky(hessian)
                direction = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError:
                status = "linear_solve_failed"
                break
            slope = float(gradient @ direction)
            if not np.isfinite(direction).all() or not np.isfinite(slope) or slope >= 0:
                status = "invalid_newton_direction"
                break
            accepted = False
            for backtrack in range(40):
                alpha = 2.0**-backtrack
                trial = x + alpha * direction
                if not np.isfinite(trial).all() or np.any(np.abs(trial) >= 1):
                    continue
                e_trial, g_trial, _ = system.evaluate(trial, hessian=False)
                slack = 64 * np.finfo(float).eps * (1 + abs(energy))
                armijo = e_trial <= energy + 1e-4 * alpha * slope
                rounded = e_trial <= energy + slack and np.max(np.abs(g_trial), initial=0) < error
                if np.isfinite(e_trial) and np.isfinite(g_trial).all() and (armijo or rounded):
                    x, accepted = trial, True
                    steps[row] += 1
                    break
            if not accepted:
                status = "line_search_exhausted"
                break
        # Only retain genuine refinement work (or exact conditional elimination).
        if steps[row] or status == "refined":
            current[inputs] = system.hi + a @ x
            current[retained] = 2 * np.arctanh(x)
            activation[row] = brain.neuron_model.activation(current)
        residual[row] = _residual(brain, current, d[row], row_nudge)
        curvature[row] = _curvature(brain, current, inputs, retained, a, b, coefficient[row])
        if status == "refined" and (
            residual[row] > local.tolerance
            or not np.isfinite(curvature[row])
            or curvature[row] <= 0
        ):
            status = "full_equation_or_curvature_failed"
        statuses.append(status)
    single = v_original.ndim == 1
    changed = not np.array_equal(v, np.atleast_2d(v_original), equal_nan=True)
    state = BrainState(
        v=v[0] if single else v,
        activation=activation[0] if single else activation,
        adaptation=np.array(local.state.adaptation, copy=True),
        steps=local.state.steps,
        trajectory=None
        if changed or local.state.trajectory is None
        else local.state.trajectory.copy(),
        activity_change=None
        if changed or local.state.activity_change is None
        else local.state.activity_change.copy(),
    )
    report = RefinementReport(
        np.array(local.residual, copy=True), steps, curvature, tuple(statuses)
    )
    return Equilibrium(state, residual, local.steps, local.tolerance, refinement=report)
