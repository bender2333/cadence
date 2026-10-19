"""The even/odd activation family, the weighted cascade certificate, and time-shared entry ties.

M12 S0 (EP-EEG) acceptance pre-checks; every number here is a unit-level identity, not an
experiment. Seeds >= 100 (dev runs, EP-EEG M12 amendment 1, N10).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import cadence as cd
from cadence.certificate import certificate, lipschitz_constant, weighted_row_mass
from cadence.learning import (
    Learner,
    LearnerConfig,
    evenodd_neuron_model,
    learning_neuron_model,
    tie_rates,
    time_shared_entry,
)
from cadence.neuron import NeuronModel, evenodd_lipschitz

GRID = np.linspace(-60.0, 60.0, 4_000_001)


def _grid_max(a: float, b: float) -> float:
    t = np.tanh(0.5 * GRID)
    return float(np.abs(0.5 * (1.0 - t * t) * (a + 2.0 * b * t)).max())


@pytest.mark.parametrize(
    "a,b", [(1.0, 0.0), (1.0, 0.25), (1.0, 0.5), (1.0, 1.0), (1.0, -1.0), (0.2, 2.0), (2.0, -2.0),
            (1.0, 2.0), (1.0, 0.6), (1.0, 1e-12), (1.0, 0.49), (1.0, 0.51)]
)
def test_lipschitz_closed_form_matches_dense_grid(a: float, b: float) -> None:
    lip, grid = evenodd_lipschitz(a, b), _grid_max(a, b)
    assert lip >= grid - 1e-12
    assert lip - grid <= 1e-8


def test_lipschitz_random_parameters() -> None:
    rng = np.random.default_rng(100)
    t = np.tanh(0.5 * GRID[::20])  # 2e5 points: the closed form must never fall below it
    for a, b in zip(rng.uniform(0.2, 2.0, 200), rng.uniform(-2.0, 2.0, 200), strict=True):
        assert evenodd_lipschitz(a, b) >= float(np.abs(0.5 * (1 - t * t) * (a + 2 * b * t)).max()) - 1e-12


def test_small_b_is_stable() -> None:
    assert evenodd_lipschitz(1.0, 1e-12) == pytest.approx(0.5, abs=1e-15)
    assert evenodd_lipschitz(1.0, 1e-6) >= 0.5


def test_activation_and_slope_match_definition_and_central_difference() -> None:
    rng = np.random.default_rng(101)
    n = 7
    a, b = rng.uniform(0.2, 2.0, n), rng.uniform(-2.0, 2.0, n)
    model = evenodd_neuron_model(a, b, linear=[0])
    v = rng.normal(0, 3, (5, n))
    t = np.tanh(v / 2)
    expect = a * t + b * t * t
    expect[:, 0] = v[:, 0]
    np.testing.assert_allclose(model.activation(v), expect, rtol=0, atol=1e-15)
    h = 1e-5
    fd = (model.activation(v + h) - model.activation(v - h)) / (2 * h)
    np.testing.assert_allclose(model.slope_at(v), fd, rtol=1e-8, atol=1e-10)
    assert model.activation(np.zeros(n)).tolist() == [0.0] * n  # rest emits nothing
    np.testing.assert_allclose(model.lipschitz_per_neuron(n)[1:], [evenodd_lipschitz(x, y) for x, y in zip(a[1:], b[1:], strict=True)])
    assert model.lipschitz_per_neuron(n)[0] == 1.0


def test_family_validation() -> None:
    with pytest.raises(ValueError):
        NeuronModel(odd=2.0)  # evenodd fields on the sigmoid family
    with pytest.raises(ValueError):
        evenodd_neuron_model(-1.0, 0.0)
    with pytest.raises(ValueError):
        NeuronModel(family="relu")
    m = evenodd_neuron_model((1.0, 2.0), 0.5, linear=[1])
    assert NeuronModel(**{k: v for k, v in m.to_dict().items() if k != "adaptation"}) == m
    assert "family" not in learning_neuron_model().to_dict()  # default records unchanged


def _graph(seed: int, n_src: int = 6, n_hid: int = 10, n_out: int = 3):
    rng = np.random.default_rng(seed)
    pre, post, w = [], [], []
    for s_ in range(n_src):
        for h in range(n_src, n_src + n_hid):
            pre.append(s_); post.append(h); w.append(rng.normal())
    free = list(range(n_src, n_src + n_hid + n_out))
    for i in free:
        for j in free:
            if i < j and not (i >= n_src + n_hid and j >= n_src + n_hid):
                x = rng.normal()
                pre += [i, j]; post += [j, i]; w += [x, x]
    n = n_src + n_hid + n_out
    conn = cd.Connectome.from_synapses(n, pre=np.array(pre), post=np.array(post))
    eff = np.array(w)
    key = {(p, q): e for p, q, e in zip(pre, post, w, strict=True)}
    eff = np.array([key[(p, q)] for p, q in zip(conn.pre, conn.post, strict=True)])
    return conn, eff, n


@pytest.mark.parametrize("backend", ["cpu", "torch"])
def test_b_zero_reproduces_the_old_tanh_path(backend: str) -> None:
    if backend == "torch":
        pytest.importorskip("torch")
    conn, eff, n = _graph(102)
    eff = eff * 0.08
    old = cd.NeuralGraph(conn, learning_neuron_model(leak=1.0), efficacy=eff, bias=np.zeros(n), backend=backend,
                         **({"device": "cpu"} if backend == "torch" else {}))
    new = cd.NeuralGraph(conn, evenodd_neuron_model(1.0, 0.0), efficacy=eff, bias=np.zeros(n), backend=backend,
                         **({"device": "cpu"} if backend == "torch" else {}))
    drive = np.zeros((4, n)); drive[:, :6] = np.random.default_rng(103).normal(0, 2, (4, 6))
    a = old.equilibrate(drive, budget=20000, tolerance=1e-11)
    b = new.equilibrate(drive, budget=20000, tolerance=1e-11)
    assert np.all(a.converged) and np.all(b.converged)
    assert np.abs(np.asarray(a.state.v) - np.asarray(b.state.v)).max() <= 1e-12
    assert np.abs(np.asarray(a.state.activation) - np.asarray(b.state.activation)).max() <= 1e-12


def test_torch_matches_cpu_for_evenodd_with_linear_sources() -> None:
    pytest.importorskip("torch")
    conn, eff, n = _graph(104)
    a = np.r_[np.ones(6), np.full(n - 6, 1.0)]
    b = np.r_[np.zeros(6), np.full(n - 6, 1.0)]
    model = evenodd_neuron_model(a, b, linear=range(6))
    eff = eff * 0.05
    drive = np.zeros((4, n)); drive[:, :6] = np.random.default_rng(105).normal(0, 2, (4, 6))
    ref = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n)).equilibrate(drive, budget=20000, tolerance=1e-11)
    tor = cd.NeuralGraph(conn, model, efficacy=eff, bias=np.zeros(n), backend="torch", device="cpu").equilibrate(
        drive, budget=20000, tolerance=1e-11)
    assert np.all(ref.converged) and np.all(tor.converged)
    np.testing.assert_allclose(np.asarray(ref.state.activation)[:, :6], drive[:, :6], atol=1e-10)  # s = v = x
    assert np.abs(np.asarray(ref.state.v) - np.asarray(tor.state.v)).max() <= 1e-9


def test_weighted_cascade_certificate() -> None:
    conn, eff, n = _graph(106)
    rng = np.random.default_rng(107)
    a, b = np.ones(n), np.r_[np.zeros(6), rng.uniform(-2, 2, n - 6)]
    model = evenodd_neuron_model(a, b, linear=range(6))
    brain = cd.NeuralGraph(conn, model, efficacy=eff * 0.1, bias=np.zeros(n))
    cert = certificate(brain, sources=range(6))
    W = np.asarray(brain.dense())  # W[pre, post]
    lip = np.array([1.0] * 6 + [evenodd_lipschitz(1.0, x) for x in b[6:]])
    ref = (np.abs(W[6:, 6:]) * lip[6:, None]).sum(0).max()
    assert cert.weighted_row_mass_free == pytest.approx(ref, rel=1e-13)
    assert cert.rate_free == pytest.approx(1 - model.dt * (1 - ref), rel=1e-13)
    assert cert.weighted_row_mass == pytest.approx((np.abs(W) * lip[:, None]).sum(0).max(), rel=1e-13)
    # uniform L: the weighted mass is L times the unweighted one (ADR-0012 form)
    uni = cd.NeuralGraph(conn, evenodd_neuron_model(1.0, 1.0, linear=range(6)), efficacy=eff * 0.1, bias=np.zeros(n))
    c2 = certificate(uni, sources=range(6))
    assert c2.weighted_row_mass_free == pytest.approx(evenodd_lipschitz(1, 1) * c2.row_mass_free, rel=1e-12)
    assert lipschitz_constant(model, n) == pytest.approx(max(1.0, lip.max()), rel=1e-15)  # sup over neurons
    assert weighted_row_mass(brain) >= cert.weighted_row_mass_free


def test_linear_neuron_with_incoming_synapse_is_refused() -> None:
    conn, eff, n = _graph(108)
    brain = cd.NeuralGraph(conn, evenodd_neuron_model(1.0, 0.5, linear=[6]), efficacy=eff, bias=np.zeros(n))
    with pytest.raises(ValueError, match="linear neuron"):
        certificate(brain)


def test_time_shared_entry_groups_and_rates() -> None:
    pre, post, tie, entry = time_shared_entry(3, 3, 4, filter_positions=[[0, 1, 2], [0, 1, 2], [0, 1], [1, 2]])
    assert len(pre) == 3 * (3 + 3 + 2 + 2)
    assert entry.max() == 9 + 10 - 1
    rate = tie_rates(tie)
    sizes = {g: int((tie == g).sum()) for g in np.unique(tie)}
    assert all(rate[i] == sizes[tie[i]] for i in range(len(tie)))
    assert sorted(set(sizes.values())) == [2, 3]
    for i in range(len(pre)):  # (p, c) -> (k, p): same channel and position
        k, p = np.argwhere(entry == post[i])[0]
        assert pre[i] // 3 == p and tie[i] == k * 3 + pre[i] % 3


def test_tied_group_step_is_the_sum_of_member_contrasts() -> None:
    pre, post, tie, entry = time_shared_entry(3, 3, 4, filter_positions=[[0, 1, 2], [0, 1, 2], [0, 1], [1, 2]])
    n_src, n_hid, n_out = 9, 10, 3
    n = n_src + n_hid + n_out
    outs = list(range(n_src + n_hid, n))
    p2, q2 = list(pre), list(post)
    tie2 = list(tie)
    for h in range(n_src, n_src + n_hid):
        for o in outs:
            p2 += [h, o]; q2 += [o, h]; tie2 += [-1, -1]
    conn = cd.Connectome.from_synapses(n, pre=np.array(p2), post=np.array(q2))
    order = {(a, b): t for a, b, t in zip(p2, q2, tie2, strict=True)}
    groups = np.array([order[(a, b)] for a, b in zip(conn.pre, conn.post, strict=True)])
    rng = np.random.default_rng(109)
    shared = rng.normal(size=groups.max() + 1)
    eff = np.where(groups >= 0, shared[np.maximum(groups, 0)], rng.normal(size=len(groups)) * 0.1)
    brain = cd.NeuralGraph(conn, evenodd_neuron_model(1.0, 0.6, linear=range(n_src)), efficacy=eff, bias=np.zeros(n))
    cfg = LearnerConfig(beta=0.04, eta=0.5, eta_bias=0.0, centered=True, free_steps=5000, nudged_steps=5000,
                        tolerance=1e-9, nudge="quadratic", normalize=0.0, momentum=0.0, decay=0.0, scale_cap=1e9,
                        qualified=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = Learner(brain, outs, config=cfg, tie_groups=groups, synapse_rate=tie_rates(groups))
    delta = rng.normal(size=len(groups))
    before = np.asarray(lr.brain.efficacy).copy()
    lr.apply(delta, np.zeros(n))
    moved = np.asarray(lr.brain.efficacy) - before
    for g in range(groups.max() + 1):
        m = groups == g
        assert np.ptp(moved[m]) <= 1e-12
        assert moved[m][0] == pytest.approx(delta[m].sum(), rel=1e-12, abs=1e-15)
        assert np.ptp(np.asarray(lr.brain.efficacy)[m]) <= 1e-12
