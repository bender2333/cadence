"""The settling certificate: row mass, slope bound, contraction rate, and the bounds it gives."""

from __future__ import annotations

import numpy as np
import pytest

import cadence as cd


def _certified_brain(mass: float, seed: int = 0) -> tuple[cd.NeuralGraph, np.ndarray]:
    connectome = cd.layered(6, 10, 3, density=1.0, seed=seed)
    model = cd.learning_neuron_model()
    raw = cd.NeuralGraph(connectome, model)
    scale = mass / cd.row_mass(raw)
    brain = cd.NeuralGraph(connectome, model, efficacy=raw.efficacy * scale)
    return brain, np.asarray(connectome.populations["input"])


def _drive(brain: cd.NeuralGraph, inputs: np.ndarray, rng: np.random.Generator, rows: int) -> np.ndarray:
    drive = np.zeros((rows, brain.connectome.n))
    drive[:, inputs] = rng.uniform(0.0, 1.0, size=(rows, len(inputs)))
    return drive


def _equilibrium(brain: cd.NeuralGraph, drive: np.ndarray) -> np.ndarray:
    state = brain.settle_batch(drive, steps=2000, tolerance=None)
    return np.asarray(state.v)


def test_row_mass_is_the_largest_absolute_incoming_weight_sum() -> None:
    brain, _ = _certified_brain(1.0)
    dense = brain.dense()  # W[pre, post]
    assert cd.row_mass(brain) == pytest.approx(np.abs(dense).sum(axis=0).max())


def test_lipschitz_constant_of_the_learning_model_is_one_half() -> None:
    assert cd.lipschitz_constant(cd.learning_neuron_model()) == pytest.approx(0.5)
    model = cd.NeuronModel(slope=4.0, threshold=1.5, leak=0.0)
    rest = model.rest_emission
    assert cd.lipschitz_constant(model) == pytest.approx(1.0 / (1.0 - rest))
    # the constant bounds the measured slope everywhere
    v = np.linspace(-6.0, 6.0, 4001)
    for m in (
        cd.learning_neuron_model(),
        model,
        cd.NeuronModel(slope=2.0, threshold=0.5, leak=0.3),
    ):
        assert np.max(m.slope_at(v)) <= cd.lipschitz_constant(m) + 1e-9


def test_certificate_rate_and_limit() -> None:
    brain, _ = _certified_brain(1.0)
    cert = cd.certificate(brain)
    assert cert.certified
    assert cert.lipschitz == pytest.approx(0.5)
    assert cert.mass_limit == pytest.approx(2.0)
    assert cert.rate == pytest.approx(1.0 - 0.5 * (1.0 - 0.5 * 1.0))
    hot, _ = _certified_brain(3.0)
    assert not cd.certificate(hot).certified
    assert np.isinf(cd.certificate(hot).error_bound(1e-3))
    with pytest.raises(ValueError):
        cd.certificate(hot).steps_for(0.1, 1e-3)
    rhythmic = cd.NeuralGraph(
        brain.connectome,
        cd.learning_neuron_model().replace(adaptation=cd.Adaptation(40.0, 1.0)),
        efficacy=brain.efficacy,
    )
    assert not cd.certificate(rhythmic).certified
    assert cert.to_dict()["certified"] is True


def test_a_priori_and_a_posteriori_bounds_hold_along_the_settling() -> None:
    brain, inputs = _certified_brain(1.5)
    cert = cd.certificate(brain)
    rng = np.random.default_rng(1)
    drive = _drive(brain, inputs, rng, 4)
    star = _equilibrium(brain, drive)
    state = brain.settle_batch(drive, steps=1, tolerance=None)
    previous = np.zeros_like(star)
    first = np.max(np.abs(np.asarray(state.v) - previous), axis=1)
    for k in range(1, 40):
        current = np.asarray(state.v)
        nxt = brain.settle_batch(drive, steps=1, state=state, tolerance=None)
        movement = np.max(np.abs(np.asarray(nxt.v) - current), axis=1)
        distance = np.max(np.abs(current - star), axis=1)
        assert np.all(distance <= cert.error_bound(movement) + 1e-12)
        assert np.all(distance <= cert.apriori_bound(first, k) + 1e-12)
        state = nxt


def test_warm_start_bound_and_step_budget() -> None:
    brain, inputs = _certified_brain(1.2)
    cert = cd.certificate(brain)
    rng = np.random.default_rng(2)
    drive = _drive(brain, inputs, rng, 3)
    changed = drive.copy()
    changed[:, inputs] += rng.uniform(-0.2, 0.2, size=(3, len(inputs)))
    delta = float(np.max(np.abs(changed - drive)))
    star, star2 = _equilibrium(brain, drive), _equilibrium(brain, changed)
    assert np.max(np.abs(star - star2)) <= cert.dt * delta / (1.0 - cert.rate) + 1e-12
    tolerance = 1e-6
    budget = cert.steps_for(delta, tolerance)
    warm = brain.settle_batch(drive, steps=2000, tolerance=None)
    after = brain.settle_batch(changed, steps=budget, state=warm, tolerance=None)
    assert np.max(np.abs(np.asarray(after.v) - star2)) <= tolerance
    assert cert.steps_for(0.0, tolerance) == 0


def test_silence_is_the_equilibrium_of_no_input() -> None:
    brain, _ = _certified_brain(1.0)
    state = brain.settle_batch(np.zeros((1, brain.connectome.n)), steps=50, tolerance=None)
    assert np.max(np.abs(np.asarray(state.v))) == 0.0
    assert np.max(np.abs(np.asarray(state.activation))) == 0.0


def test_lesion_keeps_the_certificate_and_stays_within_the_bound() -> None:
    brain, inputs = _certified_brain(1.5)
    cert = cd.certificate(brain)
    rng = np.random.default_rng(3)
    drive = _drive(brain, inputs, rng, 2)
    hidden = np.asarray(brain.connectome.populations["hidden"])
    cut = hidden[:3]
    mask = np.ones(brain.connectome.n)
    mask[cut] = 0.0
    dense = brain.dense()
    mu = np.abs(dense[cut, :]).sum(axis=0).max()
    star = _equilibrium(brain, drive)
    lesioned = np.asarray(brain.settle_batch(drive, steps=2000, mask=mask, tolerance=None).v)
    survivors = mask > 0
    gap = np.max(np.abs((star - lesioned)[:, survivors]))
    assert gap <= cert.dt * mu * 1.0 / (1.0 - cert.rate) + 1e-9


def test_cascade_certificate_certifies_scaled_input_network() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=42)
    inputs = np.asarray(connectome.populations["input"])
    model = cd.learning_neuron_model(dt=0.5)

    pre = connectome.pre
    is_input_synapse = np.isin(pre, inputs)
    efficacy = np.ones(connectome.synapses) * 0.15
    efficacy[is_input_synapse] *= 10.0

    brain = cd.NeuralGraph(connectome, model, efficacy=efficacy)
    cert_full = cd.certificate(brain)
    assert not cert_full.certified
    assert cert_full.row_mass > cert_full.mass_limit
    assert cert_full.rate > 1.0
    assert cert_full.row_mass_free is None
    assert cert_full.sources is None
    assert cert_full.rate_free is None
    assert cert_full.to_dict()["row_mass_free"] is None
    assert cert_full.to_dict()["sources"] is None

    cert_cascade = cd.certificate(brain, sources=inputs)
    assert cert_cascade.certified
    assert cert_cascade.sources == tuple(int(i) for i in inputs)
    assert cert_cascade.row_mass == cert_full.row_mass
    assert cert_cascade.row_mass_free is not None
    assert cert_cascade.row_mass_free < cert_cascade.mass_limit

    # row_mass_free matches dense free-block incoming sum
    dense = brain.dense()
    is_source = np.zeros(connectome.n, dtype=bool)
    is_source[inputs] = True
    dense_free = dense[~is_source, :][:, ~is_source]
    expected_free_mass = float(np.abs(dense_free).sum(axis=0).max())
    assert cert_cascade.row_mass_free == pytest.approx(expected_free_mass)

    rate_free = 1.0 - 0.5 * (1.0 - 0.5 * cert_cascade.row_mass_free)
    assert cert_cascade.rate_free == pytest.approx(rate_free)
    assert cert_cascade.rate == pytest.approx(max(rate_free, 1.0 - 0.5))
    assert cert_cascade.to_dict()["certified"] is True
    assert cert_cascade.to_dict()["row_mass_free"] == pytest.approx(expected_free_mass)
    assert cert_cascade.to_dict()["sources"] == tuple(int(i) for i in inputs)
    assert np.isfinite(cert_cascade.error_bound(1e-3)).all()
    assert cert_cascade.steps_for(0.1, 1e-3) > 0


def test_cascade_certificate_rejects_non_source_inputs() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=42)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    hidden_idx = int(connectome.populations["hidden"][0])

    with pytest.raises(ValueError, match="source neurons must have zero incoming effective mass"):
        cd.certificate(brain, sources=[hidden_idx])

    inputs = list(connectome.populations["input"])
    with pytest.raises(ValueError, match="source neurons must have zero incoming effective mass"):
        cd.certificate(brain, sources=inputs + [hidden_idx])

    with pytest.raises(ValueError):
        cd.certificate(brain, sources=[-1])


def test_cascade_contraction_rate_and_bounds() -> None:
    # rate is max(rate_free, 1 - dt)
    # Case 1: rate_free > 1 - dt
    c1 = cd.Certificate(row_mass=5.0, lipschitz=0.5, dt=0.5, adaptation=False, row_mass_free=0.8, sources=(0, 1))
    # rate_free = 1 - 0.5 * (1 - 0.5 * 0.8) = 1 - 0.5 * 0.6 = 0.70
    # 1 - dt = 0.50 -> rate = 0.70
    assert c1.rate_free == pytest.approx(0.70)
    assert c1.rate == pytest.approx(0.70)
    assert c1.certified

    # Case 2: rate_free < 1 - dt
    c2 = cd.Certificate(row_mass=5.0, lipschitz=0.5, dt=0.2, adaptation=False, row_mass_free=0.0, sources=(0, 1))
    # rate_free = 1 - 0.2 * (1 - 0) = 0.80
    # 1 - dt = 0.80 -> rate = 0.80
    assert c2.rate_free == pytest.approx(0.80)
    assert c2.rate == pytest.approx(0.80)

    c3 = cd.Certificate(row_mass=5.0, lipschitz=0.5, dt=0.9, adaptation=False, row_mass_free=0.0, sources=(0, 1))
    # rate_free = 1 - 0.9 * 1 = 0.10
    # 1 - dt = 0.10 -> rate = 0.10
    assert c3.rate_free == pytest.approx(0.10)
    assert c3.rate == pytest.approx(0.10)

    # Adaptation blocks certification even with small row_mass_free
    c_adapt = cd.Certificate(row_mass=5.0, lipschitz=0.5, dt=0.5, adaptation=True, row_mass_free=0.5, sources=(0, 1))
    assert not c_adapt.certified
    assert np.isinf(c_adapt.error_bound(1e-3))

    # Empty sources behaves like full graph
    connectome = cd.layered(3, 4, 2, density=1.0, seed=1)
    brain = cd.NeuralGraph(connectome, cd.learning_neuron_model())
    cert_full = cd.certificate(brain)
    cert_empty = cd.certificate(brain, sources=())
    assert cert_empty.row_mass_free == pytest.approx(cert_full.row_mass)
    assert cert_empty.sources == ()
    assert cert_empty.rate == pytest.approx(cert_full.rate)

    # Validation on Certificate dataclass
    with pytest.raises(ValueError, match="row_mass_free must be finite and nonnegative"):
        cd.Certificate(row_mass=1.0, lipschitz=0.5, dt=0.5, adaptation=False, row_mass_free=-0.1)
    with pytest.raises(ValueError, match="sources must be a tuple of nonnegative integers"):
        cd.Certificate(row_mass=1.0, lipschitz=0.5, dt=0.5, adaptation=False, sources=(-1,))
    with pytest.raises(ValueError, match="sources must be a tuple of nonnegative integers"):
        cd.Certificate(row_mass=1.0, lipschitz=0.5, dt=0.5, adaptation=False, sources=[0])  # type: ignore[arg-type]
