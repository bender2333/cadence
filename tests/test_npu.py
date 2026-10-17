"""Dedicated Ascend NPU integration test suite."""

from pathlib import Path

import numpy as np
import pytest

import cadence as cd

pytestmark = pytest.mark.skipif(
    cd.available_backends().get("torch") != "npu float32",
    reason="Ascend NPU hardware unavailable",
)


def test_multimodule_neural_graph_equilibrium_relaxation_residual() -> None:
    brain = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(8, 6),
        observers=(3,),
        seed=42,
        backend="torch",
        device="npu",
        precision="float32",
    )
    graph = brain.brain
    assert graph._torch is not None
    assert graph._torch.device.type == "npu"
    assert graph._torch.dtype == graph._torch.torch.float32

    rng = np.random.default_rng(123)
    c = graph.connectome
    drive = brain.stimulus(rng.uniform(0.1, 0.8, (2, 4)), memory=False)

    free = graph.equilibrate(drive, budget=512, tolerance=3e-3, damping=3)
    assert free.qualified.all()
    assert float(free.residual.max()) <= 0.003
    assert float(graph.residual(drive, free.state, on_device=False).max()) <= 0.003

    mask = np.zeros(c.n)
    mask[brain.motor_index] = 1.0
    target = np.zeros((2, c.n))
    target[0, brain.motor_index[0]] = 1.0
    target[1, brain.motor_index[1]] = 1.0
    nudge = cd.Nudge(target, mask, beta=0.1, softmax_temperature=0.2)

    nudged = graph.equilibrate(drive, nudge=nudge, budget=512, tolerance=3e-3, damping=3)
    assert nudged.qualified.all()
    assert float(nudged.residual.max()) <= 0.003
    assert float(graph.residual(drive, nudged.state, nudge=nudge, on_device=False).max()) <= 0.003


def test_composed_brain_continuous_interaction_cycles() -> None:
    brain = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(8,),
        seed=7,
        backend="torch",
        device="npu",
        precision="float32",
    )
    assert brain.brain._torch is not None
    assert brain.brain._torch.device.type == "npu"
    assert brain.brain.precision == "float32"

    rng = np.random.default_rng(7)
    x = rng.uniform(0.1, 0.9, (1, 4))
    first_act = brain.step(x)
    assert first_act.shape == (1,)
    assert brain.pending_feedback

    for i in range(3):
        next_x = rng.uniform(0.1, 0.9, (1, 4))
        reward = np.array([0.5 if i % 2 == 0 else -0.5])
        done = np.array([i == 2])
        act = brain.step(next_x, reward=reward, done=done)
        assert act.shape == (1,)


def test_live_arousal_routine_and_repair() -> None:
    brain = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(8,),
        arousal={"youth": 0},
        seed=1,
        backend="torch",
        device="npu",
        precision="float32",
    )
    eye = np.eye(4)
    act0 = brain.live(eye[[0]])
    assert act0.shape == (1,)
    assert brain.pending_feedback

    # Routine mode: expected outcome changes nothing
    act1 = brain.live(eye[[1]], reward=[0.0])
    assert act1.shape == (1,)
    reading1 = brain.last_arousal
    assert reading1 is not None
    assert reading1["mode"] == "routine"
    assert not reading1["learned"]

    # Surprising outcome: arouses brain, triggering repair
    act2 = brain.live(eye[[2]], reward=[5.0])
    assert act2.shape == (1,)
    reading2 = brain.last_arousal
    assert reading2 is not None
    assert reading2["mode"] == "aroused"
    assert reading2["error"] > 0

def test_step_explicit_teaching_and_reward_eligibility() -> None:
    brain = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(8,),
        seed=4,
        backend="torch",
        device="npu",
        precision="float32",
        learning_free_steps=64,
        learning_nudged_steps=12,
    )
    x = np.eye(4)[:2]
    brain.step(x)
    assert brain.pending_feedback
    initial_updates = brain.basal_ganglia.updates
    brain.step(x, reward=np.array([1.0, -1.0]), done=np.array([False, False]))
    assert brain.basal_ganglia.updates == initial_updates + 1

    learner_updates = brain.learner.updates
    brain.step(x, teacher=np.array([0, 1]))
    assert brain.learner.updates > learner_updates
    assert brain.last_learning["demonstrations"] == 2.0
    assert brain.last_learning["demonstration_accepted"] == 1.0


def test_checkpoint_round_trip_cpu_npu_transfer(tmp_path: Path) -> None:
    x = np.eye(4)[:2]
    b_npu = cd.Brain.compose(
        inputs=4,
        actions=2,
        modules=(8,),
        seed=42,
        backend="torch",
        device="npu",
        precision="float32",
    )
    b_npu.step(x)
    assert b_npu.pending_feedback

    npu_save = tmp_path / "brain_npu.npz"
    b_npu.save(npu_save)
    b_cpu = cd.Brain.load(npu_save, backend="cpu")
    assert b_cpu.brain.backend == "cpu"
    assert b_cpu.pending_feedback
    np.testing.assert_allclose(b_cpu.brain.efficacy, b_npu.brain.efficacy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b_cpu.brain.bias, b_npu.brain.bias, rtol=1e-5, atol=1e-5)

    cpu_save = tmp_path / "brain_cpu.npz"
    b_cpu.save(cpu_save)
    b_npu2 = cd.Brain.load(cpu_save, backend="torch", device="npu", precision="float32")
    assert b_npu2.brain.backend == "torch"
    assert b_npu2.brain._torch is not None
    assert b_npu2.brain._torch.device.type == "npu"
    assert b_npu2.brain.precision == "float32"
    assert b_npu2.pending_feedback
    np.testing.assert_allclose(b_npu2.brain.efficacy, b_cpu.brain.efficacy, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b_npu2.brain.bias, b_cpu.brain.bias, rtol=1e-5, atol=1e-5)

    updates = b_npu2.basal_ganglia.updates
    b_npu2.learn(np.array([1.0, 0.0]), np.array([False, False]), x)
    assert b_npu2.basal_ganglia.updates == updates + 1
    assert not b_npu2.pending_feedback
