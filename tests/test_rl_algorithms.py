"""Unit tests for DAPG, RLPD, AWAC pure loss functions and diagnostics."""

import numpy as np
import torch

from physical_ai_system.domain.advisor import parse_retrain_instruction, recommend_method_pure
from physical_ai_system.domain.models import EpochMetrics, TrainingConfig
from physical_ai_system.domain.rl_algorithms import (
    ActorNetwork,
    TransitionBatch,
    TwinQNetwork,
    compute_awac_actor_loss,
    compute_bc_loss,
    compute_critic_loss,
    compute_dapg_loss,
    compute_jerk_score,
    compute_rlpd_actor_loss,
    evaluate_training_diagnostics,
)


def _dummy_batch(bs: int = 16, obs_dim: int = 25, act_dim: int = 11) -> TransitionBatch:
    return TransitionBatch(
        obs=torch.randn(bs, obs_dim),
        act=torch.tanh(torch.randn(bs, act_dim)),
        rew=torch.rand(bs),
        next_obs=torch.randn(bs, obs_dim),
        done=torch.zeros(bs),
    )


def test_dapg_rlpd_awac_losses_are_finite() -> None:
    actor = ActorNetwork(25, 11)
    critic = TwinQNetwork(25, 11)
    target = TwinQNetwork(25, 11)
    batch = _dummy_batch()

    bc = compute_bc_loss(actor, batch.obs, batch.act, smoothness_weight=0.05)
    dapg, _ = compute_dapg_loss(actor, batch, batch.obs, batch.act, epoch=1, bc_weight=1.0, smoothness_weight=0.05)
    c_loss = compute_critic_loss(critic, target, actor, batch)
    rlpd, _ = compute_rlpd_actor_loss(actor, critic, batch.obs, batch.obs, batch.act, 1.0, 0.01, 0.05)
    awac, _ = compute_awac_actor_loss(actor, critic, batch, batch.obs, batch.act, 1.0, 0.05)

    for t in (bc, dapg, c_loss, rlpd, awac):
        assert torch.isfinite(t)


def test_diagnostics_detects_overfitting_and_jerk() -> None:
    cfg = TrainingConfig(run_id="r1", method="DAPG", mocap_id="m1")
    history = tuple(
        EpochMetrics(
            epoch=i,
            train_loss=0.05,
            val_loss=0.35,
            task_success_rate=0.45,
            mean_return=12.0,
            tracking_error=0.4,
            jerk_smoothness=0.25,
        )
        for i in range(1, 6)
    )
    diag = evaluate_training_diagnostics(history, cfg)
    assert diag.overfitting_detected is True
    assert diag.is_smooth is False
    assert diag.follows_human_motion is False

    # Smooth trajectory check
    t = np.linspace(0, 1, 50)[:, None]
    smooth_traj = np.sin(t * np.pi)
    assert compute_jerk_score(smooth_traj) < 1e-4


def test_advisor_recommendations_and_retrain_parser() -> None:
    method, cfg, reply = recommend_method_pure("過学習を抑えて滑らかにしたい", mocap_frames=120)
    assert method == "AWAC"
    assert "AWAC" in reply

    updated, note = parse_retrain_instruction("振動を抑えて滑らかに、過学習も減らしてRLPDで35エポック", cfg)
    assert updated.method == "RLPD"
    assert updated.epochs == 35
    assert updated.smoothness_weight > cfg.smoothness_weight
    assert "滑らか" in note
