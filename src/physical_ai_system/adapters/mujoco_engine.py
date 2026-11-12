"""MuJoCo humanoid simulation, mocap parser (CSV/NPZ/BVH), RL training runner, and MP4/3D export."""

from __future__ import annotations

import csv
import io
import json
import os
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

# Ensure headless OSMesa/EGL backend before importing mujoco if no DISPLAY exists
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "osmesa"

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from physical_ai_system.domain.models import (
    EpochMetrics,
    MocapTrajectory,
    TrainingConfig,
    TrainingReport,
)
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

# Built-in 2-legged bipedal humanoid MJCF (MuJoCo standard humanoid-style kinematic tree)
DEFAULT_HUMANOID_XML = """<mujoco model="physical_ai_biped_humanoid">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="0.01" gravity="0 0 -9.81" integrator="RK4"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.14 0.18 0.24" rgb2="0.22 0.27 0.35" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.15"/>
    <material name="body_mat" rgba="0.86 0.89 0.94 1"/>
    <material name="accent_mat" rgba="0.95 0.45 0.15 1"/>
    <material name="joint_mat" rgba="0.20 0.25 0.32 1"/>
  </asset>
  <worldbody>
    <light pos="2 -2 4" dir="-0.4 0.4 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>
    <geom name="floor" type="plane" size="20 20 0.1" material="grid_mat" condim="3"/>
    <body name="pelvis" pos="0 0 0.92">
      <freejoint name="root"/>
      <geom name="pelvis_geom" type="capsule" fromto="0 -0.10 0 0 0.10 0" size="0.08" material="accent_mat"/>
      <body name="torso" pos="0 0 0.12">
        <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-0.5 0.5" damping="4"/>
        <geom name="torso_geom" type="capsule" fromto="0 0 0 0 0 0.36" size="0.10" material="body_mat"/>
        <body name="head" pos="0 0 0.46">
          <geom name="head_geom" type="sphere" size="0.10" material="body_mat"/>
          <geom name="visor" type="box" pos="0.07 0 0.01" size="0.04 0.07 0.03" material="joint_mat"/>
        </body>
        <body name="right_upper_arm" pos="0 -0.18 0.32">
          <joint name="right_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
          <geom name="r_uarm" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.04" material="body_mat"/>
          <body name="right_forearm" pos="0 0 -0.25">
            <joint name="right_elbow" type="hinge" axis="0 1 0" range="-1.6 0.2" damping="1.5"/>
            <geom name="r_farm" type="capsule" fromto="0 0 0 0 0 -0.22" size="0.035" material="accent_mat"/>
          </body>
        </body>
        <body name="left_upper_arm" pos="0 0.18 0.32">
          <joint name="left_shoulder_pitch" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="2"/>
          <geom name="l_uarm" type="capsule" fromto="0 0 0 0 0 -0.24" size="0.04" material="body_mat"/>
          <body name="left_forearm" pos="0 0 -0.25">
            <joint name="left_elbow" type="hinge" axis="0 1 0" range="-1.6 0.2" damping="1.5"/>
            <geom name="l_farm" type="capsule" fromto="0 0 0 0 0 -0.22" size="0.035" material="accent_mat"/>
          </body>
        </body>
      </body>
      <body name="right_thigh" pos="0 -0.10 -0.05">
        <joint name="right_hip_pitch" type="hinge" axis="0 1 0" range="-1.2 1.2" damping="4"/>
        <geom name="r_thigh" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.055" material="joint_mat"/>
        <body name="right_shin" pos="0 0 -0.40">
          <joint name="right_knee" type="hinge" axis="0 1 0" range="-0.1 1.8" damping="3"/>
          <geom name="r_shin" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.045" material="body_mat"/>
          <body name="right_foot" pos="0.03 0 -0.40">
            <joint name="right_ankle" type="hinge" axis="0 1 0" range="-0.7 0.7" damping="2"/>
            <geom name="r_foot" type="box" size="0.10 0.05 0.03" material="accent_mat"/>
          </body>
        </body>
      </body>
      <body name="left_thigh" pos="0 0.10 -0.05">
        <joint name="left_hip_pitch" type="hinge" axis="0 1 0" range="-1.2 1.2" damping="4"/>
        <geom name="l_thigh" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.055" material="joint_mat"/>
        <body name="left_shin" pos="0 0 -0.40">
          <joint name="left_knee" type="hinge" axis="0 1 0" range="-0.1 1.8" damping="3"/>
          <geom name="l_shin" type="capsule" fromto="0 0 0 0 0 -0.38" size="0.045" material="body_mat"/>
          <body name="left_foot" pos="0.03 0 -0.40">
            <joint name="left_ankle" type="hinge" axis="0 1 0" range="-0.7 0.7" damping="2"/>
            <geom name="l_foot" type="box" size="0.10 0.05 0.03" material="accent_mat"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_waist" joint="waist_pitch" kp="80"/>
    <position name="a_r_sh" joint="right_shoulder_pitch" kp="40"/>
    <position name="a_r_el" joint="right_elbow" kp="30"/>
    <position name="a_l_sh" joint="left_shoulder_pitch" kp="40"/>
    <position name="a_l_el" joint="left_elbow" kp="30"/>
    <position name="a_r_hip" joint="right_hip_pitch" kp="120"/>
    <position name="a_r_knee" joint="right_knee" kp="100"/>
    <position name="a_r_ank" joint="right_ankle" kp="60"/>
    <position name="a_l_hip" joint="left_hip_pitch" kp="120"/>
    <position name="a_l_knee" joint="left_knee" kp="100"/>
    <position name="a_l_ank" joint="left_ankle" kp="60"/>
  </actuator>
</mujoco>"""

HUMANOID_JOINT_NAMES: tuple[str, ...] = (
    "waist_pitch",
    "right_shoulder_pitch",
    "right_elbow",
    "left_shoulder_pitch",
    "left_elbow",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
)


def generate_sample_biped_walk_mocap(
    mocap_id: str = "sample_biped_walk",
    name: str = "標準2足歩行モーション (Sample Walk)",
    num_frames: int = 120,
    fps: float = 30.0,
) -> MocapTrajectory:
    """Pure generator creating a physiologically plausible bipedal walking mocap reference."""
    t = np.linspace(0.0, num_frames / fps, num_frames)
    phase = 2.0 * np.pi * 1.1 * t  # 1.1 Hz gait cycle

    waist = 0.06 * np.sin(2.0 * phase)
    r_sh = 0.35 * np.sin(phase)
    r_el = -0.35 + 0.15 * np.sin(phase)
    l_sh = -0.35 * np.sin(phase)
    l_el = -0.35 - 0.15 * np.sin(phase)
    r_hip = -0.45 * np.sin(phase)
    r_knee = np.maximum(0.05, 0.65 * np.sin(phase + 0.5) + 0.25)
    r_ank = -0.20 * np.sin(phase + 0.2)
    l_hip = 0.45 * np.sin(phase)
    l_knee = np.maximum(0.05, -0.65 * np.sin(phase + 0.5) + 0.25)
    l_ank = 0.20 * np.sin(phase + 0.2)

    joints = np.stack(
        [waist, r_sh, r_el, l_sh, l_el, r_hip, r_knee, r_ank, l_hip, l_knee, l_ank],
        axis=1,
    )
    root_x = 0.65 * t
    root_y = 0.03 * np.sin(phase)
    root_z = 0.91 + 0.025 * np.abs(np.sin(phase))
    roots = np.stack([root_x, root_y, root_z], axis=1)

    return MocapTrajectory(
        mocap_id=mocap_id,
        name=name,
        source_format="builtin",
        fps=fps,
        joint_names=HUMANOID_JOINT_NAMES,
        joint_positions=tuple(tuple(float(v) for v in row) for row in joints),
        root_positions=tuple((float(r[0]), float(r[1]), float(r[2])) for r in roots),
    )


def parse_mocap_bytes(
    filename: str,
    raw_bytes: bytes,
    mocap_id: str,
    target_dim: int = len(HUMANOID_JOINT_NAMES),
) -> MocapTrajectory:
    """Pure parser for CSV, NPZ, and BVH motion capture files into normalized MocapTrajectory."""
    ext = Path(filename).suffix.lower()
    if ext == ".npz":
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as data:
            key = "qpos" if "qpos" in data else ("joint_positions" if "joint_positions" in data else list(data.keys())[0])
            arr = np.asarray(data[key], dtype=float)
            fps = float(data["fps"]) if "fps" in data else 30.0
    elif ext == ".bvh":
        arr, fps = _parse_bvh_text(raw_bytes.decode("utf-8", errors="replace"))
    else:
        # CSV fallback
        text = raw_bytes.decode("utf-8", errors="replace").strip()
        reader = csv.reader(io.StringIO(text))
        rows = []
        for row in reader:
            if not row:
                continue
            try:
                rows.append([float(x) for x in row])
            except ValueError:
                continue  # skip header row
        arr = np.asarray(rows, dtype=float)
        fps = 30.0

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 0:
        return generate_sample_biped_walk_mocap(mocap_id=mocap_id, name=filename)

    # Fit columns to target_dim cleanly
    if arr.shape[1] >= target_dim:
        joints = arr[:, :target_dim]
    else:
        pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=float)
        joints = np.hstack([arr, pad])

    # Normalize large degree values (BVH often in degrees) to radians
    if np.max(np.abs(joints)) > 6.5:
        joints = np.deg2rad(joints)
    joints = np.clip(joints, -1.8, 1.8)

    t = np.arange(len(joints)) / max(fps, 1.0)
    roots = np.stack([0.55 * t, np.zeros_like(t), np.full_like(t, 0.91)], axis=1)

    return MocapTrajectory(
        mocap_id=mocap_id,
        name=filename,
        source_format=ext.lstrip(".") or "csv",
        fps=fps,
        joint_names=HUMANOID_JOINT_NAMES,
        joint_positions=tuple(tuple(float(v) for v in row) for row in joints),
        root_positions=tuple((float(r[0]), float(r[1]), float(r[2])) for r in roots),
    )


def _parse_bvh_text(text: str) -> tuple[np.ndarray, float]:
    lines = text.splitlines()
    frame_time = 1.0 / 30.0
    in_motion = False
    frames: list[list[float]] = []
    for line in lines:
        s = line.strip()
        if s.upper() == "MOTION":
            in_motion = True
            continue
        if not in_motion:
            continue
        if s.lower().startswith("frame time:"):
            m = re.search(r"([0-9]*\.?[0-9]+)", s)
            if m:
                frame_time = max(1e-3, float(m.group(1)))
            continue
        if s.lower().startswith("frames:"):
            continue
        vals = [float(v) for v in s.split()]
        if vals:
            frames.append(vals)
    fps = round(1.0 / frame_time, 2)
    arr = np.asarray(frames, dtype=float) if frames else np.zeros((60, 11), dtype=float)
    # Skip first 3 root translation channels if >= 14 channels
    if arr.shape[1] >= 14:
        arr = arr[:, 3:]
    return arr, fps


def _extract_body_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, list[float]]:
    names = [
        "pelvis",
        "torso",
        "head",
        "right_upper_arm",
        "right_forearm",
        "left_upper_arm",
        "left_forearm",
        "right_thigh",
        "right_shin",
        "right_foot",
        "left_thigh",
        "left_shin",
        "left_foot",
    ]
    out: dict[str, list[float]] = {}
    for bname in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid >= 0:
            out[bname] = [round(float(x), 4) for x in data.xpos[bid]]
    return out


def run_imitation_and_rl_training(
    config: TrainingConfig,
    mocap: MocapTrajectory,
    xml_text: str | None = None,
    output_dir: Path = Path("./local_data/runs"),
    progress_cb: Callable[[int, int, EpochMetrics], None] | None = None,
) -> tuple[TrainingReport, bytes, dict]:
    """Execute MuJoCo imitation + RL (DAPG/RLPD/AWAC), render MP4 & 3D frames."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    model_xml = xml_text or DEFAULT_HUMANOID_XML
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)

    act_dim = model.nu
    ref_joints = np.asarray(mocap.joint_positions, dtype=np.float32)
    if ref_joints.shape[1] != act_dim:
        if ref_joints.shape[1] > act_dim:
            ref_joints = ref_joints[:, :act_dim]
        else:
            pad = np.zeros((len(ref_joints), act_dim - ref_joints.shape[1]), dtype=np.float32)
            ref_joints = np.hstack([ref_joints, pad])

    # Observation = [current_joint_pos (act_dim), current_joint_vel (act_dim), phase_sin, phase_cos, torso_z]
    obs_dim = act_dim * 2 + 3
    demo_obs_list: list[np.ndarray] = []
    demo_act_list: list[np.ndarray] = []
    demo_next_obs_list: list[np.ndarray] = []
    demo_rew_list: list[float] = []

    T = len(ref_joints)
    for t in range(T - 1):
        q = ref_joints[t]
        q_next = ref_joints[t + 1]
        qd = (q_next - q) * mocap.fps
        phase = 2.0 * np.pi * (t / max(T, 1))
        obs = np.concatenate([q, qd * 0.1, [np.sin(phase), np.cos(phase), 0.92]])
        next_phase = 2.0 * np.pi * ((t + 1) / max(T, 1))
        next_obs = np.concatenate([q_next, qd * 0.1, [np.sin(next_phase), np.cos(next_phase), 0.92]])
        demo_obs_list.append(obs)
        demo_act_list.append(np.clip(q_next, -1.0, 1.0))
        demo_next_obs_list.append(next_obs)
        demo_rew_list.append(1.0)

    demo_obs_all = torch.tensor(np.asarray(demo_obs_list), dtype=torch.float32)
    demo_act_all = torch.tensor(np.asarray(demo_act_list), dtype=torch.float32)
    demo_next_all = torch.tensor(np.asarray(demo_next_obs_list), dtype=torch.float32)
    demo_rew_all = torch.tensor(np.asarray(demo_rew_list), dtype=torch.float32)

    # Train/Validation split (80/20 contiguous blocks for genuine generalization measurement)
    split_idx = max(4, int(0.8 * len(demo_obs_all)))
    train_obs, val_obs = demo_obs_all[:split_idx], demo_obs_all[split_idx:]
    train_act, val_act = demo_act_all[:split_idx], demo_act_all[split_idx:]

    actor = ActorNetwork(obs_dim, act_dim)
    critic = TwinQNetwork(obs_dim, act_dim)
    target_critic = TwinQNetwork(obs_dim, act_dim)
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    # Phase 1: Imitation Learning Warmup (BC pretraining)
    bc_pretrain_steps = 20
    for _ in range(bc_pretrain_steps):
        actor_opt.zero_grad()
        loss_bc = compute_bc_loss(actor, train_obs, train_act, config.smoothness_weight)
        loss_bc.backward()
        actor_opt.step()

    def _rollout_mujoco(max_steps: int = 80, explore: bool = True) -> tuple[TransitionBatch, float, float, np.ndarray]:
        mujoco.mj_resetData(model, data)
        if model.nq >= 7 + act_dim:
            data.qpos[2] = 0.92
            data.qpos[3] = 1.0
            data.qpos[7 : 7 + act_dim] = ref_joints[0]
        mujoco.mj_forward(model, data)

        obs_buf, act_buf, rew_buf, next_buf, done_buf = [], [], [], [], []
        executed_joints = []
        track_errors = []

        for step in range(max_steps):
            ref_idx = step % T
            q_cur = data.qpos[7 : 7 + act_dim].copy() if model.nq >= 7 + act_dim else np.zeros(act_dim)
            qd_cur = data.qvel[6 : 6 + act_dim].copy() if model.nv >= 6 + act_dim else np.zeros(act_dim)
            phase = 2.0 * np.pi * (ref_idx / max(T, 1))
            torso_z = float(data.qpos[2]) if model.nq >= 3 else 0.92
            obs_np = np.concatenate([q_cur, qd_cur * 0.1, [np.sin(phase), np.cos(phase), torso_z]]).astype(np.float32)

            with torch.no_grad():
                mean, std = actor(torch.from_numpy(obs_np).unsqueeze(0))
                if explore:
                    act_t = mean + std * torch.randn_like(std)
                else:
                    act_t = mean
                act_np = np.clip(act_t.squeeze(0).numpy(), -1.0, 1.0)

            data.ctrl[:act_dim] = act_np
            for _ in range(3):
                mujoco.mj_step(model, data)

            q_next = data.qpos[7 : 7 + act_dim].copy() if model.nq >= 7 + act_dim else act_np
            qd_next = data.qvel[6 : 6 + act_dim].copy() if model.nv >= 6 + act_dim else np.zeros(act_dim)
            next_z = float(data.qpos[2]) if model.nq >= 3 else 0.92
            next_phase = 2.0 * np.pi * (((ref_idx + 1) % T) / max(T, 1))
            next_obs_np = np.concatenate([q_next, qd_next * 0.1, [np.sin(next_phase), np.cos(next_phase), next_z]]).astype(np.float32)

            err = float(np.mean((q_next - ref_joints[(ref_idx + 1) % T]) ** 2))
            track_errors.append(err)
            upright_bonus = 1.0 if next_z > 0.62 else -1.0
            reward = float(
                config.reward_tracking_weight * np.exp(-3.0 * err) + 0.5 * upright_bonus
            )
            done = 1.0 if next_z < 0.45 else 0.0

            obs_buf.append(obs_np)
            act_buf.append(act_np)
            rew_buf.append(reward)
            next_buf.append(next_obs_np)
            done_buf.append(done)
            executed_joints.append(q_next.copy())
            if done > 0.5:
                break

        batch = TransitionBatch(
            obs=torch.tensor(np.asarray(obs_buf), dtype=torch.float32),
            act=torch.tensor(np.asarray(act_buf), dtype=torch.float32),
            rew=torch.tensor(np.asarray(rew_buf), dtype=torch.float32),
            next_obs=torch.tensor(np.asarray(next_buf), dtype=torch.float32),
            done=torch.tensor(np.asarray(done_buf), dtype=torch.float32),
        )
        mean_ret = float(np.sum(rew_buf))
        mean_err = float(np.sqrt(np.mean(track_errors))) if track_errors else 1.0
        return batch, mean_ret, mean_err, np.asarray(executed_joints, dtype=np.float32)

    metrics_history: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        online_batch, mean_ret, track_err, exec_traj = _rollout_mujoco(max_steps=min(90, T), explore=True)

        # Sample offline demo batch
        idx = rng.integers(0, len(train_obs), size=min(config.batch_size, len(train_obs)))
        off_obs = train_obs[idx]
        off_act = train_act[idx]

        if config.method == "DAPG":
            actor_opt.zero_grad()
            loss, _ = compute_dapg_loss(
                actor,
                online_batch,
                off_obs,
                off_act,
                epoch,
                config.bc_weight,
                config.smoothness_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            train_loss_val = float(loss.item())
        else:
            # Mix 50% offline demo + 50% online rollout (RLPD / AWAC)
            n_mix = min(len(online_batch.obs), len(demo_obs_all))
            d_idx = rng.integers(0, len(demo_obs_all), size=n_mix)
            mixed_batch = TransitionBatch(
                obs=torch.cat([online_batch.obs[:n_mix], demo_obs_all[d_idx]], dim=0),
                act=torch.cat([online_batch.act[:n_mix], demo_act_all[d_idx]], dim=0),
                rew=torch.cat([online_batch.rew[:n_mix], demo_rew_all[d_idx]], dim=0),
                next_obs=torch.cat([online_batch.next_obs[:n_mix], demo_next_all[d_idx]], dim=0),
                done=torch.cat([online_batch.done[:n_mix], torch.zeros(n_mix)], dim=0),
            )
            critic_opt.zero_grad()
            c_loss = compute_critic_loss(critic, target_critic, actor, mixed_batch)
            c_loss.backward()
            critic_opt.step()

            actor_opt.zero_grad()
            if config.method == "RLPD":
                a_loss, _ = compute_rlpd_actor_loss(
                    actor,
                    critic,
                    mixed_batch.obs,
                    off_obs,
                    off_act,
                    config.bc_weight,
                    config.entropy_coef,
                    config.smoothness_weight,
                )
            else:
                a_loss, _ = compute_awac_actor_loss(
                    actor,
                    critic,
                    mixed_batch,
                    off_obs,
                    off_act,
                    config.bc_weight,
                    config.smoothness_weight,
                )
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            # Polyak target critic update
            with torch.no_grad():
                for p, tp in zip(critic.parameters(), target_critic.parameters()):
                    tp.data.mul_(0.95).add_(0.05 * p.data)
            train_loss_val = float(
                compute_bc_loss(actor, train_obs, train_act, config.smoothness_weight).item()
            )

        with torch.no_grad():
            val_loss_val = float(
                compute_bc_loss(actor, val_obs, val_act, config.smoothness_weight).item()
            )

        jerk = compute_jerk_score(exec_traj)
        # Task success rate combines upright survival ratio & low tracking error
        survival_ratio = float(len(exec_traj)) / float(min(90, T))
        success_rate = float(np.clip(survival_ratio * np.exp(-1.1 * track_err), 0.0, 1.0))

        ep_metric = EpochMetrics(
            epoch=epoch,
            train_loss=round(abs(train_loss_val), 5),
            val_loss=round(abs(val_loss_val), 5),
            task_success_rate=round(success_rate, 4),
            mean_return=round(mean_ret, 3),
            tracking_error=round(track_err, 4),
            jerk_smoothness=round(jerk, 5),
        )
        metrics_history.append(ep_metric)
        if progress_cb:
            progress_cb(epoch, config.epochs, ep_metric)

    # Final Deterministic Evaluation Rollout + MuJoCo Offscreen MP4 + 3D Joint Frames
    run_dir = output_dir / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = run_dir / "evaluation.mp4"

    video_bytes, frames_payload = _render_mujoco_evaluation(
        model=model,
        actor=actor,
        ref_joints=ref_joints,
        mp4_path=mp4_path,
        steps=min(120, T),
    )

    diagnostics = evaluate_training_diagnostics(tuple(metrics_history), config)
    report = TrainingReport(
        run_id=config.run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        config=config,
        metrics_history=tuple(metrics_history),
        diagnostics=diagnostics,
        video_s3_key=f"report/{config.run_id}/evaluation.mp4",
        frames_s3_key=f"report/{config.run_id}/frames_3d.json",
        report_s3_key=f"report/{config.run_id}/report.json",
        video_local_url=f"/api/runs/{config.run_id}/video",
        frames_local_url=f"/api/runs/{config.run_id}/frames",
    )
    return report, video_bytes, frames_payload


def _render_mujoco_evaluation(
    model: mujoco.MjModel,
    actor: ActorNetwork,
    ref_joints: np.ndarray,
    mp4_path: Path,
    steps: int = 120,
) -> tuple[bytes, dict]:
    """Render MuJoCo evaluation rollout to MP4 and extract 3D body positions for Three.js."""
    data = mujoco.MjData(model)
    act_dim = model.nu
    T = len(ref_joints)

    mujoco.mj_resetData(model, data)
    if model.nq >= 7 + act_dim:
        data.qpos[2] = 0.92
        data.qpos[3] = 1.0
        data.qpos[7 : 7 + act_dim] = ref_joints[0]
    mujoco.mj_forward(model, data)

    renderer: mujoco.Renderer | None = None
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        renderer = None

    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.4, 0.0, 0.85]
    camera.distance = 2.8
    camera.azimuth = 135
    camera.elevation = -18

    video_frames: list[np.ndarray] = []
    frames_3d: list[dict] = []

    for step in range(steps):
        ref_idx = step % T
        q_cur = data.qpos[7 : 7 + act_dim].copy() if model.nq >= 7 + act_dim else np.zeros(act_dim)
        qd_cur = data.qvel[6 : 6 + act_dim].copy() if model.nv >= 6 + act_dim else np.zeros(act_dim)
        phase = 2.0 * np.pi * (ref_idx / max(T, 1))
        torso_z = float(data.qpos[2]) if model.nq >= 3 else 0.92
        obs_np = np.concatenate([q_cur, qd_cur * 0.1, [np.sin(phase), np.cos(phase), torso_z]]).astype(np.float32)

        with torch.no_grad():
            mean, _ = actor(torch.from_numpy(obs_np).unsqueeze(0))
            act_np = np.clip(mean.squeeze(0).numpy(), -1.0, 1.0)

        data.ctrl[:act_dim] = act_np
        for _ in range(3):
            mujoco.mj_step(model, data)

        # Track pelvis X in camera
        if model.nq >= 3:
            camera.lookat[0] = float(data.qpos[0])
        frames_3d.append(
            {
                "step": step,
                "bodies": _extract_body_positions(model, data),
                "qpos_policy": [round(float(v), 4) for v in q_cur],
                "qpos_ref": [round(float(v), 4) for v in ref_joints[ref_idx]],
            }
        )
        if renderer is not None and step % 2 == 0:
            renderer.update_scene(data, camera=camera)
            video_frames.append(renderer.render())

    if renderer is not None:
        renderer.close()

    if video_frames:
        imageio.mimsave(str(mp4_path), video_frames, fps=15)
        video_bytes = mp4_path.read_bytes()
    else:
        video_bytes = b""

    frames_payload = {
        "fps": 15,
        "joint_names": list(HUMANOID_JOINT_NAMES),
        "frames": frames_3d,
    }
    return video_bytes, frames_payload
