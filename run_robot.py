#!/usr/bin/env python
"""Interactive Pi05 robot control - using original project processing pipeline."""
import os
os.environ.setdefault("DISPLAY", ":2")
os.environ["PYTHONUNBUFFERED"] = "1"

import sys
sys.stdout.reconfigure(line_buffering=True)

import mujoco
import mujoco.viewer
import numpy as np
import torch
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
import time
import threading
import queue

MODEL_PATH = os.environ.get("PI05_MODEL_PATH", "./models/pi05_libero_finetuned")
cmd_queue = queue.Queue()

def quat_to_axisangle(quat):
    """Convert quaternion (w, x, y, z) to axis-angle representation."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    sin_half = torch.sqrt(x**2 + y**2 + z**2)
    cos_half = w
    angle = 2 * torch.atan2(sin_half, cos_half)
    eps = 1e-8
    axis = torch.stack([x, y, z], dim=-1) / (sin_half.unsqueeze(-1) + eps)
    axisangle = axis * angle.unsqueeze(-1)
    return axisangle

def input_thread():
    """Thread to read commands from terminal."""
    print("\n=== Commands ===")
    print("Enter task (e.g., 'pick up the alphabet soup')")
    print("'reset' - reset environment")
    print("'quit'  - exit program\n", flush=True)

    while True:
        try:
            cmd = input("> ").strip()
            if cmd:
                cmd_queue.put(cmd)
                if cmd == "quit":
                    break
        except (EOFError, KeyboardInterrupt):
            cmd_queue.put("quit")
            break

def main():
    print("=== Pi05 Robot Controller ===")

    # Select task suite
    print("\nSelect task suite:")
    print("1. libero_object  - 不同物体泛化")
    print("2. libero_spatial - 空间关系理解")
    print("3. libero_goal    - 不同动作目标")

    choice = input("Enter choice (1/2/3) [default=1]: ").strip() or "1"

    suite_map = {
        "1": "libero_object",
        "2": "libero_spatial",
        "3": "libero_goal"
    }
    suite_name = suite_map.get(choice, "libero_object")
    print(f"\nUsing: {suite_name}")

    print("\nStep 1: Loading Pi05 model...")
    policy = PI05Policy.from_pretrained(MODEL_PATH)
    print("Step 2: Model loaded")
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    print(f"Step 3: Model on {device}")

    print("Step 4: Creating preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(policy.config.type, MODEL_PATH)
    print("Step 5: Preprocessors ready")

    def process_libero_obs(obs):
        """Process LIBERO observation like the original project."""
        processed = preprocess_observation(obs)

        # Flip images 180 degrees (LIBERO camera convention)
        for key in list(processed.keys()):
            if key.startswith("observation.images."):
                img = processed[key]
                processed[key] = torch.flip(img, dims=[2, 3])

        # Convert robot_state to observation.state with axis-angle
        if "observation.robot_state" in processed:
            robot_state = processed.pop("observation.robot_state")
            eef_pos = robot_state["eef"]["pos"]
            eef_quat = robot_state["eef"]["quat"]
            gripper_qpos = robot_state["gripper"]["qpos"]
            eef_axisangle = quat_to_axisangle(eef_quat)
            state = torch.cat([eef_pos, eef_axisangle, gripper_qpos], dim=-1).float()
            processed["observation.state"] = state

        return processed

    print("Loading environment...")
    task_suite = benchmark.get_benchmark_dict()[suite_name]()

    print(f"\nAvailable tasks in {suite_name}:")
    for i, task in enumerate(task_suite.tasks):
        print(f"  {i}: {task.language}")

    env = LiberoEnv(
        task_suite=task_suite,
        task_id=0,
        task_suite_name=suite_name,
        render_mode="rgb_array",
        obs_type="pixels_agent_pos"
    )
    obs, _ = env.reset()
    task_description = env.task_description
    print(f"Task: {task_description}")

    # Get MuJoCo model and data
    mj_model = env._env.env.sim.model._model
    mj_data = env._env.env.sim.data._data

    # Warmup
    print("Warming up model...")
    warmup_obs = process_libero_obs(obs)
    warmup_obs["task"] = task_description
    warmup_obs = preprocessor(warmup_obs)
    for k, v in warmup_obs.items():
        if isinstance(v, torch.Tensor):
            warmup_obs[k] = v.to(device)
    with torch.no_grad():
        _ = policy.select_action(warmup_obs)
    print("Model ready!")

    # Start input thread
    input_thd = threading.Thread(target=input_thread, daemon=True)
    input_thd.start()

    print("\nOpening MuJoCo viewer...")
    print("Controls: Left-drag=orbit, Scroll=zoom, Double-click=center on object")

    current_command = None
    step = 0

    try:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.8
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            print(f"Viewer started, is_running={viewer.is_running()}")

            while viewer.is_running():
                try:
                    cmd = cmd_queue.get_nowait()
                    if cmd == "quit":
                        break
                    if cmd == "reset":
                        print("\nResetting environment...")
                        obs, _ = env.reset()
                        current_command = None
                        step = 0
                        print("Environment reset. Ready for new command.")
                        continue
                    current_command = cmd
                    print(f"\nExecuting: {current_command}")
                    step = 0
                except queue.Empty:
                    pass

                if current_command and step < 280:
                    processed_obs = process_libero_obs(obs)
                    processed_obs["task"] = current_command
                    processed_obs = preprocessor(processed_obs)

                    for k, v in processed_obs.items():
                        if isinstance(v, torch.Tensor):
                            processed_obs[k] = v.to(device)

                    if step == 0:
                        tokens = processed_obs.get('observation.language.tokens')
                        if tokens is not None:
                            print(f"Language tokens (first 20): {tokens[0,:20].tolist()}")

                    with torch.no_grad():
                        action = policy.select_action(processed_obs)

                    action = postprocessor(action)
                    action_np = action.cpu().numpy().squeeze()

                    if step < 3:
                        print(f"Step {step} action: {action_np}")

                    obs, reward, done, truncated, info = env.step(action_np)

                    # Sync to viewer
                    sim_data = env._env.env.sim.data._data
                    mj_data.qpos[:] = sim_data.qpos
                    mj_data.qvel[:] = sim_data.qvel
                    mujoco.mj_forward(mj_model, mj_data)

                    step += 1
                    if step % 50 == 0:
                        print(f"Step {step}/280")

                    if done or info.get("success", False):
                        print(f"Task complete! Steps: {step}, Success: {info.get('success', False)}")
                        print("Environment preserved. Enter new command or 'reset' to restart.")
                        current_command = None

                elif current_command and step >= 280:
                    print(f"Step limit (280) reached. Task not completed.")
                    print("Environment preserved. Enter new command or 'reset' to restart.")
                    current_command = None

                viewer.sync()
                time.sleep(0.005)  # Reduced from 0.02 for faster simulation
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    env.close()
    print("Bye!")

if __name__ == "__main__":
    main()
