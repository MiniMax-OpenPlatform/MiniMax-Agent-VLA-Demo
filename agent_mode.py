#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Agent Mode - LLM-based task planning with VLA execution."""
import os
os.environ["DISPLAY"] = ":2"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["ANTHROPIC_BASE_URL"] = "https://api.minimaxi.com/anthropic"
os.environ["ANTHROPIC_API_KEY"] = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiIvIiwiVXNlck5hbWUiOiLlhq_pm68iLCJBY2NvdW50IjoiIiwiU3ViamVjdElEIjoiMTY4NjgzMzY3NzA0Njc4MCIsIlBob25lIjoiMTg4MTE0NDU3MjgiLCJHcm91cElEIjoiMTY4NjgzMzY3NzM2NTQ1OSIsIlBhZ2VOYW1lIjoiIiwiTWFpbCI6IjE3ODk5ODExMTNAcXEuY29tIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTEtMTUgMTE6NDA6MjgiLCJUb2tlblR5cGUiOjQsImlzcyI6Im1pbmltYXgifQ.wc8Gf75e8fJTaZO8DvIsocDxrSUYVnHuMXCPvnAin6gtYc1swnlcpCFeBCIpU28Tqc8KXTqdcmU52hkE52QeLfHJFQ2q0wn_Gq68r8EHzRUg3tHwuiBS1wD58G9hKqOXMgclRIcCIaxs4fbrzAWw9XFZy4-iofKd0Nzjd3gCD7U7yFmfPNtZDTv8_oH_tXdaW5pdWblet-cX3uvp2EsWUhakSBK1rXgaXdYR24cv5aqwqy_YOOqMt6aAWeeLHM3iU1vdIP3T-m8JQiytxYOwbDy2jAjj1h1eZyWv6dil9TNWwowOIgFAezQoSCgjX-pgIR3De6B4eFjlSn3lCz16gg"

import sys
import io
sys.path.insert(0, "/data1/devin/robot1/lerobot/src")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

import json
import mujoco
import mujoco.viewer
import numpy as np
import torch
import anthropic
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
import time
import threading
import queue

MODEL_PATH = os.environ.get("PI05_MODEL_PATH", "./models/pi05_libero_finetuned")

# Tools for LLM agent
AGENT_TOOLS = [
    {
        "name": "execute_task",
        "description": "Execute a manipulation task using the robot arm. Runs VLA for up to 280 steps. Does NOT return success/failure - use get_scene_info to verify result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A manipulation task instruction, e.g., 'open the drawer', 'close the drawer', 'pick up the butter'"
                }
            },
            "required": ["task"]
        }
    },
    {
        "name": "get_scene_info",
        "description": "Capture camera image and use VLM to analyze the current scene. Returns visual understanding of objects, their locations, drawer states, and robot arm status.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


class RobotAgent:
    """Agent that uses LLM for planning and VLA for execution."""

    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
            api_key=api_key,
            default_headers={"Authorization": f"Bearer {api_key}"}
        )
        self.policy = None
        self.env = None
        self.preprocessor = None
        self.postprocessor = None
        self.device = None
        self.obs = None
        self.last_task_success = False
        self.last_task_steps = 0
        self.last_executed_task = None  # Track last task for verification
        self.mj_model = None
        self.mj_data = None
        self.viewer = None
        self._env_terminated = False
        self.suite_name = None
        self.conversation_history = []  # Preserve history between user requests

    def initialize(self, suite_name="libero_object"):
        """Initialize the robot environment and model."""
        print("=== Initializing Robot Agent ===")
        self.suite_name = suite_name

        # Load Pi05 model
        print("Loading Pi05 model...")
        self.policy = PI05Policy.from_pretrained(MODEL_PATH)
        self.policy.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        print(f"Model on {self.device}")

        # Create preprocessors
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config.type, MODEL_PATH
        )

        # Load environment
        print(f"Loading {suite_name} environment...")
        task_suite = benchmark.get_benchmark_dict()[suite_name]()

        self.env = LiberoEnv(
            task_suite=task_suite,
            task_id=0,
            task_suite_name=suite_name,
            render_mode="rgb_array",
            obs_type="pixels_agent_pos"
        )
        self.env.auto_reset = False  # Disable auto-reset for agent mode
        self.obs, _ = self.env.reset()

        # Get MuJoCo model and data
        self.mj_model = self.env._env.env.sim.model._model
        self.mj_data = self.env._env.env.sim.data._data

        # Warmup
        print("Warming up model...")
        warmup_obs = self._process_obs(self.obs)
        warmup_obs["task"] = "pick up the butter"
        warmup_obs = self.preprocessor(warmup_obs)
        for k, v in warmup_obs.items():
            if isinstance(v, torch.Tensor):
                warmup_obs[k] = v.to(self.device)
        with torch.no_grad():
            _ = self.policy.select_action(warmup_obs)

        print("Robot Agent ready!")
        return True

    def _quat_to_axisangle(self, quat):
        """Convert quaternion to axis-angle."""
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
        sin_half = torch.sqrt(x**2 + y**2 + z**2)
        cos_half = w
        angle = 2 * torch.atan2(sin_half, cos_half)
        eps = 1e-8
        axis = torch.stack([x, y, z], dim=-1) / (sin_half.unsqueeze(-1) + eps)
        return axis * angle.unsqueeze(-1)

    def _reset_robot_pose(self):
        """Reset robot arm to initial pose without resetting scene objects."""
        try:
            env = self.env._env.env
            sim = env.sim
            robot = env.robots[0]

            # Get robot joint indices
            robot_qpos_indices = robot._ref_joint_pos_indexes
            robot_qvel_indices = robot._ref_joint_vel_indexes

            # Reset to robot's initial joint positions
            init_qpos = robot.init_qpos
            sim.data.qpos[robot_qpos_indices] = init_qpos
            sim.data.qvel[robot_qvel_indices] = 0

            # Also reset gripper
            if hasattr(robot, 'gripper') and robot.gripper is not None:
                gripper_qpos_indices = robot.gripper._ref_joint_pos_indexes if hasattr(robot.gripper, '_ref_joint_pos_indexes') else []
                gripper_qvel_indices = robot.gripper._ref_joint_vel_indexes if hasattr(robot.gripper, '_ref_joint_vel_indexes') else []
                if len(gripper_qpos_indices) > 0:
                    sim.data.qpos[gripper_qpos_indices] = robot.gripper.init_qpos
                    sim.data.qvel[gripper_qvel_indices] = 0

            # Reset robot controller state
            if hasattr(robot, 'controller') and robot.controller is not None:
                controller = robot.controller
                # For OSC controller, reset its internal goal to current (reset) pose
                if hasattr(controller, 'goal_pos'):
                    controller.goal_pos = None
                if hasattr(controller, 'goal_ori'):
                    controller.goal_ori = None
                if hasattr(controller, 'reset'):
                    controller.reset()

            # Forward simulation to update state
            sim.forward()

            # Debug: print robot joint angles after reset
            print(f"  📍 Reset qpos: {sim.data.qpos[robot_qpos_indices][:3]}...")

            # Run a few simulation steps to stabilize
            for _ in range(5):
                sim.step()
            sim.forward()

            # Sync viewer if available
            if self.viewer and self.viewer.is_running():
                self.mj_data.qpos[:] = sim.data._data.qpos
                self.mj_data.qvel[:] = sim.data._data.qvel
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer.sync()

            # Update observation
            self.obs = self.env._format_raw_obs(env._get_observations())
            print("  🔄 Robot pose reset to initial state")
        except Exception as e:
            print(f"  ⚠️ Could not reset robot pose: {e}")

    def _process_obs(self, obs):
        """Process observation for Pi05."""
        processed = preprocess_observation(obs)

        for key in list(processed.keys()):
            if key.startswith("observation.images."):
                img = processed[key]
                processed[key] = torch.flip(img, dims=[2, 3])

        if "observation.robot_state" in processed:
            robot_state = processed.pop("observation.robot_state")
            eef_pos = robot_state["eef"]["pos"]
            eef_quat = robot_state["eef"]["quat"]
            gripper_qpos = robot_state["gripper"]["qpos"]
            eef_axisangle = self._quat_to_axisangle(eef_quat)
            state = torch.cat([eef_pos, eef_axisangle, gripper_qpos], dim=-1).float()
            processed["observation.state"] = state

        return processed

    def execute_task(self, task: str, max_steps: int = 280) -> dict:
        """Execute a single task using VLA."""
        print(f"\n🤖 Executing: {task}")

        # Save last task for verification
        self.last_executed_task = task

        # Reset robot arm to initial pose (without resetting scene objects)
        self._reset_robot_pose()

        # Reset policy's action queue for new task (keeps environment state)
        self.policy.reset()

        self.last_task_success = False
        self.last_task_steps = 0

        for step in range(max_steps):
            # Process observation
            processed_obs = self._process_obs(self.obs)
            processed_obs["task"] = task
            processed_obs = self.preprocessor(processed_obs)

            for k, v in processed_obs.items():
                if isinstance(v, torch.Tensor):
                    processed_obs[k] = v.to(self.device)

            # Get action
            with torch.no_grad():
                action = self.policy.select_action(processed_obs)
            action = self.postprocessor(action)
            action_np = action.cpu().numpy().squeeze()

            if step < 3:
                print(f"  Step {step}: action={action_np[:3]}...")

            # Step environment
            self.obs, reward, done, truncated, info = self.env.step(action_np)

            # Sync viewer if available
            if self.viewer and self.viewer.is_running():
                sim_data = self.env._env.env.sim.data._data
                self.mj_data.qpos[:] = sim_data.qpos
                self.mj_data.qvel[:] = sim_data.qvel
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer.sync()

            self.last_task_steps = step + 1

            if step % 50 == 0:
                print(f"  Step {step}/{max_steps}")

        print(f"⏱️ Task completed: {max_steps} steps executed")
        return {
            "steps": max_steps,
            "message": f"Executed {max_steps} steps. Use get_scene_info to verify result."
        }

    def _call_mcp_understand_image(self, image_path: str, prompt: str) -> str:
        """Call MCP understand_image tool using mcp client library."""
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        import uuid

        # Generate unique request ID for logging
        request_id = str(uuid.uuid4())[:8]
        print(f"  🔗 MCP Request ID: {request_id}")
        print(f"  📷 Image: {image_path}")
        print(f"  💬 Prompt: {prompt[:100]}...")

        async def call_mcp():
            mcp_env = {
                **os.environ,
                "MINIMAX_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
                "MINIMAX_API_HOST": "https://api.minimaxi.com"
            }

            server_params = StdioServerParameters(
                command="uvx",
                args=["minimax-coding-plan-mcp", "-y"],
                env=mcp_env
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    result = await session.call_tool("understand_image", {
                        "image_source": image_path,
                        "prompt": prompt
                    })

                    if result.content and len(result.content) > 0:
                        return result.content[0].text
                    return "No analysis available"

        try:
            # Suppress stderr logging for this call
            import logging
            logging.getLogger("mcp.client.stdio").setLevel(logging.CRITICAL)
            result = asyncio.run(call_mcp())
            logging.getLogger("mcp.client.stdio").setLevel(logging.WARNING)
            return result
        except Exception as e:
            return f"MCP error: {str(e)}"

    def get_scene_info(self) -> dict:
        """Get current scene information using visual understanding via MCP."""
        from PIL import Image
        import time

        # Capture current camera image from environment
        try:
            # Force update physics and render to get fresh observation
            env = self.env._env.env
            env.sim.forward()

            # Render fresh image directly from sim (bypasses observation caching)
            image_array = env.sim.render(
                camera_name="agentview",
                width=256,
                height=256,
                depth=False
            )
            # Flip vertically (MuJoCo renders upside-down)
            image_array = image_array[::-1]
            if image_array is None:
                return {"error": "No camera image available"}

            # Save image to temp file with timestamp to avoid caching
            timestamp = int(time.time() * 1000)
            temp_path = f"/tmp/scene_capture_{timestamp}.png"
            image = Image.fromarray(image_array)
            image.save(temp_path)
            print(f"  📸 Captured scene image: {temp_path}")

            # Generate task-specific verification prompt
            if self.last_executed_task:
                prompt = f"""This image shows the result after attempting: "{self.last_executed_task}"

Please verify:
1. Was the task completed successfully? Look for evidence that matches the task goal.
2. Current state of objects: Where are the bowl, plate, wine bottle, cream cheese?
3. Drawer states: Which drawers (top/middle/bottom) are open or closed?
4. What is the robot arm doing now?

Be specific about whether the task "{self.last_executed_task}" was successful or not."""
            else:
                prompt = """Analyze this robot manipulation scene:
1. What objects do you see? (bowl, plate, wine bottle, cream cheese, etc.)
2. Where are objects located? (table, stove, cabinet top, drawer, rack)
3. Drawer states: Which drawers are open or closed?
4. Robot arm position and status.
Be concise."""

            analysis = self._call_mcp_understand_image(temp_path, prompt)

            return {
                "visual_analysis": analysis,
                "image_path": temp_path,
                "verified_task": self.last_executed_task
            }

        except Exception as e:
            return {"error": f"Scene capture failed: {str(e)}"}

    def _get_scene_details(self) -> dict:
        """Get scene-specific details."""
        if "libero_goal" in self.suite_name:
            return {
                "cabinet": "Has 3 drawers: top drawer, middle drawer, bottom drawer",
                "possible_tasks": [
                    "open the top drawer of the cabinet",
                    "open the middle drawer of the cabinet",
                    "open the bottom drawer of the cabinet"
                ]
            }
        elif "libero_object" in self.suite_name:
            return {
                "table": "Contains food items and a basket",
                "possible_tasks": ["pick up the [item] and place it in the basket"]
            }
        elif "libero_spatial" in self.suite_name:
            return {
                "table": "Contains bowls, plates, and other items",
                "possible_tasks": ["pick up the [item] and place it on the [target]"]
            }
        return {}

    def reset_environment(self) -> dict:
        """Reset the environment."""
        print("\n🔄 Resetting environment...")
        self.obs, _ = self.env.reset()
        self.last_task_success = False
        self.last_task_steps = 0
        self.conversation_history = []  # Clear conversation history on reset
        self._env_terminated = False
        return {"message": "Environment reset successfully"}

    def process_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Process a tool call from the LLM."""
        if tool_name == "execute_task":
            task = tool_input["task"]
            # Validate task is in supported list
            SUPPORTED_TASKS = [
                "open the middle drawer of the cabinet",
                "put the bowl on the stove",
                "put the wine bottle on top of the cabinet",
                "open the top drawer and put the bowl inside",
                "put the bowl on top of the cabinet",
                "push the plate to the front of the stove",
                "put the cream cheese in the bowl",
                "turn on the stove",
                "put the bowl on the plate",
                "put the wine bottle on the rack"
            ]
            if task not in SUPPORTED_TASKS:
                result = {"error": f"Task '{task}' is NOT supported. Only these 10 tasks are available: {SUPPORTED_TASKS}"}
            else:
                result = self.execute_task(task)
        elif tool_name == "get_scene_info":
            result = self.get_scene_info()
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        return json.dumps(result, ensure_ascii=False)

    def run_agent(self, user_request: str, max_iterations: int = 10) -> str:
        """Run the agent to complete a complex task."""
        print(f"\n{'='*50}")
        print(f"👤 User Request: {user_request}")
        print(f"{'='*50}")

        # System prompt for the agent
        system_prompt = f"""You are a robot manipulation agent. You interpret user requests and execute appropriate tasks.

Current environment: {self.suite_name}
Current scene: {self.env.task_description}

⚠️ ONLY THESE 10 TASKS ARE SUPPORTED - NO OTHER TASKS EXIST:
1. "open the middle drawer of the cabinet"
2. "put the bowl on the stove"
3. "put the wine bottle on top of the cabinet"
4. "open the top drawer and put the bowl inside"
5. "put the bowl on top of the cabinet"
6. "push the plate to the front of the stove"
7. "put the cream cheese in the bowl"
8. "turn on the stove"
9. "put the bowl on the plate"
10. "put the wine bottle on the rack"

⛔ FORBIDDEN - These tasks DO NOT EXIST and CANNOT be executed:
- "close drawer" (any drawer) - NOT SUPPORTED
- "pick up" anything - NOT SUPPORTED
- Any task not in the list above - NOT SUPPORTED

YOUR JOB:
- Interpret user's intent and map to available tasks above
- If user asks for something impossible (like closing drawer), explain it's not supported
- Execute tasks automatically using EXACT task strings from the list

Tools:
1. execute_task: Pass EXACT task string from the 10 supported tasks
2. get_scene_info: Verify task result using VLM

RULES:
1. Call ONE tool at a time. Wait for result before next action.
2. After execute_task, call get_scene_info to verify result.
3. If verification shows FAILURE, RETRY the same task ONCE more before reporting failure.
4. Execute tasks directly - do NOT ask user for confirmation.
5. If user says "继续" or "continue", continue the previous task."""

        # Add new user request to conversation history
        self.conversation_history.append({"role": "user", "content": user_request})

        for iteration in range(max_iterations):
            print(f"\n--- Agent Iteration {iteration + 1} ---")

            response = self.client.messages.create(
                model="MiniMax-M2.1",
                max_tokens=4096,
                temperature=1.0,
                system=system_prompt,
                messages=self.conversation_history,
                tools=AGENT_TOOLS
            )

            # Print API response id for debugging
            print(f"📡 API id: {response.id}")

            # Process response
            tool_use_blocks = []
            text_response = ""

            for block in response.content:
                if hasattr(block, 'type'):
                    if block.type == "thinking":
                        print(f"💭 Thinking: {block.thinking[:200]}...")
                    elif block.type == "text":
                        text_response = block.text
                        print(f"💬 Agent: {block.text}")
                    elif block.type == "tool_use":
                        tool_use_blocks.append(block)
                        print(f"🔧 Tool: {block.name}({json.dumps(block.input, ensure_ascii=False)})")

            # If no tool calls, we're done
            if not tool_use_blocks:
                # Save assistant response to history
                self.conversation_history.append({"role": "assistant", "content": response.content})
                return text_response

            # Process tool calls
            self.conversation_history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_block in tool_use_blocks:
                result = self.process_tool_call(tool_block.name, tool_block.input)
                print(f"📊 Result: {result[:200]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result
                })

            self.conversation_history.append({"role": "user", "content": tool_results})

        return "Agent reached maximum iterations without completing the task."

    def run_interactive(self):
        """Run the agent in interactive mode with viewer."""
        print("\n=== Agent Mode ===")
        print(f"Current task: {self.env.task_description}")
        print("\nExample commands for this scene:")

        # Scene-specific examples
        if "libero_object" in self.suite_name:
            print("  - Put all food items in the basket")
            print("  - Pick up the butter and milk, place them in the basket")
            print("  - Clean up the table")
        elif "libero_spatial" in self.suite_name:
            print("  - Move all bowls to the plate")
            print("  - Rearrange: put bowls on plate, then cups on tray")
        elif "libero_goal" in self.suite_name:
            print("  - Open all drawers one by one")
            print("  - Open drawer, then close it, then open again")

        print("\nCommands: 'quit' to exit, 'reset' to reset environment")
        print()

        # Launch viewer
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            self.viewer = viewer
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 1.8
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

            while viewer.is_running():
                try:
                    import sys
                    sys.stdout.write("\n🎯 Agent> ")
                    sys.stdout.flush()
                    user_input = sys.stdin.buffer.readline().decode('utf-8', errors='replace').strip()

                    if not user_input:
                        continue
                    if user_input.lower() == "quit":
                        break
                    if user_input.lower() == "reset":
                        self.reset_environment()
                        print("✅ Environment reset complete. Ready for new commands.")
                        continue

                    # Run agent
                    result = self.run_agent(user_input)
                    print(f"\n✨ Final Result: {result}")

                except (EOFError, KeyboardInterrupt):
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()

        self.env.close()
        print("Bye!")


def main():
    print("Select task suite:")
    print("1. libero_object  - 不同物体泛化")
    print("2. libero_spatial - 空间关系理解")
    print("3. libero_goal    - 不同动作目标")

    choice = input("Enter choice (1/2/3) [default=3]: ").strip() or "3"
    suite_map = {"1": "libero_object", "2": "libero_spatial", "3": "libero_goal"}
    suite_name = suite_map.get(choice, "libero_goal")

    agent = RobotAgent()
    agent.initialize(suite_name=suite_name)
    agent.run_interactive()


if __name__ == "__main__":
    main()
