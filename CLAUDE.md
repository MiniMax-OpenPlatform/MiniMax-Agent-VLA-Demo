# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This workspace contains two main projects:

1. **LeRobot** (`lerobot/`) - Hugging Face's robotics library for real-world robotics in PyTorch
2. **MuJoCo** (`mujoco/`) - Google DeepMind's physics simulator (reference/dependency)

Most development work happens in the `lerobot/` directory.

## Common Commands

### LeRobot Installation
```bash
cd lerobot
pip install -e ".[dev,test]"  # Development install with test dependencies
```

### Running Tests
```bash
# Requires git-lfs for test artifacts
git lfs install && git lfs pull

# Run all tests
pytest -sv ./tests

# Run specific test file
pytest -sv tests/test_specific_feature.py
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run checks manually
pre-commit run --all-files
```

### Training & Evaluation
```bash
# Train a policy
lerobot-train --policy=act --dataset.repo_id=lerobot/aloha_mobile_cabinet

# Evaluate a policy
lerobot-eval --policy.path=<model_path> --env.type=<env>
```

### End-to-End Tests (via Makefile)
```bash
cd lerobot
make test-end-to-end DEVICE=cpu  # Full E2E test suite
make test-act-ete-train DEVICE=cpu  # Single policy test
```

## Architecture

### LeRobot Source Structure (`lerobot/src/lerobot/`)

- **policies/** - ML policies (ACT, Diffusion, TDMPC, VQ-BeT, SmolVLA, Pi0, GR00T)
  - Each policy has: `configuration_*.py`, `modeling_*.py`, `processor_*.py`
- **robots/** - Hardware interfaces for physical robots (SO100, Koch, LeKiwi, etc.)
- **teleoperators/** - Teleoperation devices (gamepads, keyboards)
- **cameras/** - Camera backends (OpenCV, RealSense)
- **motors/** - Motor controllers (Dynamixel, Feetech)
- **datasets/** - LeRobotDataset format (Parquet + MP4), data loaders, transforms
- **envs/** - Simulation environments (Aloha, PushT, LIBERO, MetaWorld)
- **configs/** - Draccus-based configuration system for training/eval
- **scripts/** - CLI entry points (`lerobot-train`, `lerobot-eval`, etc.)

### Key Patterns

- **Configuration**: Uses `draccus` for dataclass-based configs parsed from CLI args
- **Dataset Format**: LeRobotDataset uses Parquet for state/action + MP4 for video
- **Policy Interface**: Policies implement `select_action()` and `forward()` methods
- **Robot Interface**: Robots implement `connect()`, `get_observation()`, `send_action()`

### Optional Dependencies

LeRobot has many optional extras for specific hardware/policies:
```bash
pip install lerobot[smolvla]  # SmolVLA policy
pip install lerobot[aloha]    # Aloha simulation
pip install lerobot[feetech]  # Feetech motors
pip install lerobot[all]      # Everything (may have conflicts)
```

## Code Style

- Python 3.10+
- Ruff for linting (line length 110)
- Google-style docstrings
- Type hints encouraged (mypy enabled for some modules)
