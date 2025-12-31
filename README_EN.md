# Pi05 Robot Agent

A robot agent system based on Pi05 VLA model and MiniMax-M2.1 LLM for manipulation tasks in LIBERO simulation environment.

## Demo

https://github.com/user-attachments/assets/robot.mp4

## Architecture

```
User Command → MiniMax LLM (Task Planning) → Pi05 VLA (Action Execution) → LIBERO Sim
                    ↑                                                          ↓
              MCP Vision  ← ─────────────── Scene Image ←──────────────────────┘
```

- **Pi05 VLA**: Vision-Language-Action model for task execution
- **MiniMax-M2.1 LLM**: Task planning and user intent understanding
- **MiniMax MCP**: Visual understanding for task verification
- **LIBERO/MuJoCo**: Robot simulation environment

## Setup

### 1. API Keys

```bash
# MiniMax API Key (for LLM and MCP visual understanding)
# Get it from: https://platform.minimaxi.com/
export ANTHROPIC_API_KEY="your-minimax-api-key"

# HuggingFace Token (for downloading Pi05 model)
# Get it from: https://huggingface.co/settings/tokens
export HF_TOKEN="your-huggingface-token"
```

### 2. Download Pi05 Model

```bash
pip install huggingface_hub
huggingface-cli login

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/pi05_libero',
    local_dir='./models/pi05_libero_finetuned'
)
"
```

Default model path: `./models/pi05_libero_finetuned`

> To change the path, edit `MODEL_PATH` in `agent_mode.py`

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch numpy pillow einops draccus
pip install mujoco          # Physics engine
pip install libero robosuite # Robot simulation
pip install anthropic mcp   # LLM and MCP client
pip install transformers huggingface_hub  # Pi05 model deps
```

**Dependencies:**
| Package | Source | Description |
|---------|--------|-------------|
| LeRobot | Included in project | HuggingFace robotics library with Pi05 |
| MuJoCo | `pip install mujoco` | DeepMind physics engine |
| LIBERO | `pip install libero` | Robot manipulation benchmark |
| MCP | `pip install mcp` | Model Context Protocol client |

## Usage

### Agent Mode (Recommended)

```bash
# Set display (for VNC)
export DISPLAY=:2

# Run Agent
python agent_mode.py
```

Agent mode features:
- Natural language task planning
- Automatic task decomposition
- Visual verification of results
- Auto-retry on failure

### Simple Mode

```bash
python run_robot.py
```

Direct VLA task execution without LLM planning.

## Supported Tasks

LIBERO Goal scene supports these 10 tasks:

| # | Task | Description |
|---|------|-------------|
| 1 | `open the middle drawer of the cabinet` | Open cabinet middle drawer |
| 2 | `put the bowl on the stove` | Place bowl on stove |
| 3 | `put the wine bottle on top of the cabinet` | Place wine bottle on cabinet |
| 4 | `open the top drawer and put the bowl inside` | Open top drawer and put bowl inside |
| 5 | `put the bowl on top of the cabinet` | Place bowl on cabinet |
| 6 | `push the plate to the front of the stove` | Push plate to stove front |
| 7 | `put the cream cheese in the bowl` | Put cream cheese in bowl |
| 8 | `turn on the stove` | Turn on stove |
| 9 | `put the bowl on the plate` | Place bowl on plate |
| 10 | `put the wine bottle on the rack` | Place wine bottle on rack |

## Commands

| Command | Description |
|---------|-------------|
| Task description | Chinese or English |
| `reset` | Reset environment |
| `quit` | Exit program |

## Project Structure

```
MiniMax-Agent-VLA-Demo/
├── agent_mode.py              # Agent mode: LLM + VLA + MCP
├── run_robot.py               # Simple mode: VLA only
├── requirements.txt           # Python dependencies
├── README.md                  # Chinese documentation
├── README_EN.md               # English documentation
├── lerobot/                   # LeRobot core library (included)
│   ├── envs/                  # Environment wrappers (LIBERO, MetaWorld)
│   ├── policies/              # Policy models (Pi0, Pi05)
│   ├── configs/               # Configuration system
│   ├── processor/             # Data processing
│   └── utils/                 # Utility functions
└── models/                    # (download required)
    └── pi05_libero_finetuned/ # Pi05 model weights
```

## FAQ

**Q: API error "Invalid API Key"?**
A: Check if `ANTHROPIC_API_KEY` is correctly set to your MiniMax API Key

**Q: Model loading failed?**
A: 1) Check `HF_TOKEN` configuration 2) Verify `MODEL_PATH` is correct

**Q: MCP visual understanding error?**
A: Make sure `pip install mcp` is installed and `uvx` command is available

**Q: Visualization window not showing?**
A: Set `export DISPLAY=:2` (VNC) or ensure X11 environment is available

## Technical Details

- **Pi05 Model**: Vision-Language-Action model based on PaliGemma
- **Input**: 2 camera images + robot state + language instruction
- **Output**: 7-dim action (end-effector position delta + orientation delta + gripper)
- **Control frequency**: 10Hz
- **Max steps per task**: 280
