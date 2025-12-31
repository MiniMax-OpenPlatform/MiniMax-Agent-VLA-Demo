# Pi05 Robot Agent

基于 Pi05 VLA 模型和 MiniMax-M2.1 LLM 的机器人Agent系统，在 LIBERO 仿真环境中执行操作任务。

## Demo

![Demo](./demo.webp)

## 系统架构

```
用户指令 → MiniMax LLM (任务规划) → Pi05 VLA (动作执行) → LIBERO仿真
                ↑                                              ↓
          MCP视觉理解 ← ─────────── 场景图像 ←──────────────────┘
```

- **Pi05 VLA**: 视觉-语言-动作模型，执行具体操作任务
- **MiniMax-M2.1 LLM**: 任务规划、理解用户意图
- **MiniMax MCP**: 视觉理解，验证任务执行结果
- **LIBERO/MuJoCo**: 机器人仿真环境

## 环境配置

### 1. API Keys

```bash
# MiniMax API Key (用于LLM和视觉理解MCP)
# 获取: https://platform.minimaxi.com/
export ANTHROPIC_API_KEY="your-minimax-api-key"

# HuggingFace Token (用于下载Pi05模型)
# 获取: https://huggingface.co/settings/tokens
export HF_TOKEN="your-huggingface-token"
```

### 2. 下载Pi05模型

```bash
# 安装huggingface_hub
pip install huggingface_hub

# 登录HuggingFace
huggingface-cli login

# 下载Pi05 LIBERO微调模型
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='lerobot/pi05_libero',
    local_dir='./models/pi05_libero_finetuned'
)
"
```

模型默认路径: `./models/pi05_libero_finetuned`

> 如需修改路径，编辑 `agent_mode.py` 中的 `MODEL_PATH` 变量

### 3. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 一键安装所有依赖
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch numpy pillow einops draccus
pip install mujoco          # 物理仿真引擎
pip install libero robosuite # 机器人仿真环境
pip install anthropic mcp   # LLM和MCP客户端
pip install transformers huggingface_hub  # Pi05模型依赖
```

**依赖项说明:**
| 依赖 | 来源 | 说明 |
|------|------|------|
| LeRobot | 已包含在项目中 | HuggingFace机器人学习库，包含Pi05策略 |
| MuJoCo | `pip install mujoco` | DeepMind物理仿真引擎 |
| LIBERO | `pip install libero` | 机器人操作仿真基准环境 |
| MCP | `pip install mcp` | Model Context Protocol客户端 |

## 运行

### Agent模式 (推荐)

```bash
# 设置显示 (VNC环境)
export DISPLAY=:2

# 运行Agent
python agent_mode.py
```

Agent模式支持：
- 自然语言任务规划
- 自动任务分解
- 视觉验证执行结果
- 失败自动重试

### 简单模式

```bash
python run_robot.py
```

直接执行单个VLA任务，无LLM规划。

## 支持的任务

LIBERO Goal场景支持以下10个任务：

| # | 任务 | 描述 |
|---|------|------|
| 1 | `open the middle drawer of the cabinet` | 打开橱子中间抽屉 |
| 2 | `put the bowl on the stove` | 把碗放在炉子上 |
| 3 | `put the wine bottle on top of the cabinet` | 把红酒瓶放在橱子上 |
| 4 | `open the top drawer and put the bowl inside` | 打开顶部抽屉把碗放进去 |
| 5 | `put the bowl on top of the cabinet` | 把碗放在橱子上 |
| 6 | `push the plate to the front of the stove` | 把盘子推到炉子前面 |
| 7 | `put the cream cheese in the bowl` | 把奶油奶酪放进碗里 |
| 8 | `turn on the stove` | 打开炉子 |
| 9 | `put the bowl on the plate` | 把碗放在盘子上 |
| 10 | `put the wine bottle on the rack` | 把红酒瓶放在架子上 |

## 交互命令

| 命令 | 说明 |
|------|------|
| 任务描述 | 中文或英文均可 |
| `reset` | 重置环境到初始状态 |
| `quit` | 退出程序 |
| `继续` | 继续上一个任务 |

## 文件结构

```
MiniMax-Agent-VLA-Demo/
├── agent_mode.py              # Agent模式：LLM + VLA + MCP
├── run_robot.py               # 简单模式：仅VLA
├── requirements.txt           # Python依赖
├── README.md                  # 说明文档
├── lerobot/                   # LeRobot核心库 (已包含)
│   ├── envs/                  # 环境封装 (LIBERO, MetaWorld)
│   ├── policies/              # 策略模型 (Pi0, Pi05)
│   ├── configs/               # 配置系统
│   ├── processor/             # 数据处理
│   └── utils/                 # 工具函数
└── models/                    # (需下载)
    └── pi05_libero_finetuned/ # Pi05模型权重
```

## 常见问题

**Q: API调用报错 "Invalid API Key"?**
A: 检查 `ANTHROPIC_API_KEY` 是否正确设置为MiniMax的API Key

**Q: 模型加载失败?**
A: 1) 检查 `HF_TOKEN` 是否配置 2) 检查 `MODEL_PATH` 路径是否正确

**Q: MCP视觉理解报错?**
A: 确保安装: `pip install mcp` 并且 `uvx` 命令可用

**Q: 可视化窗口不显示?**
A: 设置 `export DISPLAY=:2` (VNC) 或确保有X11环境

## 技术细节

- **Pi05模型**: 基于PaliGemma的视觉-语言-动作模型
- **输入**: 2个相机图像 + 机械臂状态 + 语言指令
- **输出**: 7维动作 (末端位置增量 + 姿态增量 + 夹爪)
- **控制频率**: 10Hz
- **最大步数**: 280步/任务
