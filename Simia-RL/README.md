# Simia-RL: Reinforcement Learning for Agent Training

Reinforcement learning training module based on GRPO algorithm, using GPT as environment simulator for dialogue generation and evaluation.

## Core Features

- 🤖 GPT-driven simulated environment
- 🚀 GRPO (Group Relative Policy Optimization) training
- 💾 Detailed training trajectory recording

## Quick Start

```bash
# 1. Install dependencies
bash setup_local.sh

# 2. Configure environment variables
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.cognitiveservices.azure.com/"
export WANDB_API_KEY="your_wandb_api_key"

# 3. Start training (auto-generates config and starts training)
bash run_simulated_env_RL.sh
```

## Configuration

Modify configuration directly in run_simulated_env_RL.sh

```bash
```
# Model configuration
model_path: Simia-Agent/Simia-Tau-SFT-Qwen3-8B
micro_batch_size_per_gpu: 1
ppo_mini_batch_size: 8

# Training configuration
trainer:
  n_gpus_per_node: 8
  total_training_steps: 64
  save_freq: 4

# GRPO environment configuration
es_manager:
  train:
    env_groups: 16       # Number of training groups
    group_size: 8        # Environments per group (same group shares samples)

# Simulated environment configuration
custom_envs:
  GeneralSimulated:
    env_config:
      train_data_path: "./APIGen_5k_processed.json"
      deployment: "gpt-5"
      max_simulation_steps: 100
```
```


## Customizing Simulated Environment

Environment code location: `subtrees/ragen/ragen/env/simulated_General/`

### Modifying Interaction Prompts

Edit methods in `env.py`:

```python
# 1. Modify environment response generation (GPT simulates human/tool)
def _generate_next_environment_message(self, agent_message: str):
    prompt = f"""You are a simulation environment...
    [Customize your interaction logic]
    """

# 2. Modify evaluation logic (GPT evaluates agent performance)
def _evaluate_final_performance(self):
    prompt = f"""Evaluate the RL model's performance...
    Evaluation criteria:
    1. [Customize evaluation criteria]
    ...
    """
```


### Interaction Flow

```
reset() → Sample data → GPT generates first message 
  ↓
step(action) → GPT responds → Check if done
  ↓
done → GPT evaluates → Return reward (0/1)
```


## ⚠️ Validation and Testing

**Tau² currently does not support online validation, offline testing required:**

### Saving Checkpoints

```yaml
trainer:
  save_freq: 4              # Save every 4 steps
  val_before_train: False   # Disable validation before training
```

Checkpoints saved at: `outputs/checkpoints/Simia-RL-8B-tau2-APIGen-MT/step_X/`


## Outputs and Logs

### Directory Structure

```
outputs/
├── checkpoints/.../step_X/         # Model checkpoints
└── training_records/.../
    └── training_record/
        └── sample_X/               # Trajectories organized by sample
            └── env_Y_epZ_*.json    # Detailed interaction records
```

### Trajectory Records

Each file contains complete conversation history, GPT call records, and evaluation results:

```json
{
  "reward": 1.0,
  "reasoning": "Evaluation reasoning...",
  "conversation_history": [...],
  "gpt_call_history": [...]
}
```





