# Tau2 Data Processing and Generation Tool

This repo is a toolkit for generating and processing Agent Airline and Retail training data, supporting data cleaning, format conversion, and conversation generation. Note: If you are evaluating our Qwen3 models on Tau2-Bench, please select the non-thinking mode.

## Quick Start



### 1. Configuration File

Unzip the pre-processed seed data
```python
cd Simia_SFT/Tau2
unzip APIGen_5k_preprocessed_zip.zip
```


Edit `config.json` to set generation parameters. Set `api_type` to either `"azure"` or `"openai"`. Fill in the relevant parameters for your chosen service and leave the other service's parameters blank.

**Example for OpenAI API:**
```json
{
  "api_type": "openai",
  "openai_api_key": "sk-your-api-key",
  "openai_base_url": "https://api.openai.com/v1",
  "openai_model": "gpt-5",
  "azure_endpoint": "",
  "api_version": "",
  "deployment": "",
  "sample_data_path": "APIGen_5k_preprocessed.json",
}
```

**Example for Azure OpenAI:**
```json
{
  "api_type": "azure",
  "openai_api_key": "",
  "openai_base_url": "",
  "openai_model": "",
  "azure_endpoint": "https://your-resource.openai.azure.com/",
  "api_version": "2025-04-01-preview",
  "deployment": "gpt-5",
  "sample_data_path": "APIGen_5k_preprocessed.json",
}
```

### 2. Conversation Generation

Use `main.py` to generate Agent multi-turn conversation data. Features include checkpoint resumption, log saving, and progress tracking by default.

```bash
# Use default configuration
python main.py

# Specify configuration file
python main.py --config config.json

# Force restart
python main.py --force-new

# Check progress
python main.py --status
```

**Customizing Prompts**: You can customize the conversation generation prompts by editing `Simia-Agent-Training/Simia_SFT/Tau2/utils/conversation_generator.py`


### 3. Post-processing Pipeline

Use `process_data_pipeline.sh` to batch process raw data:

```bash
# Basic usage
bash process_data_pipeline.sh <input_file>

# Specify output file
bash process_data_pipeline.sh <input_file> <output_file>
```

Processing workflow includes:
1. `fix_arguments.py` - Fix function argument formats
2. `tool2hermes.py` - Convert to Hermes format
3. `tool_correct.py` - Correct tool calls
4. `remove_think_tag.py` - Remove thinking tags
5. `replace_system_prompt_Hermes.py` - Replace system prompts


### 4. SFT Training on Processed Dataset

The processed dataset can be directly used for supervised fine-tuning with frameworks like [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory).

To use with LLaMA Factory, add an entry to `dataset_info.json`:

```json
"tau2_90k": {
    "file_name": your_file_path,
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "system": "system"
    }
}
```

Then create a YAML configuration file:
```yaml
### model
model_name_or_path: Qwen/Qwen2.5-7B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json
flash_attn: fa2
neat_packing: true

### dataset
dataset: yourdataset
template: qwen
overwrite_cache: true
preprocessing_num_workers: 32
cutoff_len: 12000

### output
output_dir: saves/Qwen2.5-7B-Instruct/yourdataset
logging_steps: 1
save_steps: 50
plot_loss: true
overwrite_output_dir: true
save_only_model: true

### train
lr_scheduler_type: cosine
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 0.000005
num_train_epochs: 2
bf16: true
ddp_timeout: 180000000
```


## Output Files

- Processed data: `*_processed.json`
- Generated conversations: `output/tau2_100k_gpt5.json`
- GPT logs: `gpt_log_tau2_100k_gpt5_*.jsonl`
- Progress files: `progress_*.json`

## Notes

- Ensure Azure OpenAI API key is configured
- Processing pipeline will automatically clean up intermediate files
