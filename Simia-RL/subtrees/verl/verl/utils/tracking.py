# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""
import os
import dataclasses
import tempfile
import pandas as pd
from abc import ABC, abstractmethod
from yaml import safe_dump
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Union, Optional


class Logger(ABC):
    """Abstract base class for all logger implementations"""
    
    @abstractmethod
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        """Log data at a specific step"""
        pass
    
    def log_generations(self, samples: List, step: int) -> None:
        """Log generation samples (optional override)"""
        pass
    
    def finish(self) -> None:
        """Clean up resources (optional override)"""
        pass


class WandbLogger(Logger):
    name = "wandb"
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any] = None):
        import wandb
        wandb.init(project=project_name, name=experiment_name, config=config)
        self.wandb = wandb
    
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        self.wandb.log(data=data, step=step)
    
    def log_generations(self, samples: List, step: int) -> None:
        """Log samples to wandb as a table"""
        import wandb

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], [])

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table
    
    def finish(self) -> None:
        self.wandb.finish(exit_code=0)


class SwanlabLogger(Logger):
    name = "swanlab"
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any] = None):
        import swanlab
        
        SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
        SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
        SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
        if SWANLAB_API_KEY:
            swanlab.login(SWANLAB_API_KEY)
        
        final_config = {"FRAMEWORK": "veRL"}
        if config:
            final_config.update(config)
            
        swanlab.init(
            project=project_name,
            experiment_name=experiment_name,
            config=final_config,
            logdir=SWANLAB_LOG_DIR,
            mode=SWANLAB_MODE,
        )
        self.swanlab = swanlab
    
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        self.swanlab.log(data=data, step=step)
    
    def log_generations(self, samples: List, step: int) -> None:
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_text_list = []
        for i, sample in enumerate(samples):
            row_text = f"""
            input: {sample[0]}
            
            ---
            
            output: {sample[1]}
            
            ---
            
            score: {sample[2]}
            """
            swanlab_text_list.append(swanlab.Text(row_text, caption=f"sample {i + 1}"))

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_text_list}, step=step)
    
    def finish(self) -> None:
        self.swanlab.finish()


class VemlpWandbLogger(Logger):
    name = "vemlp_wandb"
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any] = None):
        import os
        import volcengine_ml_platform
        from volcengine_ml_platform import wandb as vemlp_wandb
        
        volcengine_ml_platform.init(
            ak=os.environ["VOLC_ACCESS_KEY_ID"],
            sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
            region=os.environ["MLP_TRACKING_REGION"],
        )
        
        vemlp_wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            sync_tensorboard=True,
        )
        self.vemlp_wandb = vemlp_wandb
    
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        self.vemlp_wandb.log(data=data, step=step)
    
    def finish(self) -> None:
        self.vemlp_wandb.finish(exit_code=0)


class TensorboardLogger(Logger):
    name = "tensorboard"
    
    def __init__(self, project_name: str = None, experiment_name: str = None, config: Dict[str, Any] = None):
        import os
        from torch.utils.tensorboard import SummaryWriter
        
        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", "tensorboard_log")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)
    
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        for key in data:
            self.writer.add_scalar(key, data[key], step)
    
    def finish(self) -> None:
        self.writer.close()


class ConsoleLogger(Logger):
    name = "console"
    
    def __init__(self, project_name: str = None, experiment_name: str = None, config: Dict[str, Any] = None):
        from verl.utils.logger.aggregate_logger import LocalLogger
        self.console_logger = LocalLogger(print_to_console=True)
    
    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        self.console_logger.log(data=data, step=step)


class MlflowLogger(Logger):
    name = "mlflow"
    
    def __init__(self, project_name: str, experiment_name: str, config: Dict[str, Any] = None):
        import mlflow
        
        if config:
            try:
                mlflow.log_params(_compute_mlflow_params_from_objects(config))
            except mlflow.exceptions.RestException as e:
                print(f"WARNING: log params to mlflow failed with error {e}")
                print(f"WARNING: config: {config}")

            with tempfile.TemporaryDirectory() as tmp_dir:
                config_file = Path(tmp_dir, "config.yaml")
                with open(config_file, "w") as file:
                    safe_dump(config, file)
                mlflow.log_artifact(str(config_file))

    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        import mlflow
        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)
    
    def log_generations(self, samples: List, step: int) -> None:
        """Log validation generation to mlflow as artifacts"""
        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                base_name = f"val_step_{step:07d}"
                data = pd.DataFrame(samples, columns=["input", "output", "score"])

                # Save as JSONL file
                jsonl_file = Path(tmp_dir, f"{base_name}.jsonl")
                data.to_json(jsonl_file, orient="records", lines=True)
                mlflow.log_artifact(str(jsonl_file), artifact_path="generations")

                # Save as TSV file
                tsv_file = Path(tmp_dir, f"{base_name}.tsv")
                data.to_csv(tsv_file, sep="\t", index=False)
                mlflow.log_artifact(str(tsv_file), artifact_path="generations")
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")


class Tracking:
    def __init__(self, project_name: str, experiment_name: str, 
                 default_backend: Union[str, List[str]] = "console", 
                 config: Optional[Dict[str, Any]] = None):

        if isinstance(default_backend, str):
            default_backend = [default_backend]
        
        self.loggers: List[Logger] = []

        # Create a mapping of logger names to classes
        logger_map = {getattr(cls, "name", cls.__name__.lower()): cls 
                     for cls in Logger.__subclasses__()}

        for backend in default_backend:
            try:
                logger_cls = logger_map[backend]
                self.loggers.append(
                    logger_cls(project_name=project_name, experiment_name=experiment_name, config=config)
                )
            except KeyError:
                raise ValueError(f"Logger {backend} not found. Supported loggers are: {list(logger_map.keys())}")

    def log_metrics(self, data: Dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log_metrics(data=data, step=step)

    def log(self, data: Dict[str, Any], step: int) -> None:
        # deprecated
        return self.log_metrics(data=data, step=step)


    def log_generations(self, samples: List, step: int) -> None:
        """Log generation samples to all or specified loggers"""
        for logger in self.loggers:
            logger.log_generations(samples=samples, step=step)

    def finish(self) -> None:
        for logger in self.loggers:
            logger.finish()

    def __del__(self):
        self.finish()


def _compute_mlflow_params_from_objects(params) -> Dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: Dict[str, Any], *, sep: str) -> Dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans
