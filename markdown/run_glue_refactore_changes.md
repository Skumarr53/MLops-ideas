## USER


## USER
Currently, I am fine-tuning a natural language inference (NLI) model using an existing Bash script. I would like to transition this process into Python code by directly calling the main function from run_glue.py. Additionally, I want to make some modifications to the training function so that it not only writes to the output path but also returns the trained model and evaluation metrics. This way, I can log both the model and metrics in MLflow. The training arguments should remain the same, specifically: train_file: str, validation_file: str, and param_dict: Dict.



``` Existing code 
# centralized_nlp_package/mlflow_utils/models.py
class DeBERTaModel(BaseModel):
    def load_model(self):
        logger.info(f"Loading DeBERTa model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)

    def train(self, train_file: str, validation_file: str, param_dict: Dict[str, Any]):
        logger.info("Starting training for DeBERTa model")
        subprocess.run([
            "python", "run_glue.py",
            "--model_name_or_path", self.model_path,
            "--output_dir", param_dict.get("output_dir", "./model_output"),
            "--train_file", train_file,
            "--validation_file", validation_file,
            "--do_train",
            "--do_eval",
            "--num_train_epochs", str(param_dict.get("n_epochs", 3)),
            "--fp16",
            "--report_to", "none",
            "--learning_rate", str(param_dict.get("learning_rate", 2e-5)),
            "--weight_decay", str(param_dict.get("weight_decay", 0.01)),
            "--per_device_train_batch_size", str(param_dict.get("train_batch_size", 16)),
            "--per_device_eval_batch_size", str(param_dict.get("eval_batch_size", 16))
        ], check=True)
        logger.info("Training completed for DeBERTa model")

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        logger.info("Evaluating DeBERTa model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        logger.info(f"Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}

```


core function from the run glue
```
# centralized_nlp_package/nli_utils/run_glue.py
def run_glue(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> None:
    
    """
    Executes the GLUE task by orchestrating data preparation, model training, evaluation, and prediction.

    Args:
        model_args (ModelArguments): Configuration related to the model.
        data_args (DataTrainingArguments): Configuration related to data processing.
        training_args (TrainingArguments): Configuration related to training.

    Usage Example:
        >>> from centralized_nlp_package.nli_utils import run_glue
        >>> from centralized_nlp_package.nli_utils import DataTrainingArguments
        >>> from centralized_nlp_package.nli_utils import ModelArguments
        >>> from transformers import TrainingArguments
        >>> 
        >>> model_args = ModelArguments(
        ...     model_name_or_path="bert-base-uncased",
        ...     cache_dir="./cache",
        ... )
        >>> data_args = DataTrainingArguments(
        ...     task_name="mnli",
        ...     train_file="path/to/train.csv",
        ...     validation_file="path/to/validation.csv",
        ... )
        >>> training_args = TrainingArguments(
        ...     output_dir="./output",
        ...     do_train=True,
        ...     do_eval=True,
        ...     num_train_epochs=3,
        ...     learning_rate=2e-5,
        ...     weight_decay=0.01,
        ...     per_device_train_batch_size=16,
        ...     per_device_eval_batch_size=16,
        ...     report_to="none",
        ... )
        >>> run_glue(model_args, data_args, training_args)
    """

    setup_logging(training_args)

    logger.info("Starting GLUE task...")
    logger.info(f"Training/evaluation parameters: {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use 'overwrite_output_dir' to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected at {last_checkpoint}. Resuming training from checkpoint."
            )

    set_seed(training_args.seed)

    raw_datasets = prepare_datasets(data_args, model_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenized_datasets = preprocess_datasets(raw_datasets, data_args, tokenizer)

    if data_args.task_name is not None:
        is_regression = data_args.task_name.lower() == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()
            num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    train_dataset = tokenized_datasets["train"] if training_args.do_train else None
    if training_args.do_train and data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = (
        tokenized_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if training_args.do_eval
        else None
    )
    if training_args.do_eval and data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    predict_dataset = (
        tokenized_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if training_args.do_predict
        else None
    )
    if training_args.do_predict and data_args.max_predict_samples is not None:
        predict_dataset = predict_dataset.select(range(min(len(predict_dataset), data_args.max_predict_samples)))

    if data_args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    elif training_args.fp16:
        data_collator = transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    compute_metrics = get_compute_metrics(is_regression, data_args.task_name)

    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train(trainer, data_args, model_args)

    if training_args.do_eval:
        evaluate(trainer, data_args, model_args, task_name=data_args.task_name)

    if training_args.do_predict:
        predict(trainer, data_args, model_args, task_name=data_args.task_name)

    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()
```

additional functionalities which run_glue dependent on 

```
# centralized_nlp_package/nli_utils/metrics.py

import evaluate
import numpy as np
from transformers import EvalPrediction
from typing import Optional, Callable

def get_compute_metrics(
    is_regression: bool,
    task_name: Optional[str] = None
) -> Callable[[EvalPrediction], dict]:
    """
    Returns a metrics computation function based on the task.

    Args:
        is_regression (bool): Whether the task is a regression task.
        task_name (Optional[str]): The name of the GLUE task.

    Returns:
        Callable[[EvalPrediction], dict]: A function that computes metrics.

    Usage Example:
        >>> from transformers import EvalPrediction
        >>> compute_metrics = get_compute_metrics(is_regression=False, task_name="mnli")
        >>> preds = np.array([[0.1, 0.9], [0.8, 0.2]])
        >>> labels = np.array([1, 0])
        >>> eval_pred = EvalPrediction(predictions=preds, label_ids=labels)
        >>> metrics = compute_metrics(eval_pred)
    """
    if task_name is not None:
        metric = evaluate.load("glue", task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def compute_metrics(p: EvalPrediction) -> dict:
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    return compute_metrics
```

```
# centralized_nlp_package/nli_utils/arguments.py

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataTrainingArguments:

    task_name: Optional[str] = field(
        default="mnli",
        metadata={"help": "The name of the task to train on: mnli, cola, etc. Default is 'mnli'."},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to 'max_seq_length'. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value for debugging or quicker training."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation examples to this value for debugging or quicker evaluation."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of prediction examples to this value for debugging or quicker prediction."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A CSV or a JSON file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in {
                "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
            }:
                raise ValueError(
                    "Unknown task, please select one from: cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli."
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "You must specify either a GLUE task, a training/validation file, or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "'train_file' should be a CSV or a JSON file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "'validation_file' should have the same extension as 'train_file'."


@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default="/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Default is '/dbfs/mnt/access_work/UC25/Libraries/HuggingFace/deberta-v3-large-zeroshot-v2'."},
    )
    cache_dir: Optional[str] = field(
        default="/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3",
        metadata={"help": "Directory to store the pretrained models downloaded from huggingface.co. Default is '/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-large-zeroshot-v2_v3'."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for training. Default is 2e-5."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for optimization. Default is 0.01."},
    )
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training. Default is 16."},
    )
    per_device_eval_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for evaluation. Default is 16."},
    )
```

```
# centralized_nlp_package/nli_utils/trainer.py

import logging
import os
from typing import Optional

import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, PreTrainedTokenizer
from transformers import default_data_collator, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import set_seed

from .metrics import get_compute_metrics
from .data import prepare_datasets, preprocess_datasets
from .arguments import DataTrainingArguments, ModelArguments

logger = logging.getLogger(__name__)

def setup_logging(training_args: TrainingArguments):
    """
    Configure logging for the training process.

    Args:
        training_args (TrainingArguments): Training-related arguments.

    Usage Example:
        >>> setup_logging(training_args)
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    transformers.utils.logging.set_verbosity_info() if training_args.should_log else transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

def initialize_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizer,
    data_collator,
    compute_metrics,
) -> Trainer:
    """
    Initialize the Hugging Face Trainer.

    Args:
        model (AutoModelForSequenceClassification): The model to train.
        training_args (TrainingArguments): Training-related arguments.
        train_dataset: The training dataset.
        eval_dataset: The evaluation dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        data_collator: Data collator for the trainer.
        compute_metrics: Function to compute metrics.

    Returns:
        Trainer: An initialized Trainer instance.

    Usage Example:
        >>> trainer = initialize_trainer(model, training_args, train_dataset, eval_dataset, tokenizer, data_collator, compute_metrics)
    """
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def train(
    trainer: Trainer,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> None:
    """
    Train the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.

    Usage Example:
        >>> train(trainer, data_args, model_args)
    """
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too

    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset) if data_args.max_train_samples is None else min(data_args.max_train_samples, len(trainer.train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

def evaluate(trainer: Trainer, data_args: DataTrainingArguments, model_args: ModelArguments, task_name: Optional[str] = None) -> None:
    """
    Evaluate the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
        task_name (Optional[str]): The name of the task for evaluation.

    Usage Example:
        >>> evaluate(trainer, data_args, model_args, task_name="mnli")
    """
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(trainer.eval_dataset) if data_args.max_eval_samples is None else min(data_args.max_eval_samples, len(trainer.eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def predict(trainer: Trainer, data_args: DataTrainingArguments, model_args: ModelArguments, task_name: Optional[str] = None) -> None:
    """
    Run predictions using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
        task_name (Optional[str]): The name of the task for prediction.

    Usage Example:
        >>> predict(trainer, data_args, model_args, task_name="mnli")
    """
    logger.info("*** Predict ***")
    predictions = trainer.predict(trainer.predict_dataset).predictions
    is_regression = task_name == "stsb" if task_name else False
    predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

    output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task_name}.txt" if task_name else "predict_results.txt")
    if is_main_process(training_args.local_rank):
        with open(output_predict_file, "w") as writer:
            logger.info(f"***** Predict results {task_name if task_name else ''} *****")
            writer.write("index\tprediction\n")
            for index, item in enumerate(predictions):
                if is_regression:
                    writer.write(f"{index}\t{item:.3f}\n")
                else:
                    label = trainer.model.config.id2label[item] if hasattr(trainer.model.config, "id2label") else item
                    writer.write(f"{index}\t{label}\n")

```


Please review the code thoroughly to grasp its functionality. Suggest how you plan to implement changes based on my requirements before moving forward with code generation. I will review your suggestions and indicate if any adjustments are needed.


## ASSISTANT
To transition your training process from a Bash script to Python and enable the `train` method to return the trained model along with evaluation metrics, we need to make several modifications. Here's a detailed plan outlining the necessary changes:

### **1. Refactor `run_glue.py` to Return Model and Metrics**

Currently, the `run_glue` function orchestrates the training and evaluation but does not return any outputs. To facilitate returning the trained model and evaluation metrics, we need to modify both `run_glue.py` and its dependent functions in `trainer.py`.

**a. Modify `run_glue.py`:**

- **Update the Function Signature:** Change `run_glue` to return a tuple containing the trained model and evaluation metrics.
  
- **Capture Metrics and Model:** After training and evaluation, collect the metrics and the trained model from the `Trainer` instance.

- **Return the Collected Outputs:** Ensure that `run_glue` returns both the model and metrics.

**Updated `run_glue.py`:**

```python
# centralized_nlp_package/nli_utils/run_glue.py

def run_glue(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments
) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
    
    # Existing setup and initialization code...

    trainer = initialize_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trained_model = None
    eval_metrics = {}

    if training_args.do_train:
        train_metrics = train(trainer, data_args, model_args)
        eval_metrics.update(train_metrics)

    if training_args.do_eval:
        evaluation = evaluate(trainer, data_args, model_args, task_name=data_args.task_name)
        eval_metrics.update(evaluation)

    if training_args.push_to_hub:
        trainer.push_to_hub()
    else:
        trainer.create_model_card()

    trained_model = trainer.model
    return trained_model, eval_metrics
```

**b. Update `trainer.py`:**

- **Modify `train` Function:** Ensure the `train` function returns the training metrics.

- **Modify `evaluate` Function:** Ensure the `evaluate` function returns the evaluation metrics.

**Updated `trainer.py`:**

```python
# centralized_nlp_package/nli_utils/trainer.py

def train(
    trainer: Trainer,
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
) -> Dict[str, float]:
    """
    Train the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.

    Returns:
        Dict[str, float]: Training metrics.
    """
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    else:
        checkpoint = get_last_checkpoint(training_args.output_dir)
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too

    metrics = train_result.metrics
    metrics["train_samples"] = len(trainer.train_dataset) if data_args.max_train_samples is None else min(data_args.max_train_samples, len(trainer.train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return metrics

def evaluate(
    trainer: Trainer, 
    data_args: DataTrainingArguments, 
    model_args: ModelArguments, 
    task_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate the model using the Trainer.

    Args:
        trainer (Trainer): The Trainer instance.
        data_args (DataTrainingArguments): Data-related arguments.
        model_args (ModelArguments): Model-related arguments.
        task_name (Optional[str]): The name of the task for evaluation.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(trainer.eval_dataset) if data_args.max_eval_samples is None else min(data_args.max_eval_samples, len(trainer.eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics
```

### **2. Update `DeBERTaModel` to Use the Refactored `run_glue`**

Now that `run_glue` returns the trained model and evaluation metrics, update the `DeBERTaModel` class to utilize these changes.

**a. Remove Subprocess Call:**

Instead of invoking `run_glue.py` via `subprocess.run`, directly import and call the `run_glue` function.

**b. Prepare Arguments:**

Construct `ModelArguments`, `DataTrainingArguments`, and `TrainingArguments` based on the input parameters (`train_file`, `validation_file`, and `param_dict`).

**c. Call `run_glue` and Capture Outputs:**

Invoke `run_glue` with the prepared arguments and capture the returned model and metrics.

**d. Return the Model and Metrics:**

Modify the `train` method to return the trained model and evaluation metrics alongside saving to the output directory.

**Updated `models.py`:**

```python
# centralized_nlp_package/mlflow_utils/models.py

from typing import Any, Dict
import logging
from transformers import pipeline, AutoModelForSequenceClassification
from centralized_nlp_package.nli_utils.run_glue import run_glue
from centralized_nlp_package.nli_utils.arguments import ModelArguments, DataTrainingArguments
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

class DeBERTaModel(BaseModel):
    def load_model(self):
        logger.info(f"Loading DeBERTa model from {self.model_path}")
        return pipeline("zero-shot-classification", model=self.model_path, device=self.device)
    
    def train(self, train_file: str, validation_file: str, param_dict: Dict[str, Any]) -> Tuple[AutoModelForSequenceClassification, Dict[str, float]]:
        logger.info("Starting training for DeBERTa model")

        # Prepare ModelArguments
        model_args = ModelArguments(
            model_name_or_path=self.model_path,
            cache_dir=param_dict.get("cache_dir", "./cache"),
            learning_rate=param_dict.get("learning_rate", 2e-5),
            weight_decay=param_dict.get("weight_decay", 0.01),
            per_device_train_batch_size=param_dict.get("train_batch_size", 16),
            per_device_eval_batch_size=param_dict.get("eval_batch_size", 16)
        )

        # Prepare DataTrainingArguments
        data_args = DataTrainingArguments(
            task_name=param_dict.get("task_name", "mnli"),
            train_file=train_file,
            validation_file=validation_file,
            max_seq_length=param_dict.get("max_seq_length", 128),
            pad_to_max_length=param_dict.get("pad_to_max_length", True),
            overwrite_cache=param_dict.get("overwrite_cache", False),
            max_train_samples=param_dict.get("max_train_samples", None),
            max_eval_samples=param_dict.get("max_eval_samples", None),
            max_predict_samples=param_dict.get("max_predict_samples", None)
        )

        # Prepare TrainingArguments
        training_args = TrainingArguments(
            output_dir=param_dict.get("output_dir", "./model_output"),
            do_train=True,
            do_eval=True,
            num_train_epochs=param_dict.get("n_epochs", 3),
            learning_rate=model_args.learning_rate,
            weight_decay=model_args.weight_decay,
            per_device_train_batch_size=model_args.per_device_train_batch_size,
            per_device_eval_batch_size=model_args.per_device_eval_batch_size,
            fp16=param_dict.get("fp16", True),
            report_to="none",
            overwrite_output_dir=param_dict.get("overwrite_output_dir", True),
            push_to_hub=param_dict.get("push_to_hub", False),
            seed=param_dict.get("seed", 42)
        )

        # Call run_glue
        trained_model, eval_metrics = run_glue(model_args, data_args, training_args)

        logger.info("Training completed for DeBERTa model")
        return trained_model, eval_metrics

    def evaluate(self, validation_file: str) -> Dict[str, float]:
        logger.info("Evaluating DeBERTa model")
        # Placeholder for actual evaluation logic
        accuracy = torch.rand(1).item()  # Replace with real evaluation
        logger.info(f"Evaluation accuracy: {accuracy}")
        return {"accuracy": accuracy}
```

### **3. Ensure Argument Consistency**

Make sure that the `ModelArguments`, `DataTrainingArguments`, and `TrainingArguments` classes are properly aligned with the parameters provided in `param_dict`. This ensures that all necessary configurations are correctly passed to the `run_glue` function.

**Example Adjustment in `arguments.py`:**

If there are additional parameters in `param_dict` that are not covered by the existing `ModelArguments` or `DataTrainingArguments`, consider extending these classes or handling them appropriately within the `train` method.

### **4. Integrate with MLflow**

With the `train` method now returning both the trained model and evaluation metrics, you can seamlessly integrate this with MLflow for logging.

**Example Integration:**

```python
import mlflow

# Initialize MLflow run
with mlflow.start_run():
    # Train the model
    trained_model, eval_metrics = deberta_model.train(train_file, validation_file, param_dict)
    
    # Log metrics
    for key, value in eval_metrics.items():
        mlflow.log_metric(key, value)
    
    # Log the model
    mlflow.transformers.log_model(trained_model, "model")
```

### **5. Additional Considerations**

- **Error Handling:** Ensure that appropriate error handling is in place, especially when dealing with file paths and model loading.

- **Logging:** Maintain consistent logging to facilitate debugging and monitoring during the training process.

- **Dependency Management:** Ensure that all necessary dependencies (e.g., `transformers`, `datasets`, `evaluate`, `mlflow`) are installed and properly managed within your environment.

- **Testing:** Before deploying the changes, perform thorough testing to verify that the training process works as expected and that the model and metrics are correctly returned and logged.

### **Summary**

By refactoring the `run_glue` function to return the trained model and evaluation metrics and updating the `DeBERTaModel` class to utilize these changes, you can effectively transition from a Bash-based training process to a more flexible and Pythonic approach. This setup not only enhances integration with MLflow but also provides better control and extensibility for your training pipeline.

Let me know if you need further adjustments or specific code implementations based on this plan!