It sounds like you're encountering a common issue related to how MLflow handles different model flavors. Specifically, you're logging your model using the PyTorch flavor (`mlflow.pytorch.log_model`) but attempting to load it using the generic PyFunc interface (`mlflow.pyfunc.load_model`). This mismatch is likely causing the loaded model to be recognized as a generic function rather than your intended `AutoModelForSequenceClassification` instance.

Let's break down the problem and provide a comprehensive solution.

---

### **Understanding the Issue**

1. **Logging the Model:**
   - **Flavor Used:** `mlflow.pytorch.log_model`
   - **Result:** The model is saved with the PyTorch flavor, which includes specific artifacts and dependencies related to PyTorch.

2. **Loading the Model:**
   - **Flavor Used:** `mlflow.pyfunc.load_model`
   - **Result:** MLflow attempts to load the model using the PyFunc interface, which expects a generic model wrapped in a `python_function` environment. Since the model was saved with the PyTorch flavor, MLflow doesn't recognize how to handle it as a PyFunc, leading to unexpected behavior (like getting a function instead of the actual model).

---

### **Solution Steps**

1. **Consistent Model Logging and Loading:**
   - **Approach 1:** Use the same flavor for both logging and loading.
   - **Approach 2:** If you need to use PyFunc for some reason, define a custom PyFunc wrapper around your PyTorch model.

2. **Recommended Approach:** Use the PyTorch flavor consistently for both logging and loading.

---

### **Detailed Implementation**

#### **1. Logging the Model with PyTorch Flavor**

Your current logging code is mostly correct. However, ensure that the `self.output_dir` correctly points to the directory where the model is saved after fine-tuning.

```python
import mlflow.pytorch

# After fine-tuning
finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)

try:
    mlflow.pytorch.log_model(finetuned_model, "model")
    logger.info("Model logged successfully with PyTorch flavor")
except Exception as e:
    logger.error(f"Failed to log model: {e}")
```

#### **2. Registering the Model**

Ensure that during registration, you're pointing to the correct MLflow run and artifact path.

```python
from centralized_nlp_package.model_utils import ModelTransition

# Assuming `models_by_basemodel_version` is a list of MLflow Run objects
model_trans_obj = ModelTransition(model_name)

for run in models_by_basemodel_version:
    model_uri = f"runs:/{run.info.run_id}/model"
    model_trans_obj._register_model(model_uri)
print("Models registered successfully")
```

Ensure that the `_register_model` method in your `ModelTransition` class correctly registers the model URI.

#### **3. Loading the Model with PyTorch Flavor**

Instead of using `mlflow.pyfunc.load_model`, use `mlflow.pytorch.load_model` to load the model. This ensures that the model is loaded with the correct PyTorch-specific class.

```python
import mlflow.pytorch

# Load the production model using the PyTorch flavor
production_model = mlflow.pytorch.load_model(model_uri="models:/YourModelName/Production")

# Now, `production_model` is an instance of `AutoModelForSequenceClassification`
# You can use it directly for inference
```

**Example:**

```python
# Replace 'YourModelName' with the actual registered model name
production_model = mlflow.pytorch.load_model(model_uri="models:/YourModelName/Production")

# Example inference
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-base-model-name")
inputs = tokenizer("Example input text", return_tensors="pt")
outputs = production_model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
print(predictions)
```

#### **4. (Optional) Using PyFunc with a Custom Wrapper**

If you specifically need to use the PyFunc interface (e.g., for broader compatibility), you can create a custom PyFunc wrapper around your PyTorch model.

**a. Define a Custom PyFunc Model:**

```python
import mlflow.pyfunc
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class PyFuncNLIModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained("your-base-model-name")

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        return predictions.numpy()
```

**b. Log the Custom PyFunc Model:**

```python
import mlflow.pyfunc

# Define the artifacts
artifacts = {
    "model_path": "path/to/your/fine-tuned/model"  # This should be the same as `self.output_dir`
}

# Log the PyFunc model
mlflow.pyfunc.log_model(
    artifact_path="pyfunc_model",
    python_model=PyFuncNLIModel(),
    artifacts=artifacts,
    conda_env={
        'name': 'nli_env',
        'channels': ['defaults'],
        'dependencies': [
            'python=3.8',
            'pip',
            {
                'pip': [
                    'torch',
                    'transformers',
                    'mlflow',
                    # Add other dependencies as needed
                ],
            },
        ],
    }
)
```

**c. Load the Custom PyFunc Model:**

```python
import mlflow.pyfunc

# Load the PyFunc model
production_pyfunc_model = mlflow.pyfunc.load_model(model_uri="models:/YourModelName/Production")

# Example inference
predictions = production_pyfunc_model.predict(["Example input text"])
print(predictions)
```

**Note:** This approach requires maintaining a custom wrapper and ensuring that all necessary dependencies are included in the `conda_env`.

---

### **Summary**

- **Mismatch in Flavors:** You're logging with `mlflow.pytorch` but loading with `mlflow.pyfunc`, leading to type inconsistencies.
  
- **Solution:** Use `mlflow.pytorch.load_model` to load models logged with the PyTorch flavor.

- **Alternative (Optional):** If you need to use PyFunc, create a custom PyFunc wrapper around your PyTorch model.

By ensuring consistency in how you log and load your models, you should be able to avoid the issue of loading a model as a generic function and instead retain its intended class and functionality.

---

### **Additional Recommendations**

1. **Verify Model Registration:**
   - After logging and registering the model, navigate to the MLflow Model Registry UI to ensure that the models are correctly registered with the appropriate flavors.

2. **Consistency Across Environments:**
   - Ensure that the environments (dependencies) used during logging and loading are consistent to avoid any runtime issues.

3. **Testing:**
   - Before deploying to production, perform tests by logging a simple model and loading it to ensure the entire pipeline works as expected.

4. **Logging Model Configuration:**
   - Consider logging the tokenizer and any other necessary components as part of the model artifacts to simplify the loading and inference process.

---

Feel free to reach out if you encounter further issues or need more detailed assistance!

You're encountering a `PicklingError` because MLflow's PyTorch logger (`mlflow.pytorch.log_model`) attempts to serialize the model using Python's pickle mechanism. However, when using the `Accelerator` from the `accelerate` library for automatic mixed precision (AMP) or distributed training, the model is wrapped in a way that makes it non-pickleable directly. This is why you receive the error:

```
PicklingError: Can't pickle model with automatic mixed precision: it's not the same use accelerator.unwrap_model
```

To resolve this issue, you need to **unwrap the model** from any `Accelerator` wrappers before logging it with MLflow. Below are detailed steps and code adjustments to fix the problem.

---

### **Step-by-Step Solution**

1. **Unwrap the Model from Accelerator:**
   
   Use `accelerator.unwrap_model` to retrieve the underlying PyTorch model without any AMP or distributed training wrappers.

2. **Move the Model to CPU and Set to Evaluation Mode:**
   
   To ensure compatibility and avoid any device-related issues during logging, move the model to CPU and set it to evaluation mode.

3. **Log the Unwrapped Model with MLflow:**
   
   Use `mlflow.pytorch.log_model` to log the unwrapped model.

4. **(Optional) Save the Unwrapped Model:**
   
   If you need to save the model to a directory before logging, ensure it's saved without any wrappers.

---

### **Updated Code Implementation**

Below is the revised version of your `ExperimentManager` class's `run_experiments` method with the necessary changes:

```python
class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model_versions: str,
        train_file: str,
        validation_file: str,
        evalute_pretrained_model: bool = True,
        user_id: str = 'santhosh.kumar3@voya.com',
        output_dir: str = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/",
        **kwargs
    ):

        self.run_date = get_current_date_str()
        self.experiment_name = f"/Users/{user_id}/{base_name}_{data_src}_{self.run_date}"
        self.dataset_versions = dataset_versions
        self.hyperparameters = hyperparameters
        self.base_model_versions = base_model_versions
        self.output_dir = output_dir 
        self.validation_file = validation_file
        self.train_file = train_file
        self.evalute_pretrained_model = evalute_pretrained_model
        self.accelerator = Accelerator()
         
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        for base_model in self.base_model_versions:

            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_param_set{idx+1}"
                    with mlflow.start_run(run_name=run_name) as run:
                        logger.info(f"Starting finetuning run: {run_name}")
                        mlflow.set_tag("run_date", self.run_date)
                        mlflow.set_tag("base_model_name", base_model_name)
                        mlflow.set_tag("dataset_version", dataset_name)
                        mlflow.set_tag("run_type", "finetuned")

                        # Log hyperparameters
                        mlflow.log_params({
                            "num_train_epochs": param_set.get("n_epochs", 3),
                            "learning_rate": param_set.get("learning_rate", 2e-5),
                            "weight_decay": param_set.get("weight_decay", 0.01),
                            "per_device_train_batch_size": param_set.get("train_batch_size", 16)
                        })
                        
                        # Initialize model
                        model = get_model(
                            model_path=base_model,
                            device=0 if torch.cuda.is_available() else -1
                        )
                        
                        # Train model
                        train_file_path = self.train_file.format(data_version=dataset_version)
                        ft_model, eval_metrics = model.train(
                            train_file=train_file_path,
                            validation_file=self.validation_file,
                            param_dict=param_set,
                            output_dir = self.output_dir
                        )
                        
                        # Unwrap the model to remove Accelerator wrappers
                        unwrapped_model = self.accelerator.unwrap_model(ft_model)
                        
                        # Move the model to CPU and set to evaluation mode
                        unwrapped_model.to('cpu')
                        unwrapped_model.eval()
                        
                        try:
                            # Log the unwrapped model with MLflow
                            mlflow.pytorch.log_model(unwrapped_model, "model")
                            logger.info(f"Model logged successfully")
                        except Exception as e:
                            logger.error(f"Failed to log model: {e}")

                        # Example metrics dictionary
                        metrics = {
                            "accuracy": eval_metrics['eval_accuracy'],
                            "f1_score": eval_metrics['eval_f1_score'],
                            "precision": eval_metrics['eval_precision'],
                            "recall": eval_metrics['eval_recall'],
                            "roc_auc": eval_metrics['eval_roc_auc']
                        }

                        # Log multiple metrics at once
                        mlflow.log_metrics(metrics)
                        
                        logger.info(f"Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")
                        
                        torch.cuda.empty_cache()
                        gc.collect()
```

### **Key Changes Explained**

1. **Unwrapping the Model:**

   ```python
   unwrapped_model = self.accelerator.unwrap_model(ft_model)
   ```

   - This removes any AMP or distributed training wrappers, returning the pure PyTorch model instance.

2. **Moving to CPU and Setting Evaluation Mode:**

   ```python
   unwrapped_model.to('cpu')
   unwrapped_model.eval()
   ```

   - Ensures that the model is on CPU, which is often necessary for serialization.
   - Sets the model to evaluation mode, which is a good practice before logging/saving.

3. **Logging the Unwrapped Model:**

   ```python
   mlflow.pytorch.log_model(unwrapped_model, "model")
   ```

   - Logs the pure PyTorch model without any AMP-related wrappers, preventing `PicklingError`.

### **Additional Recommendations**

1. **Avoid Reloading the Model for Logging:**
   
   You initially attempted to load the model again using `AutoModelForSequenceClassification.from_pretrained(self.output_dir)`. This step is unnecessary if you already have the trained model (`ft_model`). Logging the unwrapped `ft_model` directly is more efficient and avoids potential issues related to reloading.

2. **Ensure Proper Saving (If Needed):**
   
   If you need to save the model to a directory before logging, ensure you do so **after** unwrapping:

   ```python
   unwrapped_model.save_pretrained(self.output_dir)
   ```

   Then, if you ever need to load it again, it will be free from AMP wrappers.

3. **Consistent Model Loading:**
   
   When loading the model for inference, use the same flavor (`mlflow.pytorch.load_model`) to maintain consistency and avoid flavor mismatches.

   ```python
   import mlflow.pytorch

   # Load the production model using the PyTorch flavor
   production_model = mlflow.pytorch.load_model(model_uri="models:/YourModelName/Production")

   # Example inference
   from transformers import AutoTokenizer
   
   tokenizer = AutoTokenizer.from_pretrained("your-base-model-name")
   inputs = tokenizer("Example input text", return_tensors="pt")
   outputs = production_model(**inputs)
   predictions = torch.argmax(outputs.logits, dim=1)
   print(predictions)
   ```

4. **Avoid Mixing Model Flavors:**
   
   Stick to using `mlflow.pytorch` for both logging and loading to prevent issues related to model flavors. Mixing `mlflow.pytorch` with `mlflow.pyfunc` can lead to the kind of issues you initially faced.

5. **Handle Dependencies Properly:**
   
   Ensure that all necessary dependencies (like `transformers`, `torch`, etc.) are specified in your MLflow environment, especially if you plan to deploy the model elsewhere.

6. **Testing:**
   
   After implementing these changes, perform a small-scale test to ensure that the model logs and loads correctly without errors.

### **Final Thoughts**

By unwrapping the model before logging, you eliminate the AMP-related wrappers that interfere with MLflow's serialization process. This ensures that your model is logged correctly and can be reliably loaded for inference without encountering type mismatches or pickling errors.

If you continue to face issues after these changes, consider the following:

- **Check MLflow Version:**
  
  Ensure you're using a compatible version of MLflow that supports the versions of PyTorch and `accelerate` you're using.

- **Review Accelerator Configuration:**
  
  Verify that the `Accelerator` is configured correctly and that no unintended wrappers remain around the model after unwrapping.

Feel free to reach out if you encounter further issues or need additional assistance!

The memory exception you're encountering is likely due to residual references to model objects and other large data structures persisting in memory across iterations of your experiment loop. Even though you're calling `torch.cuda.empty_cache()` and `gc.collect()`, certain references might still prevent Python's garbage collector from reclaiming memory effectively. This issue is common in scenarios involving large models and iterative processes like hyperparameter tuning.

To resolve this, you need to ensure that all references to large objects (like models, tokenizers, data loaders, etc.) are properly removed after each iteration. Additionally, encapsulating each experiment run within its own function can help limit the scope of variables, making it easier for Python to garbage collect them. Below are detailed steps and code modifications to address the memory leak issue.

---

## **1. Encapsulate Each Experiment in a Separate Function**

By moving the experiment logic into a separate function, you limit the scope of variables. Once the function execution completes, local variables are out of scope and eligible for garbage collection.

### **Implementation:**

```python
class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model_versions: str,
        train_file: str,
        validation_file: str,
        evalute_pretrained_model: bool = True,
        user_id: str = 'santhosh.kumar3@voya.com',
        output_dir: str = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/",
        **kwargs
    ):

        self.run_date = get_current_date_str()
        self.experiment_name = f"/Users/{user_id}/{base_name}_{data_src}_{self.run_date}"
        self.dataset_versions = dataset_versions
        self.hyperparameters = hyperparameters
        self.base_model_versions = base_model_versions
        self.output_dir = output_dir 
        self.validation_file = validation_file
        self.train_file = train_file
        self.evalute_pretrained_model = evalute_pretrained_model
        self.accelerator = Accelerator()
         
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        for base_model in self.base_model_versions:
            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_param_set{idx+1}"
                    with mlflow.start_run(run_name=run_name) as run:
                        self.run_single_experiment(
                            run_name, 
                            base_model, 
                            base_model_name, 
                            dataset_version, 
                            dataset_name, 
                            param_set
                        )

    def run_single_experiment(self, run_name, base_model, base_model_name, dataset_version, dataset_name, param_set):
        try:
            logger.info(f"Starting finetuning run: {run_name}")
            mlflow.set_tag("run_date", self.run_date)
            mlflow.set_tag("base_model_name", base_model_name)
            mlflow.set_tag("dataset_version", dataset_name)
            mlflow.set_tag("run_type", "finetuned")

            # Log hyperparameters
            mlflow.log_params({
                "num_train_epochs": param_set.get("n_epochs", 3),
                "learning_rate": param_set.get("learning_rate", 2e-5),
                "weight_decay": param_set.get("weight_decay", 0.01),
                "per_device_train_batch_size": param_set.get("train_batch_size", 16)
            })

            # Initialize model
            model = get_model(
                model_path=base_model,
                device=0 if torch.cuda.is_available() else -1
            )

            # Train model
            train_file_path = self.train_file.format(data_version=dataset_version)
            ft_model, eval_metrics = model.train(
                train_file=train_file_path,
                validation_file=self.validation_file,
                param_dict=param_set,
                output_dir=self.output_dir
            )

            # Unwrap the model to remove Accelerator wrappers
            unwrapped_model = self.accelerator.unwrap_model(ft_model)

            # Move the model to CPU and set to evaluation mode
            unwrapped_model.to('cpu')
            unwrapped_model.eval()

            # Log the unwrapped model with MLflow
            mlflow.pytorch.log_model(unwrapped_model, "model")
            logger.info("Model logged successfully")

            # Log metrics
            metrics = {
                "accuracy": eval_metrics['eval_accuracy'],
                "f1_score": eval_metrics['eval_f1_score'],
                "precision": eval_metrics['eval_precision'],
                "recall": eval_metrics['eval_recall'],
                "roc_auc": eval_metrics['eval_roc_auc']
            }
            mlflow.log_metrics(metrics)
            logger.info(f"Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")

        except Exception as e:
            logger.error(f"Failed during run {run_name}: {e}")

        finally:
            # Cleanup to free memory
            del ft_model
            del unwrapped_model
            del model
            torch.cuda.empty_cache()
            gc.collect()
```

### **Explanation:**

- **`run_single_experiment` Function:**
  - Encapsulates all operations related to a single experiment run.
  - Utilizes a `try-except-finally` block to ensure that cleanup occurs even if an error is raised during the experiment.
  
- **Cleanup in `finally` Block:**
  - **`del ft_model`, `del unwrapped_model`, `del model`:** Explicitly deletes references to model objects.
  - **`torch.cuda.empty_cache()`:** Frees up unused memory from the CUDA cache.
  - **`gc.collect()`:** Forces Python's garbage collector to reclaim memory.

- **Benefits:**
  - Limits the scope of variables, making it easier for Python to garbage collect them after each function call.
  - Ensures that even if an exception occurs, the cleanup code is executed.

---

## **2. Explicitly Delete Model and Related Objects**

Even with the encapsulation, it's essential to ensure that all large objects are explicitly deleted after their use. This includes models, tokenizers, datasets, and any other large data structures.

### **Implementation:**

As shown in the `run_single_experiment` function above, use `del` to remove references to objects:

```python
del ft_model
del unwrapped_model
del model
```

If you have other large objects (e.g., tokenizers), delete them similarly:

```python
del tokenizer
```

---

## **3. Use Context Managers for Resource Management**

Utilizing context managers (`with` statements) for handling resources can help manage memory more effectively. For example, if you're using datasets or data loaders that support context management, ensure they are properly closed after use.

### **Example:**

```python
from torch.utils.data import DataLoader

def run_single_experiment(...):
    with DataLoader(...) as data_loader:
        # Training code
        pass
    # DataLoader is automatically closed here
```

---

## **4. Avoid Unnecessary Model Reloads**

In your original code, you load the model again using `AutoModelForSequenceClassification.from_pretrained(self.output_dir)`. This step is unnecessary if you already have the trained model (`ft_model`). Logging the unwrapped `ft_model` directly is more efficient and avoids potential issues related to reloading.

### **Modification:**

Remove the following lines:

```python
finetuned_model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
```

And directly log the `unwrapped_model`:

```python
mlflow.pytorch.log_model(unwrapped_model, "model")
```

---

## **5. Monitor Memory Usage**

To better understand where the memory is being consumed, you can add memory profiling within your loop. This can help identify if certain operations are leaking memory.

### **Implementation:**

```python
import torch
import gc
import psutil

def log_memory_usage(stage):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # in GB
    logger.info(f"Memory usage after {stage}: {mem:.2f} GB")

def run_single_experiment(...):
    try:
        # Before training
        log_memory_usage("before training")

        # Training code
        ...

        # After logging model
        log_memory_usage("after logging model")

    finally:
        # Cleanup
        ...
        log_memory_usage("after cleanup")
```

### **Explanation:**

- **`log_memory_usage` Function:**
  - Logs the current memory usage at different stages of the experiment.
  
- **Usage:**
  - Call this function before and after key operations to monitor memory consumption.

---

## **6. Optimize Model and Data Handling**

While the above steps should significantly mitigate memory issues, further optimizations can help reduce memory footprint:

- **Use `torch.no_grad()` Where Appropriate:**
  
  If certain operations don't require gradient computations (e.g., evaluation), wrap them in `with torch.no_grad()` to save memory.

  ```python
  with torch.no_grad():
      outputs = model(**inputs)
  ```

- **Delete Unused Variables Promptly:**
  
  Ensure that temporary variables are deleted or go out of scope as soon as they're no longer needed.

- **Reduce Batch Sizes:**
  
  If memory usage remains high, consider reducing batch sizes during training or inference.

- **Use Mixed Precision Training:**
  
  Training with mixed precision (`fp16`) can reduce memory consumption. However, ensure that your hardware supports it.

---

## **7. Verify MLflow Logging Doesn't Retain References**

MLflow's logging mechanisms might inadvertently hold onto references of the models, especially if callbacks or custom logging functions are used. Ensure that:

- **No Additional References Are Held:**
  
  Check if MLflow callbacks or custom functions are keeping references to the models. Avoid storing models in global variables or logs.

- **Log Models Properly:**
  
  Use `mlflow.pytorch.log_model` correctly to avoid unintended side effects.

---

## **8. Example of the Complete Revised `ExperimentManager` Class**

Here's the complete revised class incorporating all the above suggestions:

```python
import mlflow
import mlflow.pytorch
import torch
import gc
from accelerate import Accelerator
from typing import List, Dict, Any
import logging
import psutil

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def log_memory_usage(stage):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # in GB
    logger.info(f"Memory usage after {stage}: {mem:.2f} GB")

class ExperimentManager:
    def __init__(
        self,
        base_name: str,
        data_src: str,
        dataset_versions: List[str],
        hyperparameters: List[Dict[str, Any]],
        base_model_versions: str,
        train_file: str,
        validation_file: str,
        evalute_pretrained_model: bool = True,
        user_id: str = 'santhosh.kumar3@voya.com',
        output_dir: str = "/dbfs/mnt/access_work/UC25/Topic Modeling/NLI Models/Fine-tune NLI models/trained_RD_deberta-v3-base-zeroshot-v2_Santhosh_test/",
        **kwargs
    ):

        self.run_date = get_current_date_str()
        self.experiment_name = f"/Users/{user_id}/{base_name}_{data_src}_{self.run_date}"
        self.dataset_versions = dataset_versions
        self.hyperparameters = hyperparameters
        self.base_model_versions = base_model_versions
        self.output_dir = output_dir 
        self.validation_file = validation_file
        self.train_file = train_file
        self.evalute_pretrained_model = evalute_pretrained_model
        self.accelerator = Accelerator()
         
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Experiment set to {self.experiment_name}")

    def run_experiments(self):
        for base_model in self.base_model_versions:
            base_model_name = base_model.split('/')[-1]

            if self.evalute_pretrained_model:
                self.evaluate_pretrained_model(base_model)

            for dataset_version in self.dataset_versions:
                for idx, param_set in enumerate(self.hyperparameters):
                    dataset_name = dataset_version.split('.')[0]

                    run_name = f"{base_model_name}_{dataset_name}_param_set{idx+1}"
                    with mlflow.start_run(run_name=run_name) as run:
                        self.run_single_experiment(
                            run_name, 
                            base_model, 
                            base_model_name, 
                            dataset_version, 
                            dataset_name, 
                            param_set
                        )

    def run_single_experiment(self, run_name, base_model, base_model_name, dataset_version, dataset_name, param_set):
        try:
            logger.info(f"Starting finetuning run: {run_name}")
            mlflow.set_tag("run_date", self.run_date)
            mlflow.set_tag("base_model_name", base_model_name)
            mlflow.set_tag("dataset_version", dataset_name)
            mlflow.set_tag("run_type", "finetuned")

            # Log hyperparameters
            mlflow.log_params({
                "num_train_epochs": param_set.get("n_epochs", 3),
                "learning_rate": param_set.get("learning_rate", 2e-5),
                "weight_decay": param_set.get("weight_decay", 0.01),
                "per_device_train_batch_size": param_set.get("train_batch_size", 16)
            })

            log_memory_usage("before training")

            # Initialize model
            model = get_model(
                model_path=base_model,
                device=0 if torch.cuda.is_available() else -1
            )

            # Train model
            train_file_path = self.train_file.format(data_version=dataset_version)
            ft_model, eval_metrics = model.train(
                train_file=train_file_path,
                validation_file=self.validation_file,
                param_dict=param_set,
                output_dir=self.output_dir
            )

            log_memory_usage("after training")

            # Unwrap the model to remove Accelerator wrappers
            unwrapped_model = self.accelerator.unwrap_model(ft_model)

            # Move the model to CPU and set to evaluation mode
            unwrapped_model.to('cpu')
            unwrapped_model.eval()

            log_memory_usage("after unwrapping and moving to CPU")

            # Log the unwrapped model with MLflow
            mlflow.pytorch.log_model(unwrapped_model, "model")
            logger.info("Model logged successfully")

            log_memory_usage("after logging model")

            # Log metrics
            metrics = {
                "accuracy": eval_metrics['eval_accuracy'],
                "f1_score": eval_metrics['eval_f1_score'],
                "precision": eval_metrics['eval_precision'],
                "recall": eval_metrics['eval_recall'],
                "roc_auc": eval_metrics['eval_roc_auc']
            }
            mlflow.log_metrics(metrics)
            logger.info(f"Run {run_name} completed with accuracy: {eval_metrics['eval_accuracy']}")

        except Exception as e:
            logger.error(f"Failed during run {run_name}: {e}")

        finally:
            # Cleanup to free memory
            del ft_model
            del unwrapped_model
            del model
            torch.cuda.empty_cache()
            gc.collect()
            log_memory_usage("after cleanup")
```

### **Key Components:**

- **Memory Logging:**
  
  The `log_memory_usage` function logs the memory usage at various stages to help you monitor memory consumption.

- **Encapsulation:**
  
  The `run_single_experiment` function encapsulates the logic for a single experiment run, ensuring that variables are out of scope after the function completes.

- **Explicit Cleanup:**
  
  The `finally` block ensures that all references to large objects are deleted and that memory is reclaimed, even if an error occurs during the experiment.

---

## **9. Additional Recommendations**

### **a. Use `torch.no_grad()` During Evaluation**

If your training loop or evaluation doesn't require gradient computations, wrapping them in `torch.no_grad()` can save memory.

```python
with torch.no_grad():
    outputs = model(**inputs)
```

### **b. Utilize Mixed Precision Training Wisely**

While mixed precision (`fp16`) can reduce memory usage, ensure that it's compatible with your hardware and doesn't introduce instability.

### **c. Profile Your Code**

Use profiling tools to identify memory bottlenecks.

- **`torch.cuda.memory_summary()`:**

  Provides a summary of CUDA memory usage.

  ```python
  logger.info(torch.cuda.memory_summary())
  ```

- **`tracemalloc`:**

  Python's built-in memory profiler.

  ```python
  import tracemalloc

  tracemalloc.start()
  # Your code here
  current, peak = tracemalloc.get_traced_memory()
  logger.info(f"Current memory usage: {current / 10**6}MB; Peak: {peak / 10**6}MB")
  tracemalloc.stop()
  ```

### **d. Limit Parallelism**

If you're using data loaders or other parallel processes, limit the number of workers to prevent excessive memory usage.

```python
DataLoader(..., num_workers=4)
```

### **e. Use Efficient Data Structures**

Ensure that you're using memory-efficient data structures for storing and processing data.

---

## **10. Summary**

The memory exception you're encountering is primarily due to residual references to large objects persisting across iterations of your experiment loop. By encapsulating each experiment within its own function, explicitly deleting references to large objects after each run, and ensuring that garbage collection is invoked properly, you can mitigate memory leaks effectively. Additionally, monitoring memory usage at various stages can provide insights into where optimizations are needed.

Implementing these strategies should help prevent memory exhaustion issues, allowing your experiments to run smoothly across all hyperparameter iterations.

Feel free to reach out if you encounter further challenges or need additional assistance!

Yes, you can log both the trained weights of your fine-tuned model and the corresponding tokenizer together as artifacts in MLflow. This approach ensures that both components are bundled within a single MLflow model, simplifying the inference process without the need to save them to an external output directory. Below, I'll outline two primary methods to achieve this:

1. **Using MLflow's Native Hugging Face Transformers Integration**
2. **Creating a Custom PyFunc Model to Bundle Model and Tokenizer**

---

## **1. Using MLflow's Native Hugging Face Transformers Integration**

MLflow provides built-in support for Hugging Face Transformers models, which inherently include both the model weights and the tokenizer. This integration simplifies the logging and loading processes.

### **Advantages:**

- **Simplicity:** Minimal code changes required.
- **Built-In Support:** MLflow handles both the model and tokenizer seamlessly.
- **Optimized:** Leverages MLflow's optimized methods for handling Transformers models.

### **Implementation Steps:**

1. **Ensure Dependencies:**

   Make sure you have the latest versions of `mlflow` and `transformers` installed.

   ```bash
   pip install --upgrade mlflow transformers
   ```

2. **Log the Transformers Model Using `mlflow.transformers`:**

   Utilize the `mlflow.transformers.log_model` function to log both the model and tokenizer together.

   ```python
   import mlflow
   import mlflow.transformers
   from transformers import AutoModelForSequenceClassification, AutoTokenizer

   # Assume 'ft_model' is your fine-tuned model
   # and 'tokenizer' is the corresponding tokenizer

   # Example:
   # ft_model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
   # tokenizer = AutoTokenizer.from_pretrained(self.output_dir)

   with mlflow.start_run(run_name="log_transformers_model"):
       # Log the model and tokenizer together
       mlflow.transformers.log_model(
           transformers_model=ft_model,
           tokenizer=tokenizer,
           artifact_path="transformers_model",
           registered_model_name="Your_Registered_Model_Name"
       )
   ```

3. **Loading the Registered Transformers Model:**

   When you need to perform inference, load the model directly using MLflow's Transformers integration.

   ```python
   import mlflow.transformers

   # Load the production model
   production_model = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_text = "Example input text for classification."
   inputs = production_model.tokenizer(input_text, return_tensors="pt")
   outputs = production_model.model(**inputs)
   predictions = outputs.logits.argmax(dim=1)
   print(f"Predicted class: {predictions.item()}")
   ```

### **Notes:**

- **Registered Model Name:** Replace `"Your_Registered_Model_Name"` with your actual model registry name.
- **Model URI:** Ensure the `model_uri` points to the correct stage (e.g., `"Production"`).

---

## **2. Creating a Custom PyFunc Model to Bundle Model and Tokenizer**

If you require more flexibility or are not using Hugging Face Transformers, you can create a custom PyFunc model that encapsulates both the model and tokenizer. This method involves defining a custom `PythonModel` class that handles both components.

### **Advantages:**

- **Flexibility:** Customize how the model and tokenizer are loaded and used.
- **Compatibility:** Works with models and tokenizers beyond Hugging Face Transformers.

### **Implementation Steps:**

1. **Define a Custom PyFunc Model Class:**

   Create a class that inherits from `mlflow.pyfunc.PythonModel` and manages both the model and tokenizer.

   ```python
   import mlflow.pyfunc
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

       def predict(self, context, model_input):
           # Assume model_input is a list of texts
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

2. **Log the Custom PyFunc Model with Both Model and Tokenizer as Artifacts:**

   Instead of saving to an output directory, use in-memory objects or temporary directories to handle the model and tokenizer. However, MLflow primarily works with file paths for artifacts. To align with MLflow's requirements while avoiding persistent disk writes, use Python's `tempfile` module to create temporary directories.

   ```python
   import mlflow.pyfunc
   import tempfile
   import os
   import shutil

   # Assume 'ft_model' is your fine-tuned model
   # and 'tokenizer' is the corresponding tokenizer

   with tempfile.TemporaryDirectory() as tmp_dir:
       model_path = os.path.join(tmp_dir, "model")
       tokenizer_path = os.path.join(tmp_dir, "tokenizer")

       # Save the model and tokenizer to the temporary directory
       ft_model.save_pretrained(model_path)
       tokenizer.save_pretrained(tokenizer_path)

       # Define artifacts dictionary
       artifacts = {
           "model_path": model_path,
           "tokenizer_path": tokenizer_path
       }

       # Log the PyFunc model
       mlflow.pyfunc.log_model(
           artifact_path="pyfunc_nli_model",
           python_model=PyFuncNLIModel(),
           artifacts=artifacts,
           conda_env={
               'name': 'nli_env',
               'channels': ['defaults'],
               'dependencies': [
                   'python=3.8',
                   'pip',
                   {
                       'pip': [
                           'torch',
                           'transformers',
                           'mlflow',
                           # Add other dependencies as needed
                       ],
                   },
               ],
           },
           registered_model_name="Your_Registered_Model_Name"
       )
   ```

3. **Register the Model:**

   Ensure that the model is registered in MLflow's Model Registry. The `registered_model_name` parameter in `log_model` handles this.

4. **Loading and Using the Custom PyFunc Model for Inference:**

   When performing inference, load the model using MLflow's PyFunc interface and utilize the bundled tokenizer and model.

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_texts = ["First example sentence.", "Second example sentence."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)
   ```

### **Key Points:**

- **Temporary Directories:** Using `tempfile.TemporaryDirectory()` ensures that intermediate files are not persisted, aligning with your preference to avoid saving to an output directory.
- **Artifacts Dictionary:** Both `model_path` and `tokenizer_path` are included in the artifacts, allowing the custom `PythonModel` to load them during inference.
- **Conda Environment:** Ensure all necessary dependencies are specified to guarantee reproducibility and compatibility during model loading and inference.

---

## **3. Comprehensive Example Integrating Both Model and Tokenizer Logging**

To provide a complete picture, here's an integrated example that encapsulates both the fine-tuning, logging (including model and tokenizer), and loading for inference without persisting to an external directory.

### **Step-by-Step Implementation:**

1. **Fine-Tune the Model and Obtain the Tokenizer:**

   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch
   import mlflow
   import mlflow.pyfunc
   import tempfile
   import os
   import shutil

   # Example fine-tuning function (simplified)
   def fine_tune_model(base_model_path, train_data, validation_data, hyperparameters):
       model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
       tokenizer = AutoTokenizer.from_pretrained(base_model_path)

       # Fine-tuning logic here
       # For illustration purposes, assume model is fine-tuned and returned
       # Replace with actual training code
       return model, tokenizer, {"accuracy": 0.95}
   ```

2. **Define the Custom PyFunc Model Class:**

   (As previously defined)

   ```python
   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

       def predict(self, context, model_input):
           # Assume model_input is a list of texts
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

3. **Fine-Tune, Log, and Register the Model:**

   ```python
   # Define hyperparameters
   hyperparameters = {
       "num_train_epochs": 3,
       "learning_rate": 2e-5,
       "weight_decay": 0.01,
       "per_device_train_batch_size": 16
   }

   # Base model path
   base_model_path = "distilbert-base-uncased"

   # Placeholder for train and validation data
   train_data = "path/to/train_data"
   validation_data = "path/to/validation_data"

   # Fine-tune the model
   ft_model, tokenizer, eval_metrics = fine_tune_model(base_model_path, train_data, validation_data, hyperparameters)

   # Start an MLflow run
   with mlflow.start_run(run_name="fine_tuned_nli_model") as run:
       # Log hyperparameters
       mlflow.log_params(hyperparameters)

       # Log metrics
       mlflow.log_metrics(eval_metrics)

       # Use temporary directories to avoid saving to disk
       with tempfile.TemporaryDirectory() as tmp_dir:
           model_path = os.path.join(tmp_dir, "model")
           tokenizer_path = os.path.join(tmp_dir, "tokenizer")

           # Save the model and tokenizer to the temporary directory
           ft_model.save_pretrained(model_path)
           tokenizer.save_pretrained(tokenizer_path)

           # Define artifacts dictionary
           artifacts = {
               "model_path": model_path,
               "tokenizer_path": tokenizer_path
           }

           # Log the PyFunc model
           mlflow.pyfunc.log_model(
               artifact_path="pyfunc_nli_model",
               python_model=PyFuncNLIModel(),
               artifacts=artifacts,
               conda_env={
                   'name': 'nli_env',
                   'channels': ['defaults'],
                   'dependencies': [
                       'python=3.8',
                       'pip',
                       {
                           'pip': [
                               'torch',
                               'transformers',
                               'mlflow',
                               # Add other dependencies as needed
                           ],
                       },
                   ],
               },
               registered_model_name="Your_Registered_Model_Name"
           )
   ```

4. **Loading and Using the Logged Model for Inference:**

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

### **Explanation:**

- **Temporary Directories:** By using `tempfile.TemporaryDirectory()`, the model and tokenizer are saved to temporary paths that are automatically cleaned up after logging, adhering to your preference to avoid persistent output directories.
- **Artifact Paths:** Both `model_path` and `tokenizer_path` are specified in the `artifacts` dictionary, ensuring they're accessible within the custom `PyFuncNLIModel`.
- **Conda Environment:** Specifies the necessary dependencies to ensure the model and tokenizer can be loaded correctly during inference.
- **Registered Model Name:** Ensures that the model is available in MLflow's Model Registry for easy access and deployment.

---

## **4. Alternative: Using `mlflow.pytorch.log_model` with Custom Artifacts**

If you prefer to continue using `mlflow.pytorch.log_model` and want to include the tokenizer as an additional artifact, you can do so by leveraging MLflow's artifact logging capabilities. However, this method is less integrated compared to using `mlflow.transformers` or a custom PyFunc model.

### **Implementation Steps:**

1. **Define a Custom PythonModel to Handle Both Model and Tokenizer:**

   ```python
   import mlflow.pyfunc
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class CustomPyTorchTokenizerModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the PyTorch model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

       def predict(self, context, model_input):
           # Assume model_input is a list of texts
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

2. **Log the Model and Tokenizer as Separate Artifacts:**

   ```python
   import mlflow.pytorch
   import tempfile
   import os

   # Assume 'ft_model' is your fine-tuned model and 'tokenizer' is your tokenizer

   with tempfile.TemporaryDirectory() as tmp_dir:
       model_path = os.path.join(tmp_dir, "model")
       tokenizer_path = os.path.join(tmp_dir, "tokenizer")

       # Save model and tokenizer to temporary paths
       ft_model.save_pretrained(model_path)
       tokenizer.save_pretrained(tokenizer_path)

       # Define artifacts
       artifacts = {
           "model_path": model_path,
           "tokenizer_path": tokenizer_path
       }

       # Log the custom PyFunc model
       mlflow.pyfunc.log_model(
           artifact_path="custom_pytorch_tokenizer_model",
           python_model=CustomPyTorchTokenizerModel(),
           artifacts=artifacts,
           conda_env={
               'name': 'custom_env',
               'channels': ['defaults'],
               'dependencies': [
                   'python=3.8',
                   'pip',
                   {
                       'pip': [
                           'torch',
                           'transformers',
                           'mlflow',
                           # Add other dependencies as needed
                       ],
                   },
               ],
           },
           registered_model_name="Your_Registered_Model_Name"
       )
   ```

3. **Loading and Using the Custom Model:**

   ```python
   import mlflow.pyfunc

   # Load the registered model
   custom_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Perform inference
   input_texts = ["I enjoy coding.", "This is a bad example."]
   predictions = custom_model.predict(input_texts)
   print(predictions)
   ```

### **Key Points:**

- **Separate Artifacts:** Both `model_path` and `tokenizer_path` are logged as separate artifacts within the MLflow model.
- **Custom PyFunc Model:** The `CustomPyTorchTokenizerModel` class handles the loading and prediction logic, ensuring both the model and tokenizer are utilized during inference.
- **Conda Environment:** Specifies necessary dependencies, ensuring the environment is reproducible.

---

## **5. Best Practices and Recommendations**

1. **Prefer Built-In Integrations When Available:**

   If you're working with Hugging Face Transformers models, leveraging `mlflow.transformers` is recommended due to its simplicity and built-in support for both models and tokenizers.

2. **Use Custom PyFunc Models for Flexibility:**

   When working with models outside of MLflow's native integrations or requiring custom handling, defining a `PythonModel` is an effective approach.

3. **Avoid Persistent Output Directories:**

   Utilize temporary directories (`tempfile`) to handle intermediate files during logging, ensuring that your system remains clean and no unnecessary files are persisted.

4. **Ensure Environment Reproducibility:**

   Clearly define the `conda_env` or use `requirements.txt` to specify all dependencies, ensuring that the model can be loaded and run in different environments without issues.

5. **Register Models in MLflow's Model Registry:**

   Registering models facilitates versioning, stage transitions (e.g., Staging to Production), and provides a centralized location for managing your models.

6. **Test Model Loading and Inference:**

   After logging and registering your models, perform test inferences to ensure that both the model and tokenizer are functioning as expected.

---

## **6. Summary**

By utilizing MLflow's native Hugging Face Transformers integration or creating a custom PyFunc model, you can effectively log both the trained weights of your fine-tuned model and the corresponding tokenizer as artifacts within a single MLflow model. This bundling simplifies the inference process, allowing you to load and use the model without managing separate files or directories manually.

Choose the method that best aligns with your project's requirements and the frameworks you are using. Leveraging MLflow's capabilities ensures a streamlined workflow for model management, tracking, and deployment.

---

Feel free to reach out if you need further clarification or assistance with specific parts of the implementation!

Yes, you can log both the trained weights of your fine-tuned model and the corresponding tokenizer together as artifacts in MLflow without persisting them to a permanent output directory. While MLflow's artifact logging inherently relies on file-based storage, you can streamline this process by utilizing temporary directories. This approach ensures that artifacts are managed efficiently without cluttering your file system.

Below, I'll guide you through two primary methods to achieve this:

1. **Using MLflow's Native Hugging Face Transformers Integration**
2. **Creating a Custom PyFunc Model with Temporary Artifacts**

Both methods ensure that the model and tokenizer are logged together, facilitating seamless inference without manual handling of output directories.

---

## **1. Using MLflow's Native Hugging Face Transformers Integration**

MLflow offers built-in support for Hugging Face Transformers, which inherently manages both the model weights and the tokenizer. This integration simplifies the logging and loading processes, ensuring that both components are bundled together effortlessly.

### **Advantages:**

- **Simplicity:** Minimal code changes required.
- **Built-In Support:** MLflow handles both the model and tokenizer seamlessly.
- **Optimized:** Leverages MLflow's optimized methods for handling Transformers models.

### **Implementation Steps:**

1. **Ensure Dependencies:**

   Make sure you have the latest versions of `mlflow` and `transformers` installed.

   ```bash
   pip install --upgrade mlflow transformers
   ```

2. **Log the Transformers Model Using `mlflow.transformers.log_model`:**

   Utilize the `mlflow.transformers.log_model` function to log both the model and tokenizer together. This method handles the temporary storage internally, ensuring no persistent directories are created.

   ```python
   import mlflow
   import mlflow.transformers
   from transformers import AutoModelForSequenceClassification, AutoTokenizer

   # Assume 'ft_model' is your fine-tuned model
   # and 'tokenizer' is the corresponding tokenizer

   with mlflow.start_run(run_name="log_transformers_model"):
       # Log the model and tokenizer together
       mlflow.transformers.log_model(
           transformers_model=ft_model,
           tokenizer=tokenizer,
           artifact_path="transformers_model",
           registered_model_name="Your_Registered_Model_Name"
       )
   ```

3. **Loading the Registered Transformers Model for Inference:**

   When you need to perform inference, load the model directly using MLflow's Transformers integration.

   ```python
   import mlflow.transformers

   # Load the production model
   production_model = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_text = "Example input text for classification."
   inputs = production_model.tokenizer(input_text, return_tensors="pt")
   outputs = production_model.model(**inputs)
   predictions = outputs.logits.argmax(dim=1)
   print(f"Predicted class: {predictions.item()}")
   ```

### **Key Points:**

- **Registered Model Name:** Replace `"Your_Registered_Model_Name"` with your actual model registry name.
- **Model URI:** Ensure the `model_uri` points to the correct stage (e.g., `"Production"`).


---

## **2. Creating a Custom PyFunc Model with Temporary Artifacts**

If you require more flexibility or are working with models outside of Hugging Face Transformers, you can create a custom PyFunc model that encapsulates both the model and tokenizer. This method involves defining a custom `PythonModel` class that handles both components, utilizing temporary directories to manage artifacts internally.

### **Advantages:**

- **Flexibility:** Customize how the model and tokenizer are loaded and used.
- **Compatibility:** Works with models and tokenizers beyond Hugging Face Transformers.

### **Implementation Steps:**

1. **Define a Custom PyFunc Model Class:**

   Create a class that inherits from `mlflow.pyfunc.PythonModel` and manages both the model and tokenizer.

   ```python
   import mlflow.pyfunc
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

       def predict(self, context, model_input):
           # Assume model_input is a list of texts
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

2. **Log the Custom PyFunc Model with Both Model and Tokenizer as Artifacts:**

   Utilize Python's `tempfile` module to create temporary directories for the model and tokenizer, ensuring that no persistent directories are created.

   ```python
   import mlflow.pyfunc
   import tempfile
   import os

   # Assume 'ft_model' is your fine-tuned model
   # and 'tokenizer' is the corresponding tokenizer

   with mlflow.start_run(run_name="log_custom_pyfunc_model"):
       # Log any parameters or metrics as needed
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)

       # Use temporary directories to avoid saving to disk
       with tempfile.TemporaryDirectory() as tmp_dir:
           model_path = os.path.join(tmp_dir, "model")
           tokenizer_path = os.path.join(tmp_dir, "tokenizer")

           # Save the model and tokenizer to the temporary directory
           ft_model.save_pretrained(model_path)
           tokenizer.save_pretrained(tokenizer_path)

           # Define artifacts dictionary
           artifacts = {
               "model_path": model_path,
               "tokenizer_path": tokenizer_path
           }

           # Log the PyFunc model
           mlflow.pyfunc.log_model(
               artifact_path="custom_pyfunc_nli_model",
               python_model=PyFuncNLIModel(),
               artifacts=artifacts,
               conda_env={
                   'name': 'nli_env',
                   'channels': ['defaults'],
                   'dependencies': [
                       'python=3.8',
                       'pip',
                       {
                           'pip': [
                               'torch',
                               'transformers',
                               'mlflow',
                               # Add other dependencies as needed
                           ],
                       },
                   ],
               },
               registered_model_name="Your_Registered_Model_Name"
           )
   ```

3. **Loading and Using the Custom PyFunc Model for Inference:**

   Load the registered model using MLflow's PyFunc interface and perform inference seamlessly.

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

### **Key Points:**

- **Temporary Directories:** Using `tempfile.TemporaryDirectory()` ensures that the model and tokenizer are saved to temporary paths that are automatically cleaned up after logging.
- **Artifact Paths:** Both `model_path` and `tokenizer_path` are included in the artifacts, allowing the custom `PythonModel` to load them during inference.
- **Conda Environment:** Specifies the necessary dependencies to ensure the model and tokenizer can be loaded correctly during inference.
- **Registered Model Name:** Ensures that the model is available in MLflow's Model Registry for easy access and deployment.

---

## **3. Comprehensive Example Integrating Both Model and Tokenizer Logging**

To provide a complete and cohesive workflow, here's an integrated example that encapsulates fine-tuning, logging (including both model and tokenizer), and loading for inference without persisting to an external directory.

### **Step-by-Step Implementation:**

1. **Fine-Tune the Model and Obtain the Tokenizer:**

   ```python
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   # Example fine-tuning function (simplified)
   def fine_tune_model(base_model_path, train_data, validation_data, hyperparameters):
       model = AutoModelForSequenceClassification.from_pretrained(base_model_path)
       tokenizer = AutoTokenizer.from_pretrained(base_model_path)

       # Placeholder for fine-tuning logic
       # Replace with actual training code
       # For demonstration, assume the model is fine-tuned and returned along with the tokenizer and metrics
       fine_tuned_model = model  # Replace with actual fine-tuned model
       metrics = {"accuracy": 0.95}  # Replace with actual evaluation metrics

       return fine_tuned_model, tokenizer, metrics
   ```

2. **Define the Custom PyFunc Model Class:**

   ```python
   import mlflow.pyfunc
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

       def predict(self, context, model_input):
           # Assume model_input is a list of texts
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

3. **Fine-Tune, Log, and Register the Model:**

   ```python
   import mlflow.pyfunc
   import tempfile
   import os

   # Define hyperparameters
   hyperparameters = {
       "num_train_epochs": 3,
       "learning_rate": 2e-5,
       "weight_decay": 0.01,
       "per_device_train_batch_size": 16
   }

   # Base model path (e.g., a Hugging Face model name)
   base_model_path = "distilbert-base-uncased"

   # Placeholder for train and validation data paths
   train_data = "path/to/train_data"
   validation_data = "path/to/validation_data"

   # Fine-tune the model
   ft_model, tokenizer, eval_metrics = fine_tune_model(base_model_path, train_data, validation_data, hyperparameters)

   # Start an MLflow run
   with mlflow.start_run(run_name="fine_tuned_nli_model") as run:
       # Log hyperparameters
       mlflow.log_params(hyperparameters)

       # Log metrics
       mlflow.log_metrics(eval_metrics)

       # Use temporary directories to avoid saving to disk
       with tempfile.TemporaryDirectory() as tmp_dir:
           model_path = os.path.join(tmp_dir, "model")
           tokenizer_path = os.path.join(tmp_dir, "tokenizer")

           # Save the model and tokenizer to the temporary directory
           ft_model.save_pretrained(model_path)
           tokenizer.save_pretrained(tokenizer_path)

           # Define artifacts dictionary
           artifacts = {
               "model_path": model_path,
               "tokenizer_path": tokenizer_path
           }

           # Log the PyFunc model
           mlflow.pyfunc.log_model(
               artifact_path="custom_pyfunc_nli_model",
               python_model=PyFuncNLIModel(),
               artifacts=artifacts,
               conda_env={
                   'name': 'nli_env',
                   'channels': ['defaults'],
                   'dependencies': [
                       'python=3.8',
                       'pip',
                       {
                           'pip': [
                               'torch',
                               'transformers',
                               'mlflow',
                               # Add other dependencies as needed
                           ],
                       },
                   ],
               },
               registered_model_name="Your_Registered_Model_Name"
           )
   ```

4. **Loading and Using the Logged Model for Inference:**

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

### **Explanation:**

- **Temporary Directories:** By using `tempfile.TemporaryDirectory()`, the model and tokenizer are saved to temporary paths that are automatically cleaned up after logging, adhering to your preference to avoid persistent output directories.
  
- **Artifact Paths:** Both `model_path` and `tokenizer_path` are specified in the artifacts dictionary, ensuring they're accessible within the custom `PyFuncNLIModel`.

- **Conda Environment:** Specifies the necessary dependencies to ensure the model and tokenizer can be loaded correctly during inference.

- **Registered Model Name:** Ensures that the model is available in MLflow's Model Registry for easy access and deployment.

---

## **3. Alternative: Using In-Memory File Systems (Advanced)**

If you **strictly** want to avoid writing to any disk-based directories, you can explore using in-memory file systems like [pyfilesystem2](https://docs.pyfilesystem.org/en/latest/), but this approach is more complex and may not be fully compatible with MLflow's artifact logging mechanisms, which expect file paths.

**Note:** This method is advanced and generally not recommended unless you have specific constraints that necessitate it.

### **Implementation Steps:**

1. **Install `pyfilesystem2`:**

   ```bash
   pip install fs
   ```

2. **Use an In-Memory File System:**

   ```python
   import mlflow.pyfunc
   from fs.memoryfs import MemoryFS
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class PyFuncNLIModelInMemory(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           from transformers import AutoModelForSequenceClassification, AutoTokenizer
           import torch

           # Access the in-memory file system
           model_fs = context.artifacts["model_fs"]
           tokenizer_fs = context.artifacts["tokenizer_fs"]

           # Load the model from in-memory FS
           with model_fs.openbin('model/pytorch_model.bin') as f:
               self.model = AutoModelForSequenceClassification.from_pretrained(model_fs.root_path)
               self.model.load_state_dict(torch.load(f))
           self.model.eval()

           # Load the tokenizer from in-memory FS
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_fs.root_path)

       def predict(self, context, model_input):
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()

   # Example usage
   with mlflow.start_run(run_name="log_in_memory_pyfunc_model"):
       # Fine-tune your model and tokenizer
       ft_model, tokenizer, eval_metrics = fine_tune_model(base_model_path, train_data, validation_data, hyperparameters)

       # Create in-memory file systems
       model_fs = MemoryFS()
       tokenizer_fs = MemoryFS()

       # Save the model and tokenizer to in-memory FS
       ft_model.save_pretrained('/')  # Saves to the root of the MemoryFS
       tokenizer.save_pretrained('/')

       # Define artifacts
       artifacts = {
           "model_fs": model_fs,
           "tokenizer_fs": tokenizer_fs
       }

       # Log the PyFunc model
       mlflow.pyfunc.log_model(
           artifact_path="in_memory_pyfunc_model",
           python_model=PyFuncNLIModelInMemory(),
           artifacts=artifacts,
           conda_env={
               'name': 'nli_env',
               'channels': ['defaults'],
               'dependencies': [
                   'python=3.8',
                   'pip',
                   {
                       'pip': [
                           'torch',
                           'transformers',
                           'mlflow',
                           'fs',  # pyfilesystem2
                           # Add other dependencies as needed
                       ],
                   },
               ],
           },
           registered_model_name="Your_InMemory_Registered_Model_Name"
       )
   ```

### **Caveats:**

- **Compatibility:** MLflow's artifact logging expects file paths. Integrating in-memory file systems can lead to compatibility issues and is not straightforward.
  
- **Performance:** This approach may introduce overhead and complexity without significant benefits over using temporary directories.

**Recommendation:** Stick to using temporary directories as outlined in the previous sections for a more reliable and simpler implementation.

---

## **4. Best Practices and Recommendations**

1. **Prefer Built-In Integrations When Available:**

   If you're working with Hugging Face Transformers models, leveraging `mlflow.transformers` is recommended due to its simplicity and built-in support for both models and tokenizers.

2. **Use Custom PyFunc Models for Flexibility:**

   When working with models outside of MLflow's native integrations or requiring custom handling, defining a `PythonModel` is an effective approach.

3. **Avoid Persistent Output Directories:**

   Utilize temporary directories (`tempfile`) to handle intermediate files during logging, ensuring that your system remains clean and no unnecessary files are persisted.

4. **Ensure Environment Reproducibility:**

   Clearly define the `conda_env` or use `requirements.txt` to specify all dependencies, ensuring that the model can be loaded and run in different environments without issues.

5. **Register Models in MLflow's Model Registry:**

   Registering models facilitates versioning, stage transitions (e.g., Staging to Production), and provides a centralized location for managing your models.

6. **Test Model Loading and Inference:**

   After logging and registering your models, perform test inferences to ensure that both the model and tokenizer are functioning as expected.

---

## **5. Summary**

While MLflow's artifact logging system is inherently file-based, you can efficiently manage both the trained model and tokenizer without persisting them to permanent directories by utilizing temporary storage solutions. Whether you leverage MLflow's native Hugging Face Transformers integration for simplicity or create a custom PyFunc model for added flexibility, both approaches ensure that your model and tokenizer are bundled together seamlessly within MLflow. This setup facilitates straightforward inference processes, aligning with best practices for model management and deployment.

By adopting these strategies, you can maintain a clean file system while ensuring that all necessary components of your model are tracked and versioned effectively within MLflow.

---

Feel free to reach out if you encounter any challenges or need further assistance with specific parts of the implementation!



## ---------------------- Other issues

MlflowException: The task type cannot be inferred from the submitted Pipeline or dictionary of model components. Please provide the task type explicitly when saving or logging this submitted Pipeline or dictionary of components.

```
                        components = {
                            "model": ft_model,
                            "tokenizer": tokenizer
                            }
                        
                        mlflow.transformers.log_model(
                            transformers_model=components,
                            artifact_path="model")
                        logger.info(f"Model logged successfully")
```