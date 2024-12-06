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

The error you're encountering stems from how you're passing the model and tokenizer to the `mlflow.transformers.log_model` function. Specifically, MLflow's `log_model` method for Transformers expects a **single model instance** (either a `PreTrainedModel` or a `Pipeline`), **not a dictionary** containing multiple components like the model and tokenizer.

Here's the error message you received for reference:

```
MlflowException: The task type cannot be inferred from the submitted Pipeline or dictionary of model components. Please provide the task type explicitly when saving or logging this submitted Pipeline or dictionary of components.
```

This indicates that MLflow is unable to determine the type of task your model is performing because it's receiving an unexpected input format (a dictionary).

To resolve this issue, you have a couple of effective options:

1. **Use a Hugging Face `Pipeline` that Bundles Both Model and Tokenizer**
2. **Create a Custom PyFunc Model to Handle Both Components**

Below, I will guide you through both methods, ensuring you can log both the model and tokenizer seamlessly without persisting them to an external output directory.

---

## **1. Using a Hugging Face `Pipeline` to Bundle Model and Tokenizer**

Hugging Face's `Pipeline` is a high-level API that abstracts away the complexity of handling models and tokenizers separately. By encapsulating both within a `Pipeline`, you provide MLflow with a single, coherent object to log, eliminating the need to manage multiple artifacts.

### **Advantages:**

- **Simplicity:** Minimal code changes required.
- **Integrated Handling:** Both model and tokenizer are managed together.
- **Task Specification:** Easily specify the task type, aiding MLflow in inferring necessary details.

### **Implementation Steps:**

1. **Ensure Dependencies Are Installed:**

   Make sure you have the latest versions of `mlflow` and `transformers` installed.

   ```bash
   pip install --upgrade mlflow transformers
   ```

2. **Create a Hugging Face `Pipeline`:**

   Initialize a `Pipeline` that includes both your fine-tuned model and tokenizer.

   ```python
   from transformers import pipeline

   # Assume 'ft_model' is your fine-tuned AutoModelForSequenceClassification instance
   # and 'tokenizer' is the corresponding tokenizer instance

   nli_pipeline = pipeline(
       task="text-classification",  # Specify the appropriate task
       model=ft_model,
       tokenizer=tokenizer
   )
   ```

3. **Log the `Pipeline` with MLflow:**

   Use `mlflow.transformers.log_model` to log the entire `Pipeline`. This method handles both the model and tokenizer internally.

   ```python
   import mlflow
   import mlflow.transformers

   with mlflow.start_run(run_name="log_transformers_pipeline"):
       # Optionally log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

       # Log the Transformers Pipeline
       mlflow.transformers.log_model(
           transformers_model=nli_pipeline,
           artifact_path="transformers_pipeline",
           registered_model_name="Your_Registered_Model_Name"  # Replace with your model registry name
       )

       print("Model and tokenizer logged successfully!")
   ```

4. **Loading the Logged `Pipeline` for Inference:**

   When you need to perform inference, load the model directly using MLflow's Transformers integration.

   ```python
   import mlflow.transformers

   # Load the production model
   production_pipeline = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_text = "I love machine learning!"
   outputs = production_pipeline(input_text)
   print(f"Predicted class: {outputs[0]['label']} with score {outputs[0]['score']}")
   ```

### **Key Points:**

- **Task Specification:** Ensure you specify the correct task (e.g., `"text-classification"`) when creating the `Pipeline`. This helps MLflow infer necessary configurations.
  
- **Registered Model Name:** Replace `"Your_Registered_Model_Name"` with the actual name you intend to use in MLflow's Model Registry.

- **Artifact Path:** The `artifact_path` parameter defines where within the MLflow run the model will be stored. Adjust as necessary.

---

## **2. Creating a Custom PyFunc Model to Handle Both Model and Tokenizer**

If you require more flexibility or are working with models outside of Hugging Face Transformers' native integrations, you can create a custom PyFunc model that encapsulates both the model and tokenizer. This approach involves defining a custom `PythonModel` class that handles the loading and prediction logic.

### **Advantages:**

- **Flexibility:** Full control over how the model and tokenizer are loaded and utilized.
- **Compatibility:** Suitable for models and tokenizers not directly supported by MLflow's native integrations.

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

   # Assume 'ft_model' is your fine-tuned AutoModelForSequenceClassification instance
   # and 'tokenizer' is the corresponding tokenizer instance

   with mlflow.start_run(run_name="log_custom_pyfunc_model"):
       # Optionally log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

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
               registered_model_name="Your_Registered_Model_Name"  # Replace accordingly
           )

       print("Custom PyFunc model with both model and tokenizer logged successfully!")
   ```

3. **Loading and Using the Custom PyFunc Model for Inference:**

   Load the registered model using MLflow's PyFunc interface and perform inference seamlessly.

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

### **Key Points:**

- **Temporary Directories:** Using `tempfile.TemporaryDirectory()` ensures that the model and tokenizer are saved to temporary paths that are automatically cleaned up after logging.
  
- **Artifact Paths:** Both `model_path` and `tokenizer_path` are specified in the artifacts dictionary, ensuring they're accessible within the custom `PyFuncNLIModel`.
  
- **Conda Environment:** Specifies the necessary dependencies to ensure the model and tokenizer can be loaded correctly during inference.
  
- **Registered Model Name:** Ensures that the model is available in MLflow's Model Registry for easy access and deployment.

---

## **Addressing the Specific Error**

Now, let's specifically address the error you encountered:

```
MlflowException: The task type cannot be inferred from the submitted Pipeline or dictionary of model components. Please provide the task type explicitly when saving or logging this submitted Pipeline or dictionary of components.
```

### **Cause of the Error:**

- **Incorrect Input Format:** You're passing a dictionary containing both the model and tokenizer to `mlflow.transformers.log_model`, which expects a single model instance (either a `PreTrainedModel` or a `Pipeline`).
  
- **Task Type Inference:** MLflow attempts to infer the task type (e.g., `text-classification`) based on the provided model. Passing a dictionary disrupts this inference process.

### **Solution:**

**Option 1: Use a Hugging Face `Pipeline`**

Instead of passing a dictionary, encapsulate both the model and tokenizer within a `Pipeline` and pass the `Pipeline` instance to `mlflow.transformers.log_model`.

```python
from transformers import pipeline
import mlflow.transformers

# Create a Pipeline that includes both model and tokenizer
nli_pipeline = pipeline(
    task="text-classification",  # Specify the task explicitly
    model=ft_model,
    tokenizer=tokenizer
)

# Log the Pipeline
mlflow.transformers.log_model(
    transformers_model=nli_pipeline,
    artifact_path="transformers_pipeline",
    registered_model_name="Your_Registered_Model_Name"  # Replace with your model registry name
)

print("Pipeline with both model and tokenizer logged successfully!")
```

**Option 2: Specify the Task Type Explicitly**

If you prefer not to use a `Pipeline`, ensure you pass only the model to `transformers_model` and specify the `task` parameter explicitly.

```python
import mlflow.transformers

# Log only the model and specify the task
mlflow.transformers.log_model(
    transformers_model=ft_model,  # Pass only the model, not a dictionary
    artifact_path="transformers_model",
    task="text-classification",  # Specify the task explicitly
    registered_model_name="Your_Registered_Model_Name"  # Replace as needed
)

print("Model logged successfully with task type specified!")
```

**Important:** This approach assumes that MLflow can internally handle the tokenizer associated with the model. However, to ensure both model and tokenizer are available during inference, using a `Pipeline` or a custom PyFunc model is more reliable.

---

## **Recommendation**

Given your preference **not** to save models and tokenizers to an output directory, the **best approach** is to **use a Hugging Face `Pipeline`**. This method seamlessly bundles both the model and tokenizer, allowing MLflow to manage them together without the need for separate artifact paths or temporary directories.

### **Complete Example Using a Hugging Face `Pipeline`:**

1. **Initialize the Pipeline:**

   ```python
   from transformers import pipeline
   import mlflow.transformers

   # Create a Pipeline that includes both the model and tokenizer
   nli_pipeline = pipeline(
       task="text-classification",  # Specify the appropriate task
       model=ft_model,
       tokenizer=tokenizer
   )
   ```

2. **Log the Pipeline with MLflow:**

   ```python
   with mlflow.start_run(run_name="log_transformers_pipeline"):
       # Optionally log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

       # Log the Transformers Pipeline
       mlflow.transformers.log_model(
           transformers_model=nli_pipeline,
           artifact_path="transformers_pipeline",
           registered_model_name="Your_Registered_Model_Name"  # Replace with your model registry name
       )

       print("Pipeline with both model and tokenizer logged successfully!")
   ```

3. **Load and Use the Logged Pipeline for Inference:**

   ```python
   import mlflow.transformers

   # Load the production model
   production_pipeline = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_text = "I love machine learning!"
   outputs = production_pipeline(input_text)
   print(f"Predicted class: {outputs[0]['label']} with score {outputs[0]['score']}")
   ```

### **Benefits of This Approach:**

- **No Need for External Directories:** The `Pipeline` manages both components internally, eliminating the need for separate directories.
  
- **Simplified Logging and Loading:** MLflow treats the `Pipeline` as a single entity, streamlining the logging and inference processes.
  
- **Task Clarity:** Explicitly specifying the task aids MLflow in managing the model correctly.

---

## **Alternative: Using a Custom PyFunc Model with Temporary Artifacts**

If, for any reason, using a `Pipeline` doesn't align with your requirements, you can revert to the custom PyFunc model approach. However, this method involves more steps and complexity.

### **Steps Overview:**

1. **Define the Custom `PythonModel` Class:** As shown earlier, handle both model and tokenizer within this class.
2. **Use Temporary Directories:** Save the model and tokenizer to temporary paths before logging.
3. **Log the Model with MLflow:** Pass the paths to the artifacts dictionary.
4. **Load and Use the Model:** Utilize MLflow's PyFunc interface for inference.

### **Key Considerations:**

- **Increased Complexity:** More boilerplate code compared to using a `Pipeline`.
- **Dependency Management:** Ensure all necessary dependencies are correctly specified in the `conda_env`.

Given these factors, the **`Pipeline` approach is recommended** for its simplicity and efficiency.

---

## **Summary**

The error you encountered arises from passing a dictionary to `mlflow.transformers.log_model` instead of a single model instance. To resolve this and effectively log both the model and tokenizer:

1. **Use a Hugging Face `Pipeline`:** This method bundles both components seamlessly, allowing MLflow to manage them together without additional complexity.
  
2. **Ensure Correct Parameters:** If opting not to use a `Pipeline`, pass only the model instance and specify the task explicitly.

3. **Consider a Custom PyFunc Model:** For scenarios requiring more flexibility, create a custom PyFunc model that handles both the model and tokenizer, managing artifacts via temporary directories.

By adopting the `Pipeline` approach, you align with MLflow's expected input formats, ensuring smooth logging and inference processes without the need to persist artifacts to external directories.

---

**Feel free to reach out if you need further assistance or encounter additional issues!**

Certainly! Let's address both of your concerns step-by-step:

1. **Logging Both the Model and Tokenizer Together in MLflow Without Using an Output Directory**
2. **Storing Custom Hyperparameters in the Hugging Face `Config` for Later Use**

---

## **1. Logging Both the Model and Tokenizer Together in MLflow Without Using an Output Directory**

### **Understanding the Issue**

You're encountering the following error when attempting to log both the model and tokenizer:

```
MlflowException: The task type cannot be inferred from the submitted Pipeline or dictionary of model components. Please provide the task type explicitly when saving or logging this submitted Pipeline or dictionary of components.
```

This error arises because `mlflow.transformers.log_model` expects a **single model instance** (like `AutoModelForSequenceClassification` or a `Pipeline`), **not a dictionary** containing multiple components such as the model and tokenizer.

### **Solution Overview**

To log both the model and tokenizer together seamlessly, you have two primary approaches:

1. **Use a Hugging Face `Pipeline`**: Encapsulate both the model and tokenizer within a `Pipeline` and log the entire pipeline.
2. **Create a Custom PyFunc Model**: Define a custom MLflow PyFunc model that handles both the model and tokenizer.

Below, I'll guide you through both methods.

---

### **Approach 1: Using a Hugging Face `Pipeline`**

Hugging Face's `Pipeline` is a high-level API that combines the model and tokenizer, making it easier to manage and deploy.

#### **Steps to Implement**

1. **Ensure Dependencies Are Installed**

   Make sure you have the latest versions of `mlflow` and `transformers` installed:

   ```bash
   pip install --upgrade mlflow transformers
   ```

2. **Create a Hugging Face `Pipeline`**

   Initialize a `Pipeline` that includes both your fine-tuned model and tokenizer.

   ```python
   from transformers import pipeline

   # Assume 'ft_model' is your fine-tuned AutoModelForSequenceClassification instance
   # and 'tokenizer' is the corresponding tokenizer instance

   nli_pipeline = pipeline(
       task="text-classification",  # Specify the appropriate task
       model=ft_model,
       tokenizer=tokenizer
   )
   ```

3. **Log the `Pipeline` with MLflow**

   Use `mlflow.transformers.log_model` to log the entire `Pipeline`. This method handles both the model and tokenizer internally.

   ```python
   import mlflow
   import mlflow.transformers

   with mlflow.start_run(run_name="log_transformers_pipeline"):
       # Optionally log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

       # Log the Transformers Pipeline
       mlflow.transformers.log_model(
           transformers_model=nli_pipeline,
           artifact_path="transformers_pipeline",
           registered_model_name="Your_Registered_Model_Name"  # Replace with your model registry name
       )

       print("Pipeline with both model and tokenizer logged successfully!")
   ```

4. **Loading the Logged `Pipeline` for Inference**

   When you need to perform inference, load the model directly using MLflow's Transformers integration.

   ```python
   import mlflow.transformers

   # Load the production model
   production_pipeline = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_text = "I love machine learning!"
   outputs = production_pipeline(input_text)
   print(f"Predicted class: {outputs[0]['label']} with score {outputs[0]['score']}")
   ```

#### **Key Points**

- **Task Specification**: Ensure you specify the correct task (e.g., `"text-classification"`) when creating the `Pipeline`. This helps MLflow infer necessary configurations.
- **Registered Model Name**: Replace `"Your_Registered_Model_Name"` with the actual name you intend to use in MLflow's Model Registry.
- **Artifact Path**: The `artifact_path` parameter defines where within the MLflow run the model will be stored. Adjust as necessary.

---

### **Approach 2: Creating a Custom PyFunc Model**

If you require more flexibility or are working with models outside of Hugging Face Transformers' native integrations, you can create a custom PyFunc model that encapsulates both the model and tokenizer.

#### **Steps to Implement**

1. **Define a Custom PyFunc Model Class**

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

2. **Log the Custom PyFunc Model with Both Model and Tokenizer as Artifacts**

   Utilize Python's `tempfile` module to create temporary directories for the model and tokenizer, ensuring that no persistent directories are created.

   ```python
   import mlflow.pyfunc
   import tempfile
   import os

   # Assume 'ft_model' is your fine-tuned AutoModelForSequenceClassification instance
   # and 'tokenizer' is the corresponding tokenizer instance

   with mlflow.start_run(run_name="log_custom_pyfunc_model"):
       # Optionally log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

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
               registered_model_name="Your_Registered_Model_Name"  # Replace accordingly
           )

       print("Custom PyFunc model with both model and tokenizer logged successfully!")
   ```

3. **Loading and Using the Custom PyFunc Model for Inference**

   Load the registered model using MLflow's PyFunc interface and perform inference seamlessly.

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

#### **Key Points**

- **Temporary Directories**: Using `tempfile.TemporaryDirectory()` ensures that the model and tokenizer are saved to temporary paths that are automatically cleaned up after logging.
- **Artifact Paths**: Both `model_path` and `tokenizer_path` are specified in the artifacts dictionary, ensuring they're accessible within the custom `PyFuncNLIModel`.
- **Conda Environment**: Specifies the necessary dependencies to ensure the model and tokenizer can be loaded correctly during inference.
- **Registered Model Name**: Ensures that the model is available in MLflow's Model Registry for easy access and deployment.

---

## **2. Storing Custom Hyperparameters in the Hugging Face `Config` for Later Use**

You want to store custom hyperparameters used during fine-tuning within the model's configuration (`Config`) for later retrieval. This is a common practice to ensure reproducibility and to retain information about the training process.

### **Understanding Hugging Face `Config`**

The Hugging Face `Config` objects (`AutoConfig`, `PretrainedConfig`, etc.) are designed to store configuration parameters for models. They inherently support **additional attributes**, allowing you to store custom hyperparameters alongside standard ones.

### **Steps to Implement**

1. **Initialize the Config with Custom Hyperparameters**

   When creating the `Config` object during fine-tuning, you can add custom hyperparameters by setting them as attributes.

   ```python
   from transformers import AutoConfig

   # Example hyperparameters
   custom_hyperparameters = {
       "dropout_rate": 0.3,
       "optimizer": "adamw",
       "scheduler": "linear"
   }

   config = AutoConfig.from_pretrained(
       model_args.config_name if model_args.config_name else model_args.model_name_or_path,
       num_labels=num_labels,
       finetuning_task=data_args.task_name,
       cache_dir=model_args.cache_dir,
       revision=model_args.model_revision,
       trust_remote_code=model_args.trust_remote_code,
   )

   # Add custom hyperparameters to the config
   for key, value in custom_hyperparameters.items():
       setattr(config, key, value)
   ```

2. **Save the Config with the Model**

   When you save the model using `save_pretrained`, the `Config` (including your custom hyperparameters) is automatically saved alongside the model.

   ```python
   # Assuming 'ft_model' is your fine-tuned model instance
   ft_model.save_pretrained(self.output_dir)
   config.save_pretrained(self.output_dir)  # Saves the config.json with custom hyperparameters
   ```

3. **Log the Model with MLflow**

   When you log the model with MLflow, ensure that the `Config` (which includes your custom hyperparameters) is saved. Depending on your logging approach (using `Pipeline` or custom PyFunc), this is handled automatically.

   **Using a `Pipeline`**:

   ```python
   from transformers import pipeline
   import mlflow.transformers

   nli_pipeline = pipeline(
       task="text-classification",
       model=ft_model,
       tokenizer=tokenizer
   )

   with mlflow.start_run(run_name="log_transformers_pipeline_with_config"):
       # Log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": 3,
           "learning_rate": 2e-5,
           "weight_decay": 0.01,
           "per_device_train_batch_size": 16
       })
       mlflow.log_metric("accuracy", 0.95)

       # Log the Pipeline
       mlflow.transformers.log_model(
           transformers_model=nli_pipeline,
           artifact_path="transformers_pipeline",
           registered_model_name="Your_Registered_Model_Name"
       )

       print("Pipeline with both model and tokenizer logged successfully!")
   ```

   **Using a Custom PyFunc Model**:

   Ensure that when saving the model and tokenizer, the config with custom hyperparameters is included.

   ```python
   import mlflow.pyfunc
   import tempfile
   import os

   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

           # Load the config to access custom hyperparameters
           config_path = context.artifacts["config_path"]
           self.config = AutoConfig.from_pretrained(config_path)

       def predict(self, context, model_input):
           # Access custom hyperparameters from config if needed
           dropout_rate = getattr(self.config, "dropout_rate", None)
           optimizer = getattr(self.config, "optimizer", None)
           scheduler = getattr(self.config, "scheduler", None)

           # Example usage of custom hyperparameters
           # (Modify as per your inference logic)
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()

   with mlflow.start_run(run_name="log_custom_pyfunc_model_with_config"):
       # Log parameters and metrics
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
           config_path = os.path.join(tmp_dir, "config")

           # Save the model, tokenizer, and config to the temporary directory
           ft_model.save_pretrained(model_path)
           tokenizer.save_pretrained(tokenizer_path)
           config.save_pretrained(config_path)

           # Define artifacts dictionary
           artifacts = {
               "model_path": model_path,
               "tokenizer_path": tokenizer_path,
               "config_path": config_path
           }

           # Log the PyFunc model
           mlflow.pyfunc.log_model(
               artifact_path="custom_pyfunc_nli_model_with_config",
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

       print("Custom PyFunc model with both model, tokenizer, and config logged successfully!")
   ```

4. **Retrieving Custom Hyperparameters During Inference**

   When you load the model for inference, you can access the custom hyperparameters from the `Config` as demonstrated in the custom PyFunc model's `predict` method.

   ```python
   import mlflow.pyfunc

   # Load the production model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )

   # Example inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices
   ```

   Within the `PyFuncNLIModel`, you can utilize the custom hyperparameters as needed during inference.

---

### **Best Practices for Storing Custom Hyperparameters**

1. **Use Meaningful Attribute Names**: Ensure that the custom hyperparameters you add to the `Config` have clear and descriptive names to avoid confusion with existing attributes.

2. **Maintain Consistency**: If you plan to use these custom hyperparameters across different models or projects, maintain a consistent naming convention.

3. **Version Control**: When updating or changing custom hyperparameters, consider versioning your models to track these changes over time.

4. **Validation**: Implement checks within your model loading and inference logic to handle scenarios where custom hyperparameters might be missing or have unexpected values.

---

## **Summary**

- **Logging Both Model and Tokenizer**: Utilize either a Hugging Face `Pipeline` or a custom PyFunc model to encapsulate and log both the model and tokenizer together in MLflow. Avoid passing a dictionary directly to `mlflow.transformers.log_model`, as it expects a single model instance or a `Pipeline`.

- **Storing Custom Hyperparameters**: Leverage the flexibility of Hugging Face's `Config` objects to store additional hyperparameters by setting them as attributes. Ensure these are saved alongside the model and tokenizer, and retrieve them during inference as needed.

By following these approaches, you can effectively manage both your model artifacts and associated configuration parameters within MLflow, ensuring a streamlined and reproducible workflow.

---

**Feel free to reach out if you have further questions or need additional assistance!**

Absolutely, you can store custom hyperparameters within the Hugging Face `Config` object during fine-tuning. This approach ensures that your hyperparameters are **embedded** within the model's configuration, facilitating easy retrieval and enhancing reproducibility.

Below, I'll guide you through the process of:

1. **Adding Custom Hyperparameters to the Config**
2. **Saving the Config with the Model**
3. **Logging the Model (with Config) to MLflow**
4. **Loading and Accessing the Custom Hyperparameters**

Let's dive into each step in detail.

---

## **1. Adding Custom Hyperparameters to the Config**

The Hugging Face `Config` objects (`AutoConfig`, `PretrainedConfig`, etc.) are designed to store configuration parameters for models. They are flexible and allow the addition of **custom attributes** beyond the predefined ones. This capability is crucial for embedding hyperparameters specific to your fine-tuning process.

### **Step-by-Step Implementation**

1. **Initialize the Config with Standard Parameters**

   You already have the following code to load the config:

   ```python
   from transformers import AutoConfig

   config = AutoConfig.from_pretrained(
       model_args.config_name if model_args.config_name else model_args.model_name_or_path,
       num_labels=num_labels,
       finetuning_task=data_args.task_name,
       cache_dir=model_args.cache_dir,
       revision=model_args.model_revision,
       trust_remote_code=model_args.trust_remote_code,
   )
   ```

2. **Define Your Custom Hyperparameters**

   Suppose you have the following custom hyperparameters:

   ```python
   custom_hyperparameters = {
       "dropout_rate": 0.3,
       "optimizer": "adamw",
       "scheduler": "linear",
       "gradient_accumulation_steps": 4
   }
   ```

3. **Add Custom Hyperparameters to the Config**

   You can add these hyperparameters as attributes to the `config` object using Python's `setattr` function:

   ```python
   for key, value in custom_hyperparameters.items():
       setattr(config, key, value)
   ```

   **Complete Example:**

   ```python
   from transformers import AutoConfig

   # Load the config
   config = AutoConfig.from_pretrained(
       model_args.config_name if model_args.config_name else model_args.model_name_or_path,
       num_labels=num_labels,
       finetuning_task=data_args.task_name,
       cache_dir=model_args.cache_dir,
       revision=model_args.model_revision,
       trust_remote_code=model_args.trust_remote_code,
   )

   # Define custom hyperparameters
   custom_hyperparameters = {
       "dropout_rate": 0.3,
       "optimizer": "adamw",
       "scheduler": "linear",
       "gradient_accumulation_steps": 4
   }

   # Add custom hyperparameters to the config
   for key, value in custom_hyperparameters.items():
       setattr(config, key, value)
   ```

   **Explanation:**

   - **`setattr(config, key, value)`**: Dynamically sets the attribute `key` of the `config` object to `value`.
   - These attributes are now part of the `config` and will be saved alongside the model.

---

## **2. Saving the Config with the Model**

After fine-tuning your model, you need to save both the model and its configuration (which now includes your custom hyperparameters).

### **Step-by-Step Implementation**

1. **Fine-Tune Your Model**

   Ensure you have your fine-tuned model instance. For example:

   ```python
   from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

   model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

   training_args = TrainingArguments(
       output_dir="./results",
       num_train_epochs=config.num_train_epochs,  # Example usage of config hyperparameters
       per_device_train_batch_size=config.per_device_train_batch_size,
       # ... other training arguments
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       # ... other trainer parameters
   )

   trainer.train()
   ```

2. **Save the Model and Config**

   Use the `save_pretrained` method to save both the model and its config. This method ensures that the `config.json` (including custom hyperparameters) is saved alongside the model weights.

   ```python
   output_dir = "./fine_tuned_model"

   # Save the model and config
   model.save_pretrained(output_dir)
   config.save_pretrained(output_dir)  # This step is optional as `save_pretrained` on the model also saves the config
   ```

   **Note:** Calling `model.save_pretrained(output_dir)` automatically saves the `config.json`. However, explicitly saving the config ensures that any modifications are correctly captured.

---

## **3. Logging the Model (with Config) to MLflow**

Now that your model and config (with custom hyperparameters) are saved, you can log them to MLflow for tracking and deployment.

### **Using a Hugging Face `Pipeline`**

As previously discussed, using a `Pipeline` is a straightforward method to encapsulate the model and tokenizer.

1. **Create a Pipeline**

   ```python
   from transformers import pipeline

   nli_pipeline = pipeline(
       task="text-classification",
       model=model,
       tokenizer=tokenizer
   )
   ```

2. **Log the Pipeline with MLflow**

   ```python
   import mlflow
   import mlflow.transformers

   with mlflow.start_run(run_name="log_transformers_pipeline_with_custom_config"):
       # Log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": config.num_train_epochs,
           "learning_rate": config.learning_rate,
           "weight_decay": config.weight_decay,
           "per_device_train_batch_size": config.per_device_train_batch_size,
           "dropout_rate": config.dropout_rate,  # Custom hyperparameter
           "optimizer": config.optimizer,        # Custom hyperparameter
           "scheduler": config.scheduler,        # Custom hyperparameter
           "gradient_accumulation_steps": config.gradient_accumulation_steps  # Custom hyperparameter
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

       # Log the Transformers Pipeline
       mlflow.transformers.log_model(
           transformers_model=nli_pipeline,
           artifact_path="transformers_pipeline",
           registered_model_name="Your_Registered_Model_Name"  # Replace accordingly
       )

       print("Pipeline with both model and tokenizer (including custom hyperparameters) logged successfully!")
   ```

   **Explanation:**

   - **`mlflow.log_params`**: Logs both standard and custom hyperparameters.
   - **`mlflow.transformers.log_model`**: Logs the entire pipeline, which includes the model, tokenizer, and config.

### **Using a Custom PyFunc Model**

If you prefer more control, especially if you're not using a `Pipeline`, you can create a custom PyFunc model that handles both the model and tokenizer.

1. **Define a Custom PyFunc Model Class**

   ```python
   import mlflow.pyfunc
   from transformers import AutoModelForSequenceClassification, AutoTokenizer
   import torch

   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

           # Load the config to access custom hyperparameters
           config_path = context.artifacts["config_path"]
           self.config = AutoConfig.from_pretrained(config_path)

       def predict(self, context, model_input):
           # Access custom hyperparameters from config if needed
           dropout_rate = getattr(self.config, "dropout_rate", None)
           optimizer = getattr(self.config, "optimizer", None)
           scheduler = getattr(self.config, "scheduler", None)
           gradient_accumulation_steps = getattr(self.config, "gradient_accumulation_steps", None)

           # Example usage of custom hyperparameters (if applicable)
           # Modify as per your inference logic
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1)
           return predictions.numpy()
   ```

2. **Log the Custom PyFunc Model with Both Model and Tokenizer as Artifacts**

   ```python
   import mlflow.pyfunc
   import tempfile
   import os

   with mlflow.start_run(run_name="log_custom_pyfunc_model_with_custom_config"):
       # Log parameters and metrics
       mlflow.log_params({
           "num_train_epochs": config.num_train_epochs,
           "learning_rate": config.learning_rate,
           "weight_decay": config.weight_decay,
           "per_device_train_batch_size": config.per_device_train_batch_size,
           "dropout_rate": config.dropout_rate,  # Custom hyperparameter
           "optimizer": config.optimizer,        # Custom hyperparameter
           "scheduler": config.scheduler,        # Custom hyperparameter
           "gradient_accumulation_steps": config.gradient_accumulation_steps  # Custom hyperparameter
       })
       mlflow.log_metric("accuracy", 0.95)  # Replace with actual metrics

       # Use temporary directories to avoid saving to disk
       with tempfile.TemporaryDirectory() as tmp_dir:
           model_path = os.path.join(tmp_dir, "model")
           tokenizer_path = os.path.join(tmp_dir, "tokenizer")
           config_path = os.path.join(tmp_dir, "config")

           # Save the model, tokenizer, and config to the temporary directory
           model.save_pretrained(model_path)
           tokenizer.save_pretrained(tokenizer_path)
           config.save_pretrained(config_path)

           # Define artifacts dictionary
           artifacts = {
               "model_path": model_path,
               "tokenizer_path": tokenizer_path,
               "config_path": config_path
           }

           # Log the PyFunc model
           mlflow.pyfunc.log_model(
               artifact_path="custom_pyfunc_nli_model_with_custom_config",
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
               registered_model_name="Your_Registered_Model_Name"  # Replace accordingly
           )

       print("Custom PyFunc model with both model, tokenizer, and custom config logged successfully!")
   ```

   **Explanation:**

   - **Temporary Directories**: Using `tempfile.TemporaryDirectory()` ensures that the model, tokenizer, and config are saved temporarily and do not persist on disk.
   - **Artifacts Dictionary**: Specifies the paths to the model, tokenizer, and config, making them accessible within the custom PyFunc model.
   - **`conda_env`**: Defines the environment dependencies required for loading and running the model.
   - **`registered_model_name`**: Replace `"Your_Registered_Model_Name"` with your desired model registry name in MLflow.

---

## **4. Loading and Accessing the Custom Hyperparameters**

After logging, you can load your model and access the custom hyperparameters during inference.

### **Using a Hugging Face `Pipeline`**

1. **Load the Pipeline**

   ```python
   import mlflow.transformers

   # Load the production pipeline
   production_pipeline = mlflow.transformers.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )
   ```

2. **Perform Inference**

   ```python
   input_text = "I love machine learning!"
   outputs = production_pipeline(input_text)
   print(f"Predicted class: {outputs[0]['label']} with score {outputs[0]['score']}")
   ```

   **Note:** Accessing custom hyperparameters directly isn't straightforward with `Pipeline`. If you need to access them, consider using the Custom PyFunc approach.

### **Using a Custom PyFunc Model**

1. **Load the Custom PyFunc Model**

   ```python
   import mlflow.pyfunc

   # Load the production custom PyFunc model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"  # Adjust as needed
   )
   ```

2. **Perform Inference and Access Custom Hyperparameters**

   ```python
   input_texts = ["I love machine learning!", "This product is terrible."]
   predictions = production_pyfunc_model.predict(input_texts)
   print(predictions)  # Outputs: array of predicted class indices

   # Accessing custom hyperparameters
   # (Assuming you modify the predict method to return them or expose them via another method)
   # For example, you can modify the PyFunc class to return a dictionary containing predictions and hyperparameters
   ```

   **Enhancing the PyFunc Model to Access Hyperparameters:**

   If you need to access custom hyperparameters alongside predictions, consider modifying the `predict` method:

   ```python
   class PyFuncNLIModel(mlflow.pyfunc.PythonModel):
       def load_context(self, context):
           import os
           from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

           # Load the model
           model_path = context.artifacts["model_path"]
           self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
           self.model.eval()  # Set to evaluation mode

           # Load the tokenizer
           tokenizer_path = context.artifacts["tokenizer_path"]
           self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

           # Load the config to access custom hyperparameters
           config_path = context.artifacts["config_path"]
           self.config = AutoConfig.from_pretrained(config_path)

       def predict(self, context, model_input):
           # Access custom hyperparameters from config if needed
           custom_params = {
               "dropout_rate": getattr(self.config, "dropout_rate", None),
               "optimizer": getattr(self.config, "optimizer", None),
               "scheduler": getattr(self.config, "scheduler", None),
               "gradient_accumulation_steps": getattr(self.config, "gradient_accumulation_steps", None)
           }

           # Perform inference
           inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True)
           with torch.no_grad():
               outputs = self.model(**inputs)
           predictions = torch.argmax(outputs.logits, dim=1).numpy()

           # Combine predictions with custom hyperparameters (if desired)
           return {
               "predictions": predictions,
               "custom_hyperparameters": custom_params
           }
   ```

   **Updated Inference Example:**

   ```python
   # Load the custom PyFunc model
   production_pyfunc_model = mlflow.pyfunc.load_model(
       model_uri="models:/Your_Registered_Model_Name/Production"
   )

   # Perform inference
   input_texts = ["I love machine learning!", "This product is terrible."]
   result = production_pyfunc_model.predict(input_texts)
   
   print("Predictions:", result["predictions"])
   print("Custom Hyperparameters:", result["custom_hyperparameters"])
   ```

   **Explanation:**

   - **`custom_params`**: Extracts custom hyperparameters from the config.
   - **Return Structure**: Returns a dictionary containing both predictions and custom hyperparameters.

   **Note:** Adjust the `predict` method based on your specific needs and how you intend to use the hyperparameters during inference.

---

## **Reviewing the Hugging Face `AutoConfig` Module**

You've provided the `AutoConfig` module code, and it's essential to confirm that it supports adding and saving custom hyperparameters. Here's a brief overview based on the provided code:

1. **Flexibility for Custom Attributes:**

   The `PretrainedConfig` class (from which all specific configs inherit) is designed to handle arbitrary additional attributes. This is facilitated through its `from_dict` and `to_dict` methods, which serialize all attributes, including custom ones.

2. **Adding Custom Attributes:**

   As demonstrated earlier, you can add custom hyperparameters by setting them as attributes on the config object. These will be serialized when saving the config.

3. **Saving and Loading Configs:**

   The `save_pretrained` and `from_pretrained` methods handle serialization and deserialization of the config, ensuring that custom attributes are preserved.

   **Example:**

   ```python
   # Adding custom hyperparameters
   setattr(config, "custom_param1", value1)
   setattr(config, "custom_param2", value2)

   # Saving the config
   config.save_pretrained(output_dir)

   # Loading the config
   loaded_config = AutoConfig.from_pretrained(output_dir)

   # Accessing custom hyperparameters
   print(loaded_config.custom_param1)
   print(loaded_config.custom_param2)
   ```

   **Explanation:**

   - **Serialization**: Custom attributes are saved in the `config.json` file within `output_dir`.
   - **Deserialization**: When loading, these attributes are available as part of the `loaded_config` object.

---

## **Best Practices and Recommendations**

1. **Use Meaningful Names for Custom Hyperparameters:**

   Ensure that the names of your custom hyperparameters do not clash with existing config attributes. Prefixing them or using descriptive names can help avoid conflicts.

   ```python
   custom_hyperparameters = {
       "custom_dropout_rate": 0.3,
       "custom_optimizer": "adamw",
       # ...
   }
   ```

2. **Validate Custom Hyperparameters:**

   Implement validation checks to ensure that custom hyperparameters have acceptable values, especially if they're critical to the model's performance.

   ```python
   if not 0.0 < config.custom_dropout_rate < 1.0:
       raise ValueError("dropout_rate must be between 0 and 1")
   ```

3. **Document Custom Hyperparameters:**

   Maintain clear documentation of the custom hyperparameters added to the config. This practice aids in maintaining clarity, especially when collaborating with others or revisiting the project after some time.

4. **Version Control:**

   When changing or adding new custom hyperparameters, consider versioning your models to track these changes. This approach enhances reproducibility and traceability.

5. **Consistent Usage Across Projects:**

   If you plan to use these custom hyperparameters across multiple projects or models, establish a consistent naming convention to maintain uniformity.

6. **Leverage MLflow's Capabilities:**

   MLflow excels in tracking experiments, parameters, and artifacts. Ensure that you log both standard and custom hyperparameters using `mlflow.log_params` to benefit from MLflow's tracking and visualization features.

   ```python
   mlflow.log_params(custom_hyperparameters)
   ```

---

## **Complete Example: Putting It All Together**

Here's a comprehensive example that integrates all the steps discussed:

```python
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, pipeline
import mlflow
import mlflow.transformers
import tempfile
import os

# 1. Define model arguments (replace with your actual arguments)
class ModelArgs:
    def __init__(self):
        self.config_name = None
        self.model_name_or_path = "microsoft/deberta-v3-base"
        self.cache_dir = "./cache"
        self.model_revision = "main"
        self.trust_remote_code = False

model_args = ModelArgs()

# 2. Define data arguments (replace with your actual arguments)
class DataArgs:
    def __init__(self):
        self.task_name = "nli"

data_args = DataArgs()

# 3. Define other necessary variables
num_labels = 3  # Example: for a 3-class classification
custom_hyperparameters = {
    "dropout_rate": 0.3,
    "optimizer": "adamw",
    "scheduler": "linear",
    "gradient_accumulation_steps": 4
}

# 4. Load and customize the config
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    trust_remote_code=model_args.trust_remote_code,
)

# Add custom hyperparameters
for key, value in custom_hyperparameters.items():
    setattr(config, key, value)

# 5. Initialize the model with the customized config
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config
)

# 6. Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    trust_remote_code=model_args.trust_remote_code,
)

# 7. Define training arguments and trainer (simplified)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    num_train_epochs=config.num_train_epochs if hasattr(config, "num_train_epochs") else 3,
    learning_rate=config.learning_rate if hasattr(config, "learning_rate") else 2e-5,
    per_device_train_batch_size=config.per_device_train_batch_size if hasattr(config, "per_device_train_batch_size") else 16,
    # ... other training arguments
)

# Assume you have train_dataset and eval_dataset defined
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     # ... other trainer parameters
# )

# # Train the model
# trainer.train()

# For demonstration, we'll skip actual training

# 8. Save the model and config
output_dir = "./fine_tuned_model"

model.save_pretrained(output_dir)
config.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# 9. Create a Hugging Face Pipeline
nli_pipeline = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer
)

# 10. Log to MLflow
with mlflow.start_run(run_name="log_fine_tuned_nli_model_with_custom_config"):
    # Log parameters and custom hyperparameters
    mlflow.log_params(custom_hyperparameters)
    
    # Log additional parameters
    mlflow.log_params({
        "num_train_epochs": config.num_train_epochs if hasattr(config, "num_train_epochs") else 3,
        "learning_rate": config.learning_rate if hasattr(config, "learning_rate") else 2e-5,
        "weight_decay": config.weight_decay if hasattr(config, "weight_decay") else 0.01,
        "per_device_train_batch_size": config.per_device_train_batch_size if hasattr(config, "per_device_train_batch_size") else 16,
    })
    
    # Log metrics (replace with actual metrics)
    mlflow.log_metric("accuracy", 0.95)
    
    # Log the Transformers Pipeline
    mlflow.transformers.log_model(
        transformers_model=nli_pipeline,
        artifact_path="transformers_pipeline",
        registered_model_name="FineTuned_DeBERTa_NLI"
    )
    
    print("Fine-tuned model with custom hyperparameters logged successfully!")
```

**Explanation:**

- **Custom Hyperparameters**: Added to the `config` using `setattr`.
- **Saving**: Both model and config are saved to `output_dir`.
- **Pipeline**: Created using the fine-tuned model and tokenizer.
- **MLflow Logging**: Parameters (both standard and custom), metrics, and the pipeline are logged to MLflow.
- **Registered Model Name**: Replace `"FineTuned_DeBERTa_NLI"` with your desired name in the Model Registry.

---

## **Accessing Custom Hyperparameters During Inference**

When you load the model for inference, you can access the custom hyperparameters from the `Config` object. Here's how to do it using the Custom PyFunc model approach:

1. **Load the Custom PyFunc Model**

   ```python
   import mlflow.pyfunc

   # Replace with your actual registered model name and stage
   model_uri = "models:/FineTuned_DeBERTa_NLI/Production"

   production_pyfunc_model = mlflow.pyfunc.load_model(model_uri)
   ```

2. **Perform Inference and Access Hyperparameters**

   ```python
   input_texts = ["I love machine learning!", "This product is terrible."]
   result = production_pyfunc_model.predict(input_texts)
   
   print("Predictions:", result["predictions"])
   print("Custom Hyperparameters:", result["custom_hyperparameters"])
   ```

   **Sample Output:**

   ```
   Predictions: [1 0]
   Custom Hyperparameters: {
       "dropout_rate": 0.3,
       "optimizer": "adamw",
       "scheduler": "linear",
       "gradient_accumulation_steps": 4
   }
   ```

   **Note:** Ensure that your `predict` method in the `PyFuncNLIModel` class is designed to return both predictions and custom hyperparameters as shown in the earlier example.

---

## **Conclusion**

Yes, it is entirely possible to store custom hyperparameters within the Hugging Face `Config` object during fine-tuning. By adding them as attributes to the config, saving the config with the model, and logging the model to MLflow (either via a `Pipeline` or a custom PyFunc model), you ensure that these hyperparameters are preserved and accessible for future use.

**Key Takeaways:**

- **Flexibility**: Hugging Face `Config` objects support custom attributes, allowing you to embed any additional information.
- **Persistence**: Saving the config alongside the model ensures that custom hyperparameters are retained.
- **MLflow Integration**: Logging models (with configs) to MLflow facilitates experiment tracking and model deployment.
- **Access During Inference**: Custom PyFunc models provide a structured way to access both predictions and embedded hyperparameters during inference.

By following the steps outlined above, you can enhance the traceability and reproducibility of your fine-tuning experiments.

If you have any further questions or need additional assistance, feel free to ask!