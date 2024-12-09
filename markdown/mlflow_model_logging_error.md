The error suggests that the model wrapped by your `accelerator` (presumably from `Accelerate` library) with automatic mixed precision cannot be directly serialized for logging via MLflow. Let's break this down:

### Problem Diagnosis:
1. **`trained_model` Type**:
   - The model is of type `AutoModelForSequenceClassification` (Hugging Face Transformers).
   - When you train this model using `Accelerator`, it may be wrapped with data-parallel wrappers or mixed-precision settings that are not serializable.

2. **`Accelerator.unwrap_model()`**:
   - This method is correctly used to retrieve the original model from the wrapper. However, there could still be attributes (like optimizer states or mixed-precision attributes) causing serialization issues.

3. **Serialization with `mlflow.pytorch.log_model()`**:
   - This method attempts to save the model using PyTorch's `torch.save()`. If the model or its attributes include non-pickleable objects, the operation fails.

### Debugging Steps:
Here are a few things to try:

#### Step 1: Validate `unwrapped_model` After `unwrap_model`
- Add debug logs to inspect the type and structure of `unwrapped_model`:
  ```python
  logger.info(f"Unwrapped model type: {type(unwrapped_model)}")
  logger.info(f"Unwrapped model state dict keys: {unwrapped_model.state_dict().keys()}")
  ```

#### Step 2: Ensure Full Precision
- Automatic mixed precision (`AMP`) stores some states in FP16, which might cause issues. Convert the model to full precision using `.float()` before logging.
  ```python
  unwrapped_model = unwrapped_model.float()
  ```

#### Step 3: Check if Mixed Precision States are Persisting
- Even after calling `unwrap_model` and converting to `.float()`, confirm whether any AMP-related attributes remain. Inspect the model for any attributes like `self._amp` or others:
  ```python
  logger.info(f"Model attributes: {dir(unwrapped_model)}")
  ```

#### Step 4: Test Model Serialization Locally
- Before logging to MLflow, attempt a local serialization to ensure the model is pickleable:
  ```python
  import torch

  try:
      torch.save(unwrapped_model.state_dict(), "test_model.pth")
      logger.info("Local serialization successful")
  except Exception as e:
      logger.error(f"Local serialization failed: {e}")
  ```

#### Step 5: Use MLflow Model Wrapper
- Ensure the `mlflow.pytorch.log_model()` is invoked with a clean model. You might try saving the state dictionary and then reloading it into a fresh instance of the `AutoModelForSequenceClassification`:
  ```python
  from transformers import AutoModelForSequenceClassification

  # Save and reload to create a fresh instance
  model_path = "temp_model.pth"
  torch.save(unwrapped_model.state_dict(), model_path)
  
  fresh_model = AutoModelForSequenceClassification.from_pretrained(
      pretrained_model_name_or_path=None,  # Optional if loading custom model
      state_dict=torch.load(model_path)
  )
  
  # Log with MLflow
  try:
      mlflow.pytorch.log_model(fresh_model, "model")
      logger.info("Model logged successfully with MLflow")
  except Exception as e:
      logger.error(f"Failed to log model with MLflow: {e}")
  ```

### Common Fixes:
- **If AMP Causes Issues**: Ensure the model is converted to `.float()` before saving.
- **Wrapper Issues**: Make sure `unwrap_model` is called only once on the `trained_model`.
- **State Mismatch**: If using `state_dict()`, validate that the loaded `state_dict()` matches the `AutoModelForSequenceClassification` architecture.

#### Final Updated Code:
```python
unwrapped_model = self.accelerator.unwrap_model(trained_model)

if hasattr(unwrapped_model, 'half'):
    unwrapped_model = unwrapped_model.float()

# Test local serialization
try:
    torch.save(unwrapped_model.state_dict(), "test_model.pth")
    logger.info("Local serialization successful")
except Exception as e:
    logger.error(f"Local serialization failed: {e}")

# Reload into fresh model instance
from transformers import AutoModelForSequenceClassification
fresh_model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=None,
    state_dict=torch.load("test_model.pth")
)

# Log with MLflow
try:
    mlflow.pytorch.log_model(fresh_model, "model")
    logger.info("Model logged successfully")
except Exception as e:
    logger.error(f"Failed to log model: {e}")
```

Let me know if this resolves your issue or if further debugging is needed!


### load prod model

```

mlflow==2.18.0


production_model = mlflow.transformers.load_model(
    model_uri="models:/prod_model/Production"
)
nli_pipeline = pipeline("zero-shot-classification", model=production_model.model, tokenizer=production_model.tokenizer, device= 0 if torch.cuda.is_available() else -1)
```