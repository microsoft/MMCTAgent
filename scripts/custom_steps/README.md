# Custom Ingestion Steps

This directory demonstrates how client applications can extend the MMCT
ingestion pipeline with their own processing steps — without modifying
the core library.

## How It Works

MMCT's pipeline framework uses a **step registry**.  Any class decorated
with `@register_step("step.type")` is available for use in a pipeline
YAML configuration.  The decorator and base classes are part of the
public API:

```python
from mmct.video_pipeline import PipelineStep, StepContext, StepResult, register_step
```

### 1. Write a custom step

Create a Python module with your step class:

```python
# my_steps/sentiment.py
from mmct.video_pipeline import PipelineStep, StepContext, StepResult, register_step

@register_step("myapp.sentiment_analysis")
class SentimentAnalysisStep(PipelineStep):
    step_type = "myapp.sentiment_analysis"
    description = "Run sentiment analysis on transcript chunks."

    async def run(self, context: StepContext) -> StepResult:
        transcript = context.data_store.get("transcribe", "transcript")
        # ... analyse sentiment ...
        return StepResult(
            step_id=self.step_id,
            outputs={"sentiment_scores": scores},
        )
```

### 2. Reference it in a pipeline YAML

Add an entry to your pipeline config (copy the default YAML and extend):

```yaml
- id: "sentiment"
  type: "myapp.sentiment_analysis"
  params:
    model: "distilbert-base-uncased-finetuned-sst-2-english"
```

### 3. Import before running the pipeline

The `@register_step` decorator fires at **import time**, so you just
need to make sure your module is imported before `PipelineRunner`
resolves step types:

```python
import my_steps.sentiment  # registers the step

from mmct.video_pipeline import IngestionPipeline

ingestion = IngestionPipeline(
    video_path="video.mp4",
    video_id="abc123",
    provider=providers,
    language=language,
    pipeline_config_path="my_custom_pipeline.yaml",
)
report = await ingestion.run()
```

## Steps in This Directory

| Module | Step Type | Description |
|---|---|---|
| `uniform_frames.py` | `ingestion.uniform_frames` | Extract 1 fps frames via ffmpeg and upload to Azure Blob Storage |
| `transcript_upload.py` | `ingestion.transcript_upload` | Upload SRT transcript to Azure Blob Storage |

These steps were originally part of the core MMCT library but are
application-specific (they depend on Azure Blob Storage conventions).
They are provided here as working examples of the custom step pattern.

## Key Concepts

- **`StepContext`** — shared state passed to every step; includes
  `video_path`, `provider`, `data_store`, `logger`, and `user_params`.
- **`StepResult`** — returned by `run()`; contains `outputs` (stored in
  `data_store` for downstream steps), `metrics`, and `artifacts`.
- **`data_store`** — inter-step communication; use
  `context.data_store.get(step_id, key)` to read outputs from earlier
  steps and `StepResult.outputs` to publish your own.
- **`get_param(key, context, default)`** — resolves parameters with
  priority: runtime `user_params` > YAML `params` > default value.
