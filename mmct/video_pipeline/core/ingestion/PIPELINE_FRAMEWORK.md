# Pipeline Framework Implementation

## Overview

This document describes the new step-based pipeline framework for video ingestion, following the design pattern from the GitHub reference repo.

## Directory Structure

```
ingestion/
├── ingestion_pipeline.py          # Main pipeline class (to be updated)
├── languages.py                    # Shared language enums
├── models.py                       # Shared data models
├── utils/                          # Shared utilities
│
├── pipelines/                      # Pipeline Framework
│   ├── __init__.py                 # Main exports
│   ├── config.py                   # YAML configuration loader
│   ├── runner.py                   # Pipeline orchestrator
│   │
│   └── steps/                      # All pipeline steps
│       ├── __init__.py             # Exports base classes
│       ├── base.py                 # PipelineStep, StepContext, StepResult
│       ├── registry.py             # Step registration system
│       ├── data_store.py           # Inter-step communication
│       ├── builtins.py             # Registers all built-in steps
│       │
│       ├── Simple Steps (standalone files)
│       │   ├── early_check_step.py
│       │   ├── validate_audio_step.py
│       │   └── transcription_step.py
│       │
│       └── Complex Steps (folders with components)
│           ├── compress/           # Video compression
│           ├── keyframes/          # Keyframe extraction
│           ├── chapters/           # Chapter generation
│           │   └── semantic_chunking/
│           ├── embeddings/         # Embedding generation
│           ├── upload/             # Upload and indexing
│           └── cleanup/            # Cleanup
│
├── experiments/                    # Pipeline configurations
│   └── default_ingestion.yaml
│
└── scripts/                        # Test and run scripts
    ├── __init__.py
    └── test_framework.py
```

## Key Design Patterns

### 1. **Step-Based Architecture**
Each processing stage is an independent step implementing the `PipelineStep` interface:

```python
from mmct.video_pipeline.core.ingestion.pipelines.steps import PipelineStep, StepContext, StepResult, register_step

@register_step("ingestion.my_step")
class MyStep(PipelineStep):
    step_type = "ingestion.my_step"
    description = "Description of what this step does"

    async def run(self, context: StepContext) -> StepResult:
        # Access providers from context
        provider = context.provider.some_provider

        # Get parameters (YAML params or user runtime params)
        param_value = self.get_param("param_name", context, default="default_value")

        # Get data from previous steps
        previous_output = context.data_store.get("previous_step_id", "output_key")

        # Do work...
        result = do_something()

        # Return result
        return StepResult(
            step_id=self.step_id,
            outputs={"output_key": result},
            metrics={"processing_time": 1.5},
            artifacts=["/path/to/file.json"]
        )
```

### 2. **Self-Registering Steps**
Steps register themselves using the `@register_step` decorator. No manual registry maintenance needed.

### 3. **Configuration-Driven**
Pipeline topology defined in YAML (`experiments/default_ingestion.yaml`):

```yaml
pipeline:
  name: "default_ingestion"
  mode: "sequential"

  steps:
    - id: "compress"
      type: "ingestion.compress"
      params:
        max_size_mb: 500

    - id: "keyframes"
      type: "ingestion.keyframes"
      params:
        source_step: "compress"  # References previous step
```

### 4. **Provider-Based**
Providers (LLM, embedding, storage, etc.) are passed via `StepContext`, not stored in YAML:

```python
# Steps access providers from context
llm_provider = context.provider.llm_provider
storage_provider = context.provider.storage_provider
```

### 5. **Inter-Step Communication**
Steps communicate via `StepDataStore`:

```python
# Step 1: Store data
context.data_store.set("compress", "video_path", compressed_path)

# Step 2: Retrieve data
video_path = context.data_store.get("compress", "video_path")
```

## Registered Steps

| Step Type | Description |
|-----------|-------------|
| `ingestion.early_check` | Check if video already ingested |
| `ingestion.validate_audio` | Validate audio stream exists |
| `ingestion.compress` | Compress video if needed |
| `ingestion.keyframes` | Extract keyframes with timestamps |
| `ingestion.transcribe` | Transcribe or use external transcript |
| `ingestion.chapters` | Generate semantic chapters |
| `ingestion.embeddings` | Generate embeddings (parallel) |
| `ingestion.upload` | Upload files and index data |
| `ingestion.cleanup` | Clean up temporary files |

## How to Test

### Run the test script:

```bash
# From project root
cd /home/v-amanpatkar/work/latest_v4/MMCTAgent

# Run standalone test
python test_ingestion_pipeline.py
```

This will verify:
1. All steps are registered correctly
2. Configuration loads properly
3. Data store works
4. Pipeline structure is valid

## How to Use

### Basic Usage

```python
from mmct.video_pipeline.core.ingestion.pipelines import (
    PipelineRunner,
    load_pipeline_config,
    StepContext,
    StepDataStore,
)
from mmct.config.providers import IngestionProviderConfig

# Load pipeline configuration
config = load_pipeline_config("experiments/default_ingestion.yaml")

# Create context with providers and parameters
context = StepContext(
    video_path="/path/to/video.mp4",
    provider=your_provider_config,  # IngestionProviderConfig
    language=Languages.ENGLISH,
    url="https://example.com/video",
    transcript_path=None,
    output_dir="/tmp/output",
    video_id="video123",
    parent_id="video123",
    parent_duration=100.0,
    video_duration=100.0,
    data_store=StepDataStore(),
    logger=logger,
    user_params={
        "keyframe_config": {"motion_threshold": 1.5, "sample_fps": 2},
        "frame_stacking_grid_size": 4,
    }
)

# Run pipeline
runner = PipelineRunner(pipeline_config=config, context=context)
report = await runner.run()

# Check results
print(f"Pipeline: {report.status}")
print(f"Duration: {report.total_duration_seconds:.2f}s")
```

### Custom Pipeline Configuration

Create a new YAML file in `experiments/`:

```yaml
# experiments/quick_ingestion.yaml
pipeline:
  name: "quick_ingestion"
  mode: "sequential"

  steps:
    - id: "transcribe"
      type: "ingestion.transcribe"
      params:
        video_step: "input"

    - id: "chapters"
      type: "ingestion.chapters"
      params:
        transcript_step: "transcribe"
```

Then load and run:

```python
config = load_pipeline_config("experiments/quick_ingestion.yaml")
runner = PipelineRunner(pipeline_config=config, context=context)
report = await runner.run()
```

## Next Steps

1. **Test the Framework**
   - Run `python test_ingestion_pipeline.py`
   - Verify all steps are registered
   - Check configuration loads

2. **Update IngestionPipeline Class**
   - Modify `ingestion_pipeline.py` to use the new framework
   - Keep the same user-facing API
   - Use PipelineRunner internally

3. **Integration Testing**
   - Test with a real video
   - Verify all steps work end-to-end
   - Check that providers are correctly passed

4. **Add Custom Steps** (if needed)
   - Create new step classes
   - Register with `@register_step`
   - Add to YAML configuration

## Benefits

✅ **Modular**: Each step is independent and testable
✅ **Flexible**: Easy to add/remove/reorder steps via YAML
✅ **Maintainable**: Clear separation of concerns
✅ **Reusable**: Steps can be used in different pipelines
✅ **Configurable**: YAML configs for different scenarios
✅ **Type-Safe**: Proper type hints throughout
✅ **Resume-Capable**: Pipeline can resume from failures
✅ **Observable**: Detailed metrics and reporting

## Differences from Old Design

| Aspect | Old Design | New Design |
|--------|------------|------------|
| Structure | Monolithic class | Modular steps |
| Configuration | Constructor params | YAML + runtime params |
| Steps | Private methods | Independent classes |
| Testing | Full pipeline required | Individual step testing |
| Extensibility | Modify class | Add new step class |
| Reusability | Low | High |

## Architecture Alignment

This implementation follows the design pattern from:
https://github.com/microsoft/MMCTAgent/tree/kchourasia/chapter-gen-exp/chapter-gen-exp

Key alignments:
- Steps with `base.py` and `registry.py` inside `steps/` directory
- YAML configurations in separate `experiments/` directory
- Entry points in `scripts/` directory
- Sequential execution with pluggable steps
- Configuration-driven pipeline topology
