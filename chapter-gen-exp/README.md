# Chapter Generation Experimentation Sandbox

This repository now includes a lightweight, fully-pluggable experimentation pipeline for transforming raw video transcripts and frame samples into chapter-centric knowledge packs. Each stage of the pipeline is swappable so you can benchmark different strategies (FPS vs scene-based frames, sequential vs LLM-backed chaptering, etc.) without rebuilding orchestration code.

## Layout

```
pipelines/
  config/           # YAML -> dataclass loaders
  runner.py         # Sequential orchestration + reporting
  steps/            # Step interfaces plus built-in samples
    frames/         # FPS sampler placeholder
    transcripts/    # Transcript cleaners/alignment steps
    chapters/       # Chapter generators (sequential today)
    export/         # Knowledge-pack writers
samples/transcripts/  # Demo transcript for quick testing
experiments/          # YAML experiment definitions
scripts/run_experiment.py  # CLI entry point
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiment.py --config experiments/sample_config.yaml --report outputs/demo/report.json
```

The sample config wires together these steps:
1. `video.chunk.basic` – emits chunk spans (currently a single chunk covering the video).
2. `video.chunk.align-transcript` – snaps chunk boundaries to transcript sentences and attaches the exact text assigned to each chunk.
3. `frames.fps` – produces virtual keyframes via constant FPS sampling inside each chunk.
4. `transcript.clean` – normalizes and filters transcript segments.
5. `chapters.sequential` – groups cleaned segments into greedy windows.
6. `export.knowledge-pack` – writes a JSON bundle with chapters, frames, and metadata.

Results land under `outputs/demo/` (configurable), including a `demo_knowledge_pack.json` ready for downstream embedding/indexing experiments. Swap step `type` values or add new modules under `pipelines/steps/**` to expand the pipeline without touching the runner.

Both frame samplers (`frames.fps` and `frames.optical-flow`) enforce `max_frames_per_chunk`, ensuring each chunk contributes at most a fixed number of keyframes while still respecting the global `max_frames` ceiling. Always point `chunks_step` at the transcript-aligned chunk step so frame extraction respects the sentence-safe boundaries.


Note: video duration is inferred directly from the source file during runtime, so you no longer need to supply `video_duration_seconds` in experiment configs.

Additional chunking / frame options available out of the box:
- `video.chunk.align-transcript` (sentence-safe chunk adjustment) – run this after any chunker to snap boundaries to transcript sentences and expose chunk-level transcript text.
- `video.chunk.scene` (PySceneDetect-based scene boundaries) feeding any chunk-aware frame sampler – see `experiments/scene_config.yaml` for usage with `frames.fps`.
- `frames.optical-flow` (motion-triggered keyframes) – see `experiments/optical_flow_config.yaml` (chunk-aware motion thresholds).
- `chapters.scene-llm` (frame + transcript aware LLM chapters) – see `experiments/scene_llm_stub_config.yaml` for a provider-backed GPT-4o example.
- `chapters.context-enrich` (sequential enrichment + optional object roster) – consumes the output of an earlier chaptering step, refines each summary/action using a sliding window of prior chapters, and can simultaneously maintain a deduplicated object roster without a second pipeline pass.
- `chapters.object-enrich` (standalone object roster) – optional legacy step that still performs only the global object consolidation if you need to run it separately.

### LLM-backed chapters

`chapters.scene-llm` consumes the transcript-aligned chunks plus their associated frames and dispatches up to five chapter requests in parallel. Each response must conform to the `ChapterCreationResponse` model defined in `chapter_generator/models.py`. The step accepts:

- `chunks_step` / `frames_step`: identifiers for upstream steps (`video.chunk.align-transcript` and any frame sampler).
- `batch_size` & `max_parallel_requests`: control concurrency (defaults: 5).
- `max_frames_per_chapter`: caps how many frames from each chunk are embedded in the prompt.
- `llm_provider`: optional override for `ProviderFactory.create_llm_provider`. If omitted, the factory uses the provider configured in `MMCTConfig` (defaults to GPT-4o in the packaged settings).
- `llm_request_options`: dictionary passed directly to the provider's `chat_completion` call (e.g., `temperature`, `max_tokens`).

To exercise the GPT-4o-backed flow, run:

```bash
python scripts/run_experiment.py --config experiments/scene_llm_stub_config.yaml --report outputs/scene-llm-demo/report.json
```
Ensure your global MMCT configuration points the provider factory at a GPT-4o deployment (Azure OpenAI by default) and add any custom temperatures or token limits via `llm_request_options`.

### Context-aware enrichment

`chapters.context-enrich` runs after any chapter-producing step (typically `chapters.scene-llm`). It walks the chapters in order and re-prompts the LLM with a limited history window so each enriched summary can reference key moments that happened slightly earlier in the video without exceeding context limits. When desired, the same pass can also reconcile a global object roster by enabling the `object_enrichment` block (see below), removing the need for a follow-up `chapters.object-enrich` step.

Parameters:

- `chapters_step`: upstream step id that produced the base chapters.
- `context_window`: number of previous chapters (default 3) to include in the contextual prompt.
- `llm_request_options`: forwarded to the underlying provider just like in `chapters.scene-llm` for temperature/token overrides.
- `object_enrichment`: optional nested configuration that enables inline object tracking. Accepts:
  - `enabled` (default `true` when block present).
  - `max_active_context`, `min_screen_time_seconds`, `min_chunk_occurrences`: identical semantics to the standalone object step.
  - `llm_request_options`: overrides for the object-tracker prompt (can differ from the chapter enrichment call).

Add it to a pipeline immediately after the base chapter generator:

```yaml
- id: llm_chapters
  type: chapters.scene-llm
  params:
    chunks_step: scene_chunk_alignment
    frames_step: of_scene_frames
- id: enriched_chapters
  type: chapters.context-enrich
  params:
    chapters_step: llm_chapters
    context_window: 4
    llm_request_options:
      temperature: 0.2
    object_enrichment:
      enabled: true
      max_active_context: 10
      min_screen_time_seconds: 8.0
      min_chunk_occurrences: 2
      llm_request_options:
        temperature: 0.1
```

Downstream exporters can then point at `enriched_chapters` to retrieve continuity-aware chapter text while still retaining the original summaries for reference.

### Standalone global object enrichment (optional)

`chapters.object-enrich` remains available if you want to run object consolidation in a dedicated step (for example, to reuse previously enriched chapters without re-running the chapter LLM). It feeds the LLM with the current chapter’s objects plus the list of “active” objects from earlier chapters and asks for structured add/update/remove operations, then applies those deltas to maintain a single coherent `object_collection` spanning the entire video.

Parameters:

- `chapters_step`: upstream step containing chapters with `object_collection` data.
- `max_active_context`: how many active objects to pass back to the LLM each turn (defaults to 12) to limit prompt size.
- `min_screen_time_seconds`: drop objects whose cumulative on-screen duration (based on chunk lengths) is below this threshold (defaults to 8 seconds).
- `min_chunk_occurrences`: minimum number of chunks an object must appear in to stay in the final roster (defaults to 2).
- `llm_request_options`: forwarded to the provider for temperature/token tweaks.

Example wiring (only needed when you skip the inline `object_enrichment` block):

```yaml
- id: enriched_chapters
  type: chapters.context-enrich
  params:
    chapters_step: llm_chapters
- id: object_roster
  type: chapters.object-enrich
  params:
    chapters_step: enriched_chapters
    max_active_context: 10
    llm_request_options:
      temperature: 0.1
```

Exporters (or downstream embedding steps) can now reference either `enriched_chapters` (inline mode) or `object_roster` (standalone mode) to get a single deduplicated view of the people/items present across the entire video with consolidated appearance and identity notes.
