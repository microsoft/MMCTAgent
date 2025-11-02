
## Providers Guide

This page gives a quick, easy-to-read summary of the providers you can use in this repo, how to configure them, and short copy-paste examples. For the authoritative types and defaults see `mmct/config/settings.py`.

What you'll find here
- Quick start (2-min setup)
- Config snippets (env + minimal values)
- How to call providers in code
- Provider-specific notes (Azure search, Local FAISS)

---

## Quick start

1. Pick providers by setting environment variables in a `.env` file (examples below).
2. Install the package (editable) and use ProviderFactory in code to create the provider you want.
3. If you use `local_faiss`, point `index_path` at the directory containing `<index_name>.index` and `<index_name>.meta.json`.

Example .env for Azure-first setup

```
LLM_PROVIDER=azure
LLM_ENDPOINT=https://<your-openai-endpoint>
LLM_DEPLOYMENT_NAME=<deployment>
EMBEDDING_PROVIDER=azure
EMBEDDING_SERVICE_ENDPOINT=https://<your-openai-endpoint>
EMBEDDING_SERVICE_DEPLOYMENT_NAME=<embedding-deployment>
SEARCH_PROVIDER=azure_ai_search
SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
SEARCH_USE_MANAGED_IDENTITY=true
```

Example .env for local FAISS search (no Azure)

```
SEARCH_PROVIDER=local_faiss
SEARCH_INDEX_NAME=keyframes-local_search_index
# point to a folder that contains the example index files
SEARCH_INDEX_PATH=examples/mmct_faiss_indices
```

Note: `SEARCH_INDEX_PATH` is not an enforced env var in code — when using `local_faiss` you can set `prov.config['index_path']` programmatically or use the helper tools which detect `examples/mmct_faiss_indices`.

---

## Minimal code examples

Create a provider instance (recommended):

```python
from mmct.providers.factory import provider_factory

# create the default search provider from config
search = provider_factory.create_search_provider()

# or force a specific provider
faiss = provider_factory.create_search_provider('local_faiss')
faiss.config['index_path'] = 'examples/mmct_faiss_indices'
faiss.config['index_name'] = 'keyframes-local_search_index'
faiss.client = faiss._initialize_client()

# run a search with a precomputed embedding (list[float])
results = await faiss.search(query=None, index_name='keyframes-local_search_index', embedding=embedding, top=5)
```

Using the high-level VideoFrameSearchClient (preferred for video frame flows):

```python
from mmct.video_pipeline.utils.video_frame_search import VideoFrameSearchClient

client = VideoFrameSearchClient(provider_name='local_faiss', provider_config={'index_path': 'examples/mmct_faiss_indices'})
hits = await client.search_similar_frames(query_vector=embedding, top_k=10)
```

---

## Provider notes & gotchas

- Local FAISS (`local_faiss`)
  - Expects `embedding` as a kwarg when searching (list[float] or numpy array).
  - Persists two files per index under `index_path`: `<index_name>.index` (FAISS binary) and `<index_name>.meta.json` (metadata + stored documents).
  - Returns results as a list of dicts like `{'id': docid, 'score': <distance>, 'document': { ... }}`. The `score` is an L2 distance (lower == more similar).
  - It does NOT evaluate OData `filter` strings. If your app relies on filters (for example `video_id eq '...'`) you must post-filter FAISS results in code.

- Azure AI Search (`azure_ai_search`)
  - Accepts `vector_queries` (Azure VectorizedQuery) and OData `filter` strings.
  - Returns documents as flattened dicts (not nested under `document`).

Normalization recommendation

Because different providers return different result shapes, normalize results in one place (for example, in `VideoFrameSearchClient` or a small helper) so the rest of your code can expect the same format: `{'id','score','document':{...}}`.

Similarity thresholds

- FAISS returns distances (L2). If you want cosine similarity either index normalized vectors or compute cosine at query time using stored embeddings.
- Choose a threshold empirically (look at distances for known-good pairs and set a cutoff). Consider returning scores and letting the caller decide.

---

## Adding a provider — checklist

1. Implement a class that inherits from the correct abstract base under `mmct/providers/base`.
2. Put the implementation file under `mmct/providers/custom_providers/`.
3. Export your class in `mmct/providers/custom_providers/__init__.py`.
4. Register the provider name/class in `mmct/providers/factory.py`.
5. Add docs/examples in this guide and tests.

---

If you'd like, I can also:
- add a small normalization helper in `VideoFrameSearchClient` so callers get one consistent result shape (recommended), or
- add a short example script under `examples/` demonstrating end-to-end: load index, search, normalize and display results.

Last updated: synthesized from repository code on branch `v-soumyade/local-providers`.


---

## How to override provider selection at runtime

The system defaults to providers configured via `MMCTConfig` (which reads your `.env`). To override per-call you can use `ProviderFactory` directly.

Example — create and use the local FAISS search provider (same pattern used in tests)

```python
from mmct.providers.factory import provider_factory

# create provider instance by name
prov = provider_factory.create_search_provider('local_faiss')
# override index path if needed
prov.config['index_path'] = 'examples/mmct_faiss_indices'
prov.config['index_name'] = 'keyframes-local_search_index'
# ensure client shim
prov.client = prov._initialize_client()

# call search (embedding is a list[float])
results = await prov.search(query=None, index_name='keyframes-local_search_index', embedding=embedding_vector, top=5)
```

Example — create Azure search provider via factory (uses `mmct/config` values by default)

```python
from mmct.providers.factory import provider_factory
prov = provider_factory.create_search_provider('azure_ai_search')
# prov.config contains endpoint, api_key/use_managed_identity, index_name etc.
results = await prov.search(query='some text', top=10)
```

Example — `VideoFrameSearchClient` usage and provider_config override

```python
from mmct.video_pipeline.utils.video_frame_search import VideoFrameSearchClient

client = VideoFrameSearchClient(provider_name='local_faiss', provider_config={'index_path': 'examples/mmct_faiss_indices'})
results = await client.search_similar_frames(query_vector=embedding_vector, top_k=5)
```

---

## Practical recommendations

- Normalize provider outputs centrally (the local FAISS provider returns hits under `result['document']` while Azure returns a top-level document). Use `VideoFrameSearchClient` or a helper to normalize results for callers.
- For FAISS/IndexFlatL2 the provider returns L2 distances (lower = closer). Convert or rescore if you prefer cosine similarity. Consider normalizing vectors at index time for cosine.
- When using local FAISS exported indices, ensure `index_path` is the directory containing `<index_name>.index` and `<index_name>.meta.json` and that files are readable.
- Use managed identity for Azure services when running in Azure to avoid storing secrets in `.env`.

---

If you want, I can:
- add a small helper in `VideoFrameSearchClient` that normalizes search results across providers (recommended), or
- add a short example script under `examples/` showing how to create a provider, run a query, and normalize results.