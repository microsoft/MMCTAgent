# **Application built on MMCT Pipelines**

A modular **FastAPI** application for querying images and videos, ingesting videos directly or via an event queue, and consuming ingestion events for processing pipelines.

---

## 🧠 What’s Included

### ✅ Endpoints

#### **/query-on-images**  

Process an uploaded image and tools list (e.g., `"ocr,recog,object_detection,vit"` or `["ocr","recog","object_detection","vit"]`) via `ImageAgent`.  
- **Schema**: `ImageQueryRequest` (has query, tools, flags)  
- **Response**: `{ "result": ... }`

#### **/query-on-videos**  
Run a query against video content via `VideoAgent`.  
- **Schema**: `VideoQueryRequest` (query, index name, tools flags)  
- **Response**: `{ "result": ... }`

#### **/ingest-video**  
Ingests directly: uploads video locally, runs `IngestionPipeline`

- **Schema**: `IngestionRequest`

#### **/ingest-video-queue**  
Uploads video to Blob Storage, pushes an event payload to Event Hub, and triggers background ingestion via consumer script.  
- **Schema**: `IngestionRequest`

### 🧬 ingestion_consumer.py
A separate script that connects to the event hub, consumes ingestion queue messages, downloads blobs, runs pipelines, and performs cleanup.

---

## 🎯 Why This Structure?

- **Separation of Concerns**:  
  Routers ➝ API definitions  
  Schemas ➝ Validation  
  Services ➝ Business logic  
  Core/Config ➝ Environment settings

- **Validation & Documentation**:  
  Pydantic models validate inputs and define examples for Swagger UI.

- **Scalability**:  
  Modular design makes it easy to maintain, test, and extend.

---

## 🛠️ Getting Started

1. **Install dependencies**
  
    Install the dependencies mentioned in the repository
   ```bash
    pip install -r requirements.txt
   ```
2. **Run the API**

    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
3. **Run the consumer if event hub deployed**
   
   In another terminal
    ```bash
    python ingestion_consumer.py
    ```
4. **View docs**
   
   Swagger UI: http://localhost:8000/docs

> NOTE: Ensure you have the necessary environment variables set for Azure services. Highly recommended approach is to generate the env through the infra guide. 