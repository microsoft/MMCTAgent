# Docker Build, Test & Push Instructions

All commands are run from the **repository root** (`MMCTAgent/`).

---

## 1. Build

### Build the base image (only needed when dependencies change)

```bash
DOCKER_BUILDKIT=0 docker build -f Dockerfile.base -t mmct-base:latest .
```

### Build the app image

The patch version is auto-bumped and the build timestamp is stamped during this step.

```bash
./app/build.sh
```

Or build manually (without auto-bump):

```bash
DOCKER_BUILDKIT=0 docker build -f app/Dockerfile.main \
  -t mmct-lively-fastapi:latest \
  --build-arg BASE_IMAGE=mmct-base:latest .
```

---

## 2. Test locally

### Start the container

```bash
docker run -d --name mmct-test -p 8000:8000 \
  --env-file app/.env.gpt4.1 \
  mmct-lively-fastapi:latest
```

### Run the API test suite

```bash
python app/test_docker_api.py
# or with a custom port
python app/test_docker_api.py --base-url http://localhost:8000
```

### Quick manual smoke test

```bash
curl -s http://localhost:8000/          | python3 -m json.tool   # version + build time
curl -s http://localhost:8000/health    | python3 -m json.tool   # health check
curl -s http://localhost:8000/videos    | python3 -m json.tool   # list videos
```

### Stop and remove the test container

```bash
docker rm -f mmct-test
```

---

## 3. Push to Azure Container Registry

### Login to ACR

```bash
az acr login --name geckocontainerregistry
```

### Tag and push base image

```bash
docker tag mmct-base:latest geckocontainerregistry.azurecr.io/mmct-base:latest
docker push geckocontainerregistry.azurecr.io/mmct-base:latest
```

### Tag and push app image

```bash
docker tag mmct-lively-fastapi:latest geckocontainerregistry.azurecr.io/mmct-lively-fastapi:latest
docker push geckocontainerregistry.azurecr.io/mmct-lively-fastapi:latest
```

---

## 4. Deploy to Azure Container Apps

### Option A: Restart existing revision (pulls latest image)

```bash
az containerapp update \
  --name mmct-lively-fastapi-app \
  -g DefaultResourceGroup-CCAN \
  --image geckocontainerregistry.azurecr.io/mmct-lively-fastapi:latest
```

### Option B: Full Bicep deployment

```bash
export NEO4J_PASSWORD='<your-neo4j-password>'
cd app/deploy && ./deploy.sh
```

### Option C: Deploy only (skip image build)

```bash
export NEO4J_PASSWORD='<your-neo4j-password>'
cd app/deploy && ./deploy.sh --deploy-only
```

---

## Quick Reference: Full Build → Test → Push

```bash
# Build
DOCKER_BUILDKIT=0 docker build -f Dockerfile.base -t mmct-base:latest .
./app/build.sh

# Test
docker run -d --name mmct-test -p 8000:8000 --env-file app/.env.gpt4.1 mmct-lively-fastapi:latest
python app/test_docker_api.py
docker rm -f mmct-test

# Push
az acr login --name geckocontainerregistry
docker tag mmct-base:latest geckocontainerregistry.azurecr.io/mmct-base:latest
docker tag mmct-lively-fastapi:latest geckocontainerregistry.azurecr.io/mmct-lively-fastapi:latest
docker push geckocontainerregistry.azurecr.io/mmct-base:latest
docker push geckocontainerregistry.azurecr.io/mmct-lively-fastapi:latest
```
