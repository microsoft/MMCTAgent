#!/usr/bin/env python3
"""
MMCTAgent — Container Apps deployment orchestrator.

Reads infra/container_app/container_app_config.yaml and:
  1. Builds the base Docker image (Dockerfile.base)
  2. Builds the service image(s) — API, MCP, or both
  3. Logs in to Azure Container Registry and pushes the image(s)
  4. Provisions the Container Apps Environment (idempotent Bicep)
  5. Creates or updates the Container App(s) with all .env variables injected

Usage (from repo root):
    python infra/container_app/deploy.py

Required shell env vars:
    AZURE_RESOURCE_GROUP   existing Azure resource group
    AZURE_LOCATION         Azure region, e.g. eastus

All other settings (ACR name, image tags, replica counts, etc.) live in
infra/container_app/container_app_config.yaml.

Pre-requisites:
    - Docker running         : docker info
    - Azure CLI logged in    : az login
    - .env file at repo root : cp .env.example .env && fill it in
    - ACR already exists     : az acr show --name <acr_name>
    - Resource group exists  : az group show --name $AZURE_RESOURCE_GROUP
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = pathlib.Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "container_app_config.yaml"
BICEP_DIR   = SCRIPT_DIR / "bicep"
BICEP_FILE  = BICEP_DIR  / "container_env.bicep"
PARAMS_FILE = BICEP_DIR  / "container_env.parameters.json"
ENV_FILE    = REPO_ROOT  / ".env"

DOCKERFILE_BASE = REPO_ROOT / "Dockerfile.base"
DOCKERFILE = {
    "api": SCRIPT_DIR / "Dockerfile.api",
    "mcp": SCRIPT_DIR / "Dockerfile.mcp",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


def _ok(msg: str) -> None:
    print(f"\n✓  {msg}")


def _error(msg: str) -> None:
    print(f"\n✗  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming output live."""
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def _capture(cmd: list[str]) -> str:
    """Run a command and return stdout as a stripped string."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()


# ── Config loading & validation ───────────────────────────────────────────────

def load_config() -> dict:
    """Load container_app_config.yaml and validate required fields."""
    if not CONFIG_PATH.exists():
        _error(f"Config file not found: {CONFIG_PATH}\n"
               "  Edit infra/container_app/container_app_config.yaml.")

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("container_app")
    if not isinstance(cfg, dict):
        _error("container_app_config.yaml must have a top-level 'container_app:' key.")

    service = cfg.get("service", "api")
    if service not in ("api", "mcp", "both"):
        _error(f"container_app.service must be 'api', 'mcp', or 'both', got: {service!r}")

    acr_name = cfg.get("acr_name", "")
    if not acr_name or acr_name == "<your-acr-name>":
        _error(
            "container_app.acr_name is not set.\n"
            "  Edit infra/container_app/container_app_config.yaml and set a real ACR name."
        )

    env_name = cfg.get("environment_name", "")
    if not env_name:
        _error("container_app.environment_name must be set in container_app_config.yaml.")

    return cfg


def print_config_summary(cfg: dict, resource_group: str, location: str) -> None:
    service  = cfg["service"]
    acr_name = cfg["acr_name"]
    env_name = cfg["environment_name"]

    print()
    print("  Container App deployment config")
    print(f"  ├── Service          : {service}")
    print(f"  ├── ACR              : {acr_name}.azurecr.io")
    print(f"  ├── Environment      : {env_name}")
    print(f"  ├── Resource group   : {resource_group}")
    print(f"  └── Location         : {location}")
    if service in ("api", "both"):
        api = cfg.get("api", {})
        print(f"\n  API container")
        print(f"  ├── App name         : {api.get('app_name', 'mmctagent-api')}")
        print(f"  ├── Image tag        : {api.get('image_tag', 'latest')}")
        print(f"  ├── Replicas         : {api.get('min_replicas', 1)}–{api.get('max_replicas', 3)}")
        print(f"  └── Resources        : {api.get('cpu', '1.0')} vCPU / {api.get('memory', '2Gi')}")
    if service in ("mcp", "both"):
        mcp = cfg.get("mcp", {})
        print(f"\n  MCP container")
        print(f"  ├── App name         : {mcp.get('app_name', 'mmctagent-mcp')}")
        print(f"  ├── Image tag        : {mcp.get('image_tag', 'latest')}")
        print(f"  ├── Replicas         : {mcp.get('min_replicas', 1)}–{mcp.get('max_replicas', 2)}")
        print(f"  └── Resources        : {mcp.get('cpu', '0.5')} vCPU / {mcp.get('memory', '1Gi')}")


# ── Pre-flight checks ─────────────────────────────────────────────────────────

def _check_docker() -> None:
    if shutil.which("docker") is None:
        _error("Docker is not installed.\n"
               "  Install Docker Desktop: https://docs.docker.com/get-docker/")
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        _error("Docker is not running.\n"
               "  Start Docker Desktop or: sudo systemctl start docker")


def _check_az() -> None:
    if shutil.which("az") is None:
        _error("Azure CLI (az) is not installed.\n"
               "  Install: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
    if subprocess.run(["az", "account", "show"], capture_output=True).returncode != 0:
        _error("Not logged in to Azure CLI.\n  Run: az login")


def _require_env(var: str, hint: str = "") -> str:
    value = os.environ.get(var, "").strip()
    if not value:
        detail = f"\n  Hint: {hint}" if hint else ""
        _error(f"Environment variable {var!r} is not set.{detail}\n"
               f"  Export it:  export {var}=<value>")
    return value


# ── .env parsing ──────────────────────────────────────────────────────────────

def load_env_vars() -> dict[str, str]:
    """Read the repo-root .env file and return KEY → VALUE pairs."""
    if not ENV_FILE.exists():
        _error(
            f".env file not found at {ENV_FILE}\n"
            "  Copy the template first:\n"
            "    cp .env.example .env\n"
            "  Then fill in all required values."
        )
    env: dict[str, str] = {}
    with open(ENV_FILE) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                env[key] = value
    if not env:
        _error(".env file exists but contains no key=value pairs.\n"
               "  Fill in the required values — see .env.example for the full list.")
    return env


def build_env_vars_flag(env: dict[str, str]) -> list[str]:
    """Convert a dict of env vars to az containerapp --env-vars format."""
    return [f"{k}={v}" for k, v in env.items()]


# ── Docker build ──────────────────────────────────────────────────────────────

def build_images(services: list[str], acr_name: str, cfg: dict) -> dict[str, str]:
    """
    Build the base image, then each service image.
    Returns {service: full_image_ref} for each service.
    """
    print(f"    Building base image: mmctagent-base:latest")
    _run([
        "docker", "build",
        "-f", str(DOCKERFILE_BASE),
        "-t", "mmctagent-base:latest",
        str(REPO_ROOT),
    ])

    full_images: dict[str, str] = {}
    for svc in services:
        svc_cfg   = cfg.get(svc, {})
        img_name  = svc_cfg.get("image_name", f"mmctagent-{svc}")
        img_tag   = svc_cfg.get("image_tag", "latest")
        full_img  = f"{acr_name}.azurecr.io/{img_name}:{img_tag}"
        print(f"    Building {svc} image: {full_img}")
        _run([
            "docker", "build",
            "-f", str(DOCKERFILE[svc]),
            "-t", full_img,
            str(REPO_ROOT),
        ])
        full_images[svc] = full_img

    return full_images


# ── ACR push ──────────────────────────────────────────────────────────────────

def push_images(acr_name: str, full_images: dict[str, str]) -> None:
    """Login to ACR and push all images."""
    _run(["az", "acr", "login", "--name", acr_name])
    for svc, full_img in full_images.items():
        print(f"    Pushing {svc}: {full_img}")
        _run(["docker", "push", full_img])


# ── Container Apps Environment (Bicep) ───────────────────────────────────────

def deploy_environment(resource_group: str, location: str, env_name: str,
                       log_workspace: str) -> None:
    """Idempotently deploy the Container Apps Environment + Log Analytics."""
    _run([
        "az", "deployment", "group", "create",
        "--resource-group",  resource_group,
        "--template-file",   str(BICEP_FILE),
        "--parameters",      str(PARAMS_FILE),
        "--parameters",
        f"location={location}",
        f"environmentName={env_name}",
        f"logAnalyticsWorkspaceName={log_workspace}",
        "--mode",   "Incremental",
        "--output", "table",
    ])


# ── Container App create / update ─────────────────────────────────────────────

def _app_exists(app_name: str, resource_group: str) -> bool:
    result = subprocess.run([
        "az", "containerapp", "show",
        "--name",            app_name,
        "--resource-group",  resource_group,
    ], capture_output=True)
    return result.returncode == 0


def deploy_container_app(
    svc: str,
    cfg: dict,
    resource_group: str,
    acr_name: str,
    env_name: str,
    full_image: str,
    env_vars: list[str],
) -> str:
    """Create or update a Container App. Returns the FQDN."""
    svc_cfg      = cfg.get(svc, {})
    app_name     = svc_cfg.get("app_name", f"mmctagent-{svc}")
    min_replicas = str(svc_cfg.get("min_replicas", 1))
    max_replicas = str(svc_cfg.get("max_replicas", 3))
    cpu          = str(svc_cfg.get("cpu", "1.0"))
    memory       = str(svc_cfg.get("memory", "2Gi"))

    # --registry-identity system: use the Container App's system-assigned managed
    # identity to pull from ACR. Azure auto-assigns the AcrPull role on the ACR
    # during creation — no admin credentials or manual role assignment needed.
    base_flags = [
        "--name",               app_name,
        "--resource-group",     resource_group,
        "--image",              full_image,
        "--registry-server",    f"{acr_name}.azurecr.io",
        "--registry-identity",  "system",
        "--target-port",        "8000",
        "--cpu",                cpu,
        "--memory",             memory,
        "--env-vars",           *env_vars,
    ]

    if _app_exists(app_name, resource_group):
        print(f"    Container App '{app_name}' exists — updating ...")
        _run(["az", "containerapp", "update"] + base_flags)
    else:
        print(f"    Container App '{app_name}' not found — creating ...")
        _run([
            "az", "containerapp", "create",
            "--environment",    env_name,
            "--ingress",        "external",
            "--min-replicas",   min_replicas,
            "--max-replicas",   max_replicas,
        ] + base_flags)

    fqdn = _capture([
        "az", "containerapp", "show",
        "--name",           app_name,
        "--resource-group", resource_group,
        "--query",          "properties.configuration.ingress.fqdn",
        "--output",         "tsv",
    ])
    return fqdn


# ── Summary boxes ─────────────────────────────────────────────────────────────

def _print_api_summary(fqdn: str, app_name: str) -> None:
    pad = lambda s, w: ' ' * max(0, w - len(s))
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  MMCTAgent API — Live                               │")
    print("  │                                                     │")
    print(f"  │  Endpoint  →  https://{fqdn}{pad(fqdn, 29)}│")
    print(f"  │  Health    →  https://{fqdn}/health{pad(fqdn + '/health', 29)}│")
    print(f"  │  Docs      →  https://{fqdn}/docs{pad(fqdn + '/docs', 29)}│")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Stream live logs:")
    print(f"    az containerapp logs show --name {app_name} \\")
    print( "      --resource-group $AZURE_RESOURCE_GROUP --follow")


def _print_mcp_summary(fqdn: str, app_name: str) -> None:
    pad = lambda s, w: ' ' * max(0, w - len(s))
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  MMCTAgent MCP Server — Live                        │")
    print("  │                                                     │")
    print(f"  │  Health    →  https://{fqdn}/{pad(fqdn + '/', 29)}│")
    print(f"  │  MCP       →  https://{fqdn}/mcp{pad(fqdn + '/mcp', 29)}│")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Stream live logs:")
    print(f"    az containerapp logs show --name {app_name} \\")
    print( "      --resource-group $AZURE_RESOURCE_GROUP --follow")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MMCTAgent — Container Apps Deploy")
    print("=" * 60)

    cfg            = load_config()
    service        = cfg["service"]
    acr_name       = cfg["acr_name"]
    env_name       = cfg["environment_name"]
    log_workspace  = cfg.get("log_analytics_workspace", "mmct-logs")

    resource_group = _require_env("AZURE_RESOURCE_GROUP")
    location       = _require_env("AZURE_LOCATION")

    print_config_summary(cfg, resource_group, location)

    services = ["api", "mcp"] if service == "both" else [service]
    total    = 5

    # ── Step 1: pre-flight ─────────────────────────────────────────────────
    _step(1, total, "Running pre-flight checks ...")
    _check_docker()
    _check_az()
    env_vars_dict = load_env_vars()
    print(f"    .env loaded : {len(env_vars_dict)} variables")

    # ── Step 2: build images ───────────────────────────────────────────────
    _step(2, total, f"Building Docker image(s) for: {', '.join(services)} ...")
    full_images = build_images(services, acr_name, cfg)

    # ── Step 3: push to ACR ────────────────────────────────────────────────
    _step(3, total, f"Logging in to ACR ({acr_name}) and pushing image(s) ...")
    push_images(acr_name, full_images)

    # ── Step 4: deploy environment (idempotent) ────────────────────────────
    _step(4, total, "Deploying Container Apps Environment (idempotent) ...")
    deploy_environment(resource_group, location, env_name, log_workspace)

    # ── Step 5: deploy Container App(s) ───────────────────────────────────
    _step(5, total, f"Deploying Container App(s): {', '.join(services)} ...")
    env_vars_list = build_env_vars_flag(env_vars_dict)

    fqdns: dict[str, str] = {}
    for svc in services:
        fqdns[svc] = deploy_container_app(
            svc, cfg, resource_group, acr_name, env_name,
            full_images[svc], env_vars_list,
        )

    _ok("Deployment complete.")

    for svc in services:
        svc_cfg  = cfg.get(svc, {})
        app_name = svc_cfg.get("app_name", f"mmctagent-{svc}")
        if svc == "api":
            _print_api_summary(fqdns[svc], app_name)
        else:
            _print_mcp_summary(fqdns[svc], app_name)


if __name__ == "__main__":
    main()
