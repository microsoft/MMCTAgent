#!/usr/bin/env python3
"""
Neo4j deployment orchestrator for MMCTAgent.

Reads infra/neo4j/neo4j_config.yaml and routes to one of two paths:

  vm_deployment: false  →  runs Neo4j as a local Docker container
  vm_deployment: true   →  provisions an Azure VM + VNet + LB via Bicep

Usage (from repo root):
    python infra/neo4j/deploy.py

Pre-requisites:
  Local path  : Docker Engine / Docker Desktop running
  Azure path  : az CLI installed + logged in, plus three env vars:
                  AZURE_RESOURCE_GROUP, AZURE_LOCATION, AZURE_ADMIN_PASSWORD
"""

import os
import pathlib
import shutil
import subprocess
import sys

import yaml

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = pathlib.Path(__file__).parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
CONFIG_PATH = SCRIPT_DIR / "neo4j_config.yaml"
BICEP_DIR   = SCRIPT_DIR / "bicep"

# Edition-specific Bicep templates
BICEP_FILE = {
    "community":  BICEP_DIR / "main.bicep",
    "enterprise": BICEP_DIR / "main-enterprise.bicep",
}
PARAMS_FILE = {
    "community":  BICEP_DIR / "main.parameters.json",
    "enterprise": BICEP_DIR / "main-enterprise.parameters.json",
}

NEO4J_VERSION = "5.26.0"

DOCKER_IMAGES = {
    "community": f"neo4j:{NEO4J_VERSION}-community",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


def _ok(msg: str) -> None:
    print(f"\n✓  {msg}")


def _error(msg: str) -> None:
    print(f"\n✗  ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, streaming its output live."""
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


# ── Config loading & validation ──────────────────────────────────────────────

def load_config() -> dict:
    """Load neo4j_config.yaml and validate required fields."""
    if not CONFIG_PATH.exists():
        _error(f"Config file not found: {CONFIG_PATH}\n"
               "  Copy neo4j_config.yaml.example or create it from the template.")

    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)

    cfg = raw.get("neo4j")
    if not isinstance(cfg, dict):
        _error("neo4j_config.yaml must have a top-level 'neo4j:' key.")

    edition       = cfg.get("version", "community")
    vm_deployment = cfg.get("vm_deployment", False)

    if edition not in ("community", "enterprise"):
        _error(f"neo4j.version must be 'community' or 'enterprise', got: {edition!r}")

    if edition == "enterprise" and not vm_deployment:
        _error(
            "Enterprise edition is not supported for local Docker deployment.\n"
            "  Local Docker only supports Community edition.\n"
            "  To use Enterprise, set vm_deployment: true and deploy to Azure VM."
        )

    password = cfg.get("password", "")
    if not password or password in ("your_password_here", "neo4j"):
        _error("neo4j.password must be set to a non-default, non-empty value.\n"
               "  Edit infra/neo4j/neo4j_config.yaml and set a real password.")

    return cfg


def print_config_summary(cfg: dict) -> None:
    edition       = cfg["version"]
    vm_deployment = cfg.get("vm_deployment", False)
    mode          = "Azure VM (Bicep)" if vm_deployment else "Local Docker"
    data_path     = cfg.get("data_path", "./neo4j/data") if not vm_deployment else "Azure Premium SSD (/var/lib/neo4j)"

    print()
    print("  Neo4j deployment config")
    print(f"  ├── Edition       : {edition}")
    print(f"  ├── Mode          : {mode}")
    print(f"  └── Data path     : {data_path}")


# ── Local Docker path ─────────────────────────────────────────────────────────

def _check_docker() -> None:
    """Verify Docker is reachable."""
    if shutil.which("docker") is None:
        _error("Docker is not installed or not on PATH.\n"
               "  Install Docker Desktop: https://docs.docker.com/get-docker/")
    result = subprocess.run(["docker", "info"], capture_output=True)
    if result.returncode != 0:
        _error("Docker is installed but not running.\n"
               "  Start Docker Desktop or run: sudo systemctl start docker")


def deploy_local(cfg: dict) -> None:
    edition   = cfg["version"]
    password  = cfg["password"]
    image     = DOCKER_IMAGES[edition]
    raw_path  = cfg.get("data_path", "./neo4j/data")
    data_path = (REPO_ROOT / raw_path).resolve() if not pathlib.Path(raw_path).is_absolute() \
                else pathlib.Path(raw_path)

    total = 4

    _step(1, total, "Checking Docker ...")
    _check_docker()

    _step(2, total, f"Pulling {image} ...")
    _run(["docker", "pull", image])

    _step(3, total, "Stopping any existing 'neo4j' container ...")
    subprocess.run(["docker", "rm", "-f", "neo4j"], capture_output=True)

    _step(4, total, f"Starting Neo4j ({edition}) with data at {data_path} ...")
    data_path.mkdir(parents=True, exist_ok=True)

    env_flags = ["-e", f"NEO4J_AUTH=neo4j/{password}"]

    _run([
        "docker", "run", "-d",
        "--name", "neo4j",
        "--restart", "unless-stopped",
        "-p", "7474:7474",
        "-p", "7687:7687",
        "-v", f"{data_path}:/data",
        *env_flags,
        image,
    ])

    _ok("Neo4j is running.")
    _print_local_summary(password)


def _print_local_summary(password: str) -> None:
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  Neo4j Local Connection                             │")
    print("  │                                                     │")
    print("  │  Browser  →  http://localhost:7474                  │")
    print("  │  Bolt     →  bolt://localhost:7687                  │")
    print("  │  User     →  neo4j                                  │")
    print(f"  │  Password →  {password:<37}│")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Add these lines to your .env file:")
    print()
    print("    NEO4J_URI=bolt://localhost:7687")
    print("    NEO4J_USERNAME=neo4j")
    print(f"    NEO4J_PASSWORD={password}")
    print("    NEO4J_DATABASE=neo4j")
    print()
    print("  To stop Neo4j:  docker stop neo4j")
    print("  To remove it :  docker rm -f neo4j")


# ── Azure VM path ─────────────────────────────────────────────────────────────

def _check_az_cli() -> None:
    """Verify az CLI is installed and authenticated."""
    if shutil.which("az") is None:
        _error("Azure CLI (az) is not installed.\n"
               "  Install it: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli")
    result = subprocess.run(["az", "account", "show"], capture_output=True)
    if result.returncode != 0:
        _error("Not logged in to Azure CLI.\n"
               "  Run: az login")


def _require_env(var: str) -> str:
    """Return env var value or exit with a helpful message."""
    value = os.environ.get(var)
    if not value:
        _error(
            f"Environment variable {var!r} is not set.\n"
            f"  Export it before running this script:\n"
            f"    export {var}=<value>"
        )
    return value


def deploy_vm(cfg: dict) -> None:
    edition        = cfg["version"]
    password       = cfg["password"]
    license_key    = cfg.get("license_key", "")

    bicep_file  = BICEP_FILE[edition]
    params_file = PARAMS_FILE[edition]

    total = 3

    _step(1, total, "Validating Azure environment ...")
    _check_az_cli()
    resource_group = _require_env("AZURE_RESOURCE_GROUP")
    location       = _require_env("AZURE_LOCATION")
    admin_password = _require_env("AZURE_ADMIN_PASSWORD")

    print(f"    Resource group : {resource_group}")
    print(f"    Location       : {location}")
    print(f"    Edition        : {edition}")
    print(f"    Bicep template : {bicep_file.name}")
    if edition == "enterprise":
        license_mode = "paid license file" if license_key else "evaluation (env var)"
        print(f"    License        : {license_mode}")

    _step(2, total, f"Running Bicep deployment ({bicep_file.name}) — this takes ~10 minutes ...")

    # Base parameters shared by both editions
    extra_params = [
        f"neo4jPassword={password}",
        f"location={location}",
        f"adminPasswordOrKey={admin_password}",
    ]

    # Community template still carries the edition parameter
    if edition == "community":
        extra_params.append("neo4jEdition=community")

    # Enterprise template accepts an optional base64-encoded license key
    if edition == "enterprise" and license_key:
        import base64 as _b64
        encoded = _b64.b64encode(license_key.encode()).decode()
        extra_params.append(f"neo4jLicenseKeyBase64={encoded}")

    # Use a deterministic deployment name tied to the edition so re-runs
    # never pull stale outputs from a different edition's deployment.
    deployment_name = f"neo4j-{edition}"

    _run([
        "az", "deployment", "group", "create",
        "--resource-group",  resource_group,
        "--name",            deployment_name,
        "--template-file",   str(bicep_file),
        "--parameters",      str(params_file),
        "--parameters",      *extra_params,
        "--output", "table",
    ])

    _step(3, total, "Fetching deployment outputs ...")
    result = subprocess.run([
        "az", "deployment", "group", "show",
        "--resource-group", resource_group,
        "--name", deployment_name,
        "--query", "properties.outputs",
        "--output", "json",
    ], capture_output=True, text=True)

    public_ip = "N/A (enablePublicIP=false)"
    if result.returncode == 0:
        import json
        try:
            outputs   = json.loads(result.stdout)
            public_ip = outputs.get("loadBalancerPublicIP", {}).get("value", public_ip)
        except Exception:
            pass

    _ok("Neo4j deployed on Azure VM.")
    _print_vm_summary(password, public_ip, edition)


def _print_vm_summary(password: str, public_ip: str, edition: str = "community") -> None:
    pad = lambda s: ' ' * max(0, 22 - len(s))
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Neo4j {edition.capitalize():<10} Azure VM Connection          │")
    print("  │                                                     │")
    print("  │  Private Bolt  →  bolt://10.0.1.10:7687             │")
    print("  │  Private HTTP  →  http://10.0.1.10:7474             │")
    print(f"  │  Public Bolt   →  bolt://{public_ip}:7687{pad(public_ip)}│")
    print(f"  │  Public HTTP   →  http://{public_ip}:7474{pad(public_ip)}│")
    if edition == "enterprise":
        print("  │  Metrics       →  http://10.0.1.10:2004/metrics    │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    print("  Add these lines to your .env file (use private IP if within the VNet,")
    print("  or public IP if connecting from outside Azure):")
    print()
    print("    NEO4J_URI=bolt://10.0.1.10:7687")
    print("    NEO4J_USERNAME=neo4j")
    print(f"    NEO4J_PASSWORD={password}")
    print("    NEO4J_DATABASE=neo4j")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  MMCTAgent — Neo4j Deploy")
    print("=" * 60)

    cfg           = load_config()
    vm_deployment = cfg.get("vm_deployment", False)

    print_config_summary(cfg)

    if vm_deployment:
        deploy_vm(cfg)
    else:
        deploy_local(cfg)


if __name__ == "__main__":
    main()