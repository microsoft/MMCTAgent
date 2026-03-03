"""Single source of truth for API version and build metadata.

- MAJOR.MINOR.PATCH is bumped automatically at Docker build time.
- BUILD_TIMESTAMP is stamped during Docker build.
- Both are injected via build-args → environment variables so the
  running container always reflects when it was built.
"""

import os
from datetime import datetime, timezone

# Fallback values used during local development (no Docker build).
_DEFAULT_VERSION = "1.3.0"
_DEFAULT_BUILD_TIMESTAMP = ""

# At Docker build time these are baked into the image via ENV.
API_VERSION: str = os.environ.get("APP_API_VERSION", _DEFAULT_VERSION)

_raw_ts = os.environ.get("APP_BUILD_TIMESTAMP", _DEFAULT_BUILD_TIMESTAMP)
if _raw_ts:
    BUILD_TIMESTAMP: str = _raw_ts
else:
    BUILD_TIMESTAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
