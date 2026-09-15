"""Send drift / MLOps alerts via webhook (Slack-compatible)."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen


def send_webhook_alert(
    message: str,
    *,
    webhook_url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """
    Post a JSON payload to a Slack/Teams/generic webhook.

    Returns True when the webhook accepted the message, False when no URL is configured.
    """
    url = webhook_url or os.environ.get("ML_ALERT_WEBHOOK_URL", "").strip()
    if not url:
        return False

    payload: dict[str, Any] = {"text": message}
    if extra:
        payload.update(extra)

    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        return 200 <= response.status < 300
