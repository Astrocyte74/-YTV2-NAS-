"""
Render Dashboard Probe
Checks if Render has a specific report file
"""

import os
import requests

def render_has(stem: str) -> bool:
    """Check if Render dashboard has this report"""
    base = os.environ.get("RENDER_DASHBOARD_URL", "").rstrip("/")
    if not base:
        return False
    
    try:
        r = requests.get(f"{base}/{stem}.json", timeout=5)
        return r.status_code == 200
    except Exception:
        return False