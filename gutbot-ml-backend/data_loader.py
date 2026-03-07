# data_loader.py
import os, json, time, hashlib, logging
from typing import Optional, Dict, Any
from functools import lru_cache
import requests

logger = logging.getLogger(__name__)

BASE_DATA_URL = os.getenv("MEDICAL_DATA_BASE_URL", "").rstrip("/")
if BASE_DATA_URL and not BASE_DATA_URL.endswith("/"):
    BASE_DATA_URL += "/"

REDIS_URL = os.getenv("REDIS_URL", "")
try:
    import redis  # optional
    redis_client = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    redis_client = None

DATA_TTL = int(os.getenv("DATA_CACHE_TTL_SECONDS", "86400"))   # 1 day
ETAG_TTL = int(os.getenv("ETAG_CACHE_TTL_SECONDS", "604800"))  # 7 days
TIMEOUT = (5, 10)  # connect/read

def _safe_join(*parts: str) -> str:
    clean = []
    for p in parts:
        p = p.strip().lstrip("/").rstrip("/")
        if ".." in p or p.startswith("http"):
            raise ValueError("Invalid path part")
        if p:
            clean.append(p)
    return "/".join(clean)

def _cache_keys(path: str) -> Dict[str, str]:
    h = hashlib.sha256(path.encode("utf-8")).hexdigest()
    return {"etag": f"md:etag:{h}", "blob": f"md:blob:{h}", "ts": f"md:ts:{h}"}

@lru_cache(maxsize=1)
def _base_url() -> str:
    return BASE_DATA_URL

class RemoteJSONLoader:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or _base_url()).rstrip("/") + "/"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "GutBot/1.0 (+health)"})

    def _url(self, *path_parts: str) -> str:
        return self.base_url + _safe_join(*path_parts)

    def _get_redis(self, key: str) -> Optional[str]:
        if not redis_client: return None
        try:
            v = redis_client.get(key)
            return v.decode("utf-8") if v else None
        except Exception:
            return None

    def _set_redis(self, key: str, val: str, ttl: int) -> None:
        if not redis_client: return
        try:
            redis_client.setex(key, ttl, val)
        except Exception:
            pass

    def _get_blob(self, k_blob: str, k_ts: str):
        if not redis_client: return None
        try:
            ts = self._get_redis(k_ts)
            if not ts: return None
            if time.time() - float(ts) > DATA_TTL:
                return None
            blob = redis_client.get(k_blob)
            return json.loads(blob) if blob else None
        except Exception:
            return None

    def _set_blob(self, k_blob: str, k_ts: str, payload: Dict[str, Any]):
        if not redis_client: return
        try:
            self._set_redis(k_blob, json.dumps(payload), DATA_TTL)
            self._set_redis(k_ts, str(time.time()), DATA_TTL)
        except Exception:
            pass

    def get_json(self, *path_parts: str) -> Dict[str, Any]:
        if not _base_url():
            raise RuntimeError("MEDICAL_DATA_BASE_URL not set; remote fetch unavailable")

        rel = _safe_join(*path_parts)
        url = self._url(rel)
        keys = _cache_keys(rel)

        cached = self._get_blob(keys["blob"], keys["ts"])
        headers = {}
        etag = self._get_redis(keys["etag"])
        if etag:
            headers["If-None-Match"] = etag

        backoff = 0.5
        for _ in range(4):
            try:
                resp = self.session.get(url, headers=headers, timeout=TIMEOUT)
                if resp.status_code == 304 and cached is not None:
                    return cached
                if resp.status_code == 200:
                    data = resp.json()
                    new_etag = resp.headers.get("ETag")
                    if new_etag:
                        self._set_redis(keys["etag"], new_etag, ETAG_TTL)
                    self._set_blob(keys["blob"], keys["ts"], data)
                    return data
                if resp.status_code == 404:
                    raise FileNotFoundError(f"Remote not found: {url}")
            except requests.RequestException as e:
                logger.warning("Remote fetch error %s: %s", url, e)
            time.sleep(backoff)
            backoff = min(4.0, backoff * 2)

        if cached is not None:
            logger.warning("Serving stale cached data for %s", rel)
            return cached
        raise RuntimeError(f"Unable to fetch remote JSON: {url}")

loader = RemoteJSONLoader()
