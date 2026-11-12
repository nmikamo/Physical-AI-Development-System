"""S3 I/O adapter isolating persistence of papers, text mining, mocap, and reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class S3Store:
    """Thin I/O boundary over Amazon S3 with local mirror cache for <3s UI latency."""

    def __init__(self, bucket: str, region: str, local_dir: Path) -> None:
        self._bucket = bucket
        self._region = region
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)
        self._client = boto3.client("s3", region_name=region)

    @property
    def bucket(self) -> str:
        return self._bucket

    def _local_path(self, key: str) -> Path:
        safe_rel = key.lstrip("/")
        path = self._local_dir / safe_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def put_json(self, key: str, payload: Any) -> str:
        """Save JSON to local cache and upload to S3."""
        raw = json.dumps(payload, ensure_ascii=False, indent=2)
        self._local_path(key).write_text(raw, encoding="utf-8")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=raw.encode("utf-8"),
                ContentType="application/json; charset=utf-8",
            )
        except (BotoCoreError, ClientError) as exc:
            logger.warning("S3 put_json fallback to local only for %s: %s", key, exc)
        return key

    def get_json(self, key: str) -> Any | None:
        """Read JSON from local cache first for <3s response, falling back to S3."""
        local = self._local_path(key)
        if local.exists():
            try:
                return json.loads(local.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            body = resp["Body"].read().decode("utf-8")
            local.write_text(body, encoding="utf-8")
            return json.loads(body)
        except (BotoCoreError, ClientError):
            return None

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Save binary data (mocap CSV/NPZ/BVH, MP4 video, XML) to local cache and S3."""
        self._local_path(key).write_bytes(data)
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.warning("S3 put_bytes fallback to local only for %s: %s", key, exc)
        return key

    def upload_file(self, local_file: Path, key: str, content_type: str) -> str:
        """Upload an existing local file to S3 and mirror in cache."""
        dest = self._local_path(key)
        if local_file.resolve() != dest.resolve():
            dest.write_bytes(local_file.read_bytes())
        try:
            self._client.upload_file(
                str(local_file),
                self._bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except (BotoCoreError, ClientError) as exc:
            logger.warning("S3 upload_file failed for %s: %s", key, exc)
        return key

    def get_bytes(self, key: str) -> bytes | None:
        local = self._local_path(key)
        if local.exists():
            return local.read_bytes()
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            data = resp["Body"].read()
            local.write_bytes(data)
            return data
        except (BotoCoreError, ClientError):
            return None

    def ensure_local_file(self, key: str) -> Path | None:
        """Ensure object exists on local disk for FileResponse HTTP Range streaming."""
        local = self._local_path(key)
        if local.exists() and local.stat().st_size > 0:
            return local
        data = self.get_bytes(key)
        return local if data else None

    def list_keys(self, prefix: str) -> tuple[str, ...]:
        """List keys from S3 merged with local cache keys."""
        keys: set[str] = set()
        local_prefix_dir = self._local_dir / prefix.lstrip("/")
        if local_prefix_dir.exists():
            for p in local_prefix_dir.rglob("*"):
                if p.is_file():
                    keys.add(p.relative_to(self._local_dir).as_posix())
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    k = str(obj["Key"])
                    if not k.endswith("/"):
                        keys.add(k)
        except (BotoCoreError, ClientError) as exc:
            logger.warning("S3 list_keys error for %s: %s", prefix, exc)
        return tuple(sorted(keys, reverse=True))
