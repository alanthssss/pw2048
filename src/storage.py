"""AWS S3 storage helpers for pw2048 results.

This module is optional: S3 features are only activated when a bucket name
is supplied by the caller.  ``boto3`` is imported lazily so the rest of the
application works without it.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Optional

_HAS_BOTO3 = False
try:
    import boto3  # type: ignore[import]
    _HAS_BOTO3 = True
except ImportError:
    pass


def _require_boto3() -> None:
    if not _HAS_BOTO3:
        raise ImportError(
            "boto3 is required for S3 operations.  "
            "Install it with:  pip install boto3"
        )


# ---------------------------------------------------------------------------
# Low-level S3 helpers
# ---------------------------------------------------------------------------

def upload_file(
    local_path: str | Path,
    bucket: str,
    s3_key: str,
    content_type: Optional[str] = None,
    public_read: bool = False,
) -> str:
    """Upload *local_path* to *s3_key* inside *bucket*.

    Parameters
    ----------
    local_path:
        Path to the local file to upload.
    bucket:
        Target S3 bucket name.
    s3_key:
        Destination key inside the bucket (no leading ``/``).
    content_type:
        MIME type to set on the S3 object.  Auto-detected when ``None``.
    public_read:
        When ``True``, applies the ``public-read`` ACL so the file is
        accessible via a plain HTTPS URL without signing.

    Returns
    -------
    str
        The HTTPS URL of the uploaded object.
    """
    _require_boto3()
    local_path = Path(local_path)

    extra_args: dict = {}
    if public_read:
        extra_args["ACL"] = "public-read"
    if content_type is None:
        content_type, _ = mimetypes.guess_type(str(local_path))
    if content_type:
        extra_args["ContentType"] = content_type

    client = boto3.client("s3")
    client.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra_args or None)

    region = (
        client.get_bucket_location(Bucket=bucket).get("LocationConstraint")
        or "us-east-1"
    )
    if region == "us-east-1":
        url = f"https://{bucket}.s3.amazonaws.com/{s3_key}"
    else:
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
    return url


def list_result_stems(bucket: str, prefix: str) -> list[str]:
    """Return a sorted list of unique timestamp stems under *prefix* in S3.

    For example, if the bucket contains::

        results/Random/20260307_120000.csv
        results/Random/20260307_120000.png
        results/Random/20260307_130000.csv
        results/Random/20260307_130000.png

    calling ``list_result_stems(bucket, "results/Random/")`` returns::

        ["20260307_120000", "20260307_130000"]
    """
    _require_boto3()
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    stems: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            stems.add(Path(key).stem)
    return sorted(stems)


def delete_s3_objects(bucket: str, keys: list[str]) -> None:
    """Delete *keys* from *bucket* in one batch API call."""
    _require_boto3()
    if not keys:
        return
    client = boto3.client("s3")
    client.delete_objects(
        Bucket=bucket,
        Delete={"Objects": [{"Key": k} for k in keys]},
    )


# ---------------------------------------------------------------------------
# Higher-level helpers used by main.py
# ---------------------------------------------------------------------------

def prune_s3_results(
    bucket: str,
    s3_prefix: str,
    algorithm_name: str,
    keep_n: int,
) -> list[str]:
    """Delete old algorithm results from S3, keeping only the latest *keep_n* runs.

    Parameters
    ----------
    bucket:
        Target S3 bucket.
    s3_prefix:
        Bucket-level prefix that precedes ``/<algorithm_name>/``.
    algorithm_name:
        Name of the algorithm sub-directory (e.g. ``"Random"``).
    keep_n:
        Number of most-recent runs to retain.

    Returns
    -------
    list[str]
        S3 keys that were deleted.
    """
    prefix = f"{s3_prefix.rstrip('/')}/{algorithm_name}/"
    stems = list_result_stems(bucket, prefix)
    old_stems = stems[:-keep_n] if len(stems) > keep_n else []

    keys_to_delete: list[str] = []
    for stem in old_stems:
        for ext in ("csv", "png"):
            keys_to_delete.append(f"{prefix}{stem}.{ext}")

    delete_s3_objects(bucket, keys_to_delete)
    return keys_to_delete


def sync_run_to_s3(
    output_dir: Path,
    timestamp: str,
    bucket: str,
    s3_prefix: str,
    algorithm_name: str,
    keep_n: int,
    public_read: bool = False,
) -> dict[str, str]:
    """Upload one run's CSV and PNG to S3, then prune older runs.

    Parameters
    ----------
    output_dir:
        Local directory that contains the timestamped files.
    timestamp:
        The timestamp stem used for the run (e.g. ``"20260307_120000"``).
    bucket:
        S3 bucket name.
    s3_prefix:
        Top-level S3 prefix (e.g. ``"results"``).
    algorithm_name:
        Algorithm name used as the sub-directory.
    keep_n:
        Number of runs to keep per algorithm after pruning.
    public_read:
        Whether to apply public-read ACL to the uploaded objects.

    Returns
    -------
    dict[str, str]
        Mapping of ``local_path_str → s3_url`` for every file uploaded.
    """
    _require_boto3()
    algo_prefix = f"{s3_prefix.rstrip('/')}/{algorithm_name}"
    uploaded: dict[str, str] = {}

    for ext, ctype in (("csv", "text/csv"), ("png", "image/png")):
        local = output_dir / f"{timestamp}.{ext}"
        if local.exists():
            key = f"{algo_prefix}/{timestamp}.{ext}"
            url = upload_file(local, bucket, key, content_type=ctype, public_read=public_read)
            uploaded[str(local)] = url
            print(f"  Uploaded → s3://{bucket}/{key}")

    deleted = prune_s3_results(bucket, s3_prefix, algorithm_name, keep_n)
    if deleted:
        print(f"  Pruned {len(deleted)} old S3 object(s) for '{algorithm_name}'")

    return uploaded


def upload_report(
    report_path: Path,
    bucket: str,
    s3_prefix: str,
    public_read: bool = False,
) -> str:
    """Upload the HTML report to ``<s3_prefix>/index.html`` and return its URL.

    Parameters
    ----------
    report_path:
        Local path of the generated HTML file.
    bucket:
        S3 bucket name.
    s3_prefix:
        Top-level S3 prefix.
    public_read:
        Whether to apply public-read ACL.

    Returns
    -------
    str
        HTTPS URL of the uploaded report.
    """
    _require_boto3()
    key = f"{s3_prefix.rstrip('/')}/index.html"
    url = upload_file(report_path, bucket, key, content_type="text/html", public_read=public_read)
    print(f"  Report  → s3://{bucket}/{key}")
    return url
