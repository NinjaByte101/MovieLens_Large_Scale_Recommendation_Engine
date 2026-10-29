"""Publish step: put the new Triton model repository live and clear stale cached recommendations.

Cached lists in Redis were produced by the OLD models, so they are deleted after the swap.
The swap itself is two renames (old -> .old, new -> live), so a reader never sees a half-copied
repository. Triton should run with --model-control-mode=poll to notice the new files.

Run:
    python src/pipeline/publish.py --src outputs/model_repository --live live/model_repository \
        --redis localhost:6379
"""
import argparse
import shutil
from pathlib import Path


def swap_repository(src: Path, live: Path):
    staging = live.with_name(live.name + ".new")
    previous = live.with_name(live.name + ".old")
    live.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(src, staging)                 # full copy first, so `live` is never half-written
    shutil.rmtree(previous, ignore_errors=True)
    if live.exists():
        live.rename(previous)                     # keep the last good version for a quick rollback
    staging.rename(live)


def clear_cache(address, prefix):
    """Delete cached recommendations. Returns how many keys were removed (None if Redis is unreachable)."""
    import redis
    host, port = address.split(":")
    client = redis.Redis(host=host, port=int(port), decode_responses=True)
    try:
        client.ping()
    except Exception as exc:
        print(f"Redis not reachable ({exc}); cache NOT cleared")
        return None
    removed = 0
    for key in client.scan_iter(f"{prefix}:*"):
        removed += client.delete(key)
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="freshly exported model repository")
    ap.add_argument("--live", type=Path, required=True, help="directory Triton serves from")
    ap.add_argument("--redis", default="", help="host:port, optional")
    ap.add_argument("--prefix", default="rec:v1")
    args = ap.parse_args()

    swap_repository(args.src, args.live)
    print(f"published {args.src} -> {args.live} (previous version kept as {args.live.name}.old)")
    if args.redis:
        removed = clear_cache(args.redis, args.prefix)
        if removed is not None:
            print(f"cleared {removed} cached recommendation lists")


if __name__ == "__main__":
    main()
