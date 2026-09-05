"""Fetch and verify the large artifacts that are not stored in git.

See ARTIFACTS.md. Set ARTIFACT_BASE (below, or the ARTIFACT_BASE environment
variable) to a stable public HTTPS directory containing the four files under
their exact names.

  python scripts/fetch_artifacts.py           # download whatever is missing
  python scripts/fetch_artifacts.py --check   # verify only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# SET THIS ONCE. Everything else picks it up. See ARTIFACTS.md > Hosting.
ARTIFACT_BASE = os.environ.get(
    "ARTIFACT_BASE",
    "https://github.com/YOUR-GITHUB-USERNAME/street-signatures"
    "/releases/download/v1.0-artifacts",
)
# ---------------------------------------------------------------------------

ARTIFACTS = {
    "checkpoints/runA_best.pt": (
        94528075,
        "8656994a6e715947a59c378f38d9d59e7c5fb88e0297eb3c4cc9d4603a62c0a6"),
    "data/interim/emb_trained.npz": (
        456236109,
        "f4b78ce8f24a510c95e1b56fbd4c9fa424aff4c3fad93fff4dcabd9d4595e540"),
    "data/interim/emb_imagenet.npz": (
        341480561,
        "c9ace847611df97647d33c45ae58f80f823baf1c9dc30a475bfc54e0f81152c4"),
    "data/interim/emb_s11_vicreg_a_ep100.npz": (
        389033061,
        "5f6c1ac09f0d1e178df12381dfcf10f823be2ffe1812ae8720741e01d2f0e201"),
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: Path, size: int, digest: str) -> bool:
    return path.exists() and path.stat().st_size == size and sha256(path) == digest


def fetch(rel: str, size: int, digest: str, base: str) -> bool:
    dest = Path(rel)
    if dest.exists():
        if verify(dest, size, digest):
            print(f"  ok       {rel}")
            return True
        print(f"  MISMATCH {rel} — re-downloading")
    if "YOUR-GITHUB-USERNAME" in base:
        print(f"  MISSING  {rel}")
        print("           ARTIFACT_BASE is still a placeholder — see ARTIFACTS.md")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base.rstrip('/')}/{dest.name}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetching {rel}  ({size/1e6:.0f} MB)")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:                                   # noqa: BLE001
        print(f"  FAILED   {rel}: {type(e).__name__}: {e}")
        tmp.unlink(missing_ok=True)
        return False
    if sha256(tmp) != digest:
        print(f"  FAILED   {rel}: checksum mismatch after download")
        print("           (a Drive-style HTML interstitial will land here)")
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    print(f"  ok       {rel}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--base", default=ARTIFACT_BASE)
    args = ap.parse_args()

    ok = True
    for rel, (size, digest) in ARTIFACTS.items():
        if args.check:
            good = verify(Path(rel), size, digest)
            print(f"  {'ok      ' if good else 'BAD/MISS'} {rel}")
            ok &= good
        else:
            ok &= fetch(rel, size, digest, args.base)
    if not ok:
        print("\nSome artifacts are missing or unverified — see ARTIFACTS.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
