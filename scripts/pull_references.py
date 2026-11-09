#!/usr/bin/env python3
"""Acquire and verify pinned external references without executing their code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Iterable
from urllib.request import urlopen


LFS_PREFIX = b"version https://git-lfs.github.com/spec/v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReferenceError(RuntimeError):
    """An actionable acquisition or verification failure."""


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ReferenceError(f"{path}: unsupported or missing schema_version")
    references = manifest.get("references")
    if not isinstance(references, list) or not references:
        raise ReferenceError(f"{path}: references must be a non-empty list")
    names = [item.get("name") for item in references]
    if any(not isinstance(name, str) or not name for name in names):
        raise ReferenceError(f"{path}: every reference needs a non-empty name")
    if len(names) != len(set(names)):
        raise ReferenceError(f"{path}: duplicate reference names")
    return manifest


def select_references(
    manifest: dict[str, Any], subset: str | None
) -> list[dict[str, Any]]:
    references = manifest["references"]
    by_name = {item["name"]: item for item in references}
    if subset is None:
        return [item for item in references if not item.get("optional", False)]

    names = [name.strip() for name in subset.split(",") if name.strip()]
    if not names:
        raise ReferenceError("--subset requires at least one comma-separated name")
    if len(names) != len(set(names)):
        raise ReferenceError("--subset contains a duplicate name")
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ReferenceError(
            f"unknown reference(s): {', '.join(unknown)}; "
            f"available: {', '.join(sorted(by_name))}"
        )
    return [by_name[name] for name in names]


def rooted_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReferenceError(f"{field} must be a non-empty repository-relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise ReferenceError(f"{field} is not a safe relative path: {value}")
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*relative.parts)
    try:
        candidate.resolve(strict=False).relative_to(resolved_root)
    except ValueError as error:
        raise ReferenceError(f"{field} escapes the repository root: {value}") from error
    return candidate


def run_git(checkout: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(checkout), *arguments]
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise ReferenceError(f"could not execute git: {error}") from error
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise ReferenceError(
            f"git command failed (exit {result.returncode}): {' '.join(command)}\n{detail}"
        )
    return result


def checkout_head(checkout: Path) -> str | None:
    result = run_git(checkout, "rev-parse", "--verify", "HEAD", check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def require_clean_checkout(checkout: Path) -> None:
    status = run_git(
        checkout, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout
    if status:
        raise ReferenceError(
            f"dirty checkout preserved; clean or move it before acquisition: {checkout}\n"
            f"{status.rstrip()}"
        )


def prepare_checkout(reference: dict[str, Any], checkout: Path) -> None:
    url = reference.get("url")
    if not isinstance(url, str) or not url:
        raise ReferenceError(f"{reference['name']}: missing repository URL")

    if checkout.exists() and not checkout.is_dir():
        raise ReferenceError(f"checkout exists but is not a directory: {checkout}")
    if checkout.exists() and not (checkout / ".git").exists():
        if any(checkout.iterdir()):
            raise ReferenceError(
                f"non-git directory is not empty; refusing to overwrite: {checkout}"
            )
    if not checkout.exists():
        checkout.mkdir(parents=True, exist_ok=False)

    if not (checkout / ".git").exists():
        run_git(checkout.parent, "init", "--quiet", str(checkout))
        run_git(checkout, "remote", "add", "origin", url)
    else:
        inside = run_git(
            checkout, "rev-parse", "--is-inside-work-tree", check=False
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            raise ReferenceError(f"not a usable git checkout: {checkout}")
        remote = run_git(checkout, "remote", "get-url", "origin", check=False)
        if remote.returncode != 0 or remote.stdout.strip() != url:
            actual = remote.stdout.strip() if remote.returncode == 0 else "missing"
            raise ReferenceError(
                f"{reference['name']}: origin URL mismatch (expected {url}, got {actual})"
            )

    # Check cleanliness before either checkout or fetch can alter an existing clone.
    if checkout_head(checkout) is not None:
        require_clean_checkout(checkout)


def fetch_commit(checkout: Path, revision: str) -> None:
    # The revision, never HEAD or a branch name, is deliberately sent to the remote.
    run_git(
        checkout,
        "fetch",
        "--no-tags",
        "--depth",
        "1",
        "origin",
        revision,
    )


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(len(LFS_PREFIX)) == LFS_PREFIX


def find_lfs_pointers(checkout: Path) -> list[str]:
    pointers: list[str] = []
    for path in sorted(checkout.rglob("*")):
        relative = path.relative_to(checkout)
        if ".git" in relative.parts or path.is_symlink() or not path.is_file():
            continue
        if is_lfs_pointer(path):
            pointers.append(relative.as_posix())
    return pointers


def license_message(reference: dict[str, Any], checkout: Path) -> str:
    license_info = reference.get("license")
    if not isinstance(license_info, dict):
        raise ReferenceError(f"{reference['name']}: missing license evidence object")
    license_path = license_info.get("path")
    spdx = license_info.get("spdx", "missing/unclear")
    additional_paths = license_info.get("additional_paths", [])
    if not isinstance(additional_paths, list) or any(
        not isinstance(path, str) or not path for path in additional_paths
    ):
        raise ReferenceError(f"{reference['name']}: invalid license.additional_paths")
    if license_path is None:
        if additional_paths:
            raise ReferenceError(
                f"{reference['name']}: additional license paths require a primary license.path"
            )
        return f"license={spdx or 'missing/unclear'} evidence=none"
    if not isinstance(license_path, str) or not license_path:
        raise ReferenceError(f"{reference['name']}: invalid license.path")
    evidence_paths = [license_path, *additional_paths]
    for relative_path in evidence_paths:
        evidence = rooted_path(
            checkout, relative_path, f"{reference['name']}.license.path"
        )
        if not evidence.is_file():
            raise ReferenceError(
                f"{reference['name']}: declared license evidence is absent: {relative_path}"
            )
    return f"license={spdx} evidence={','.join(evidence_paths)}"


def validate_revision(reference: dict[str, Any]) -> str:
    revision = reference.get("revision")
    if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
        if reference.get("optional") and reference.get("status") == "not-acquired":
            raise ReferenceError(
                f"{reference['name']}: optional reference has no pin; "
                "select a source revision before requesting acquisition"
            )
        raise ReferenceError(
            f"{reference['name']}: revision must be an exact 40-character lowercase SHA"
        )
    return revision


def acquire_source(
    reference: dict[str, Any], checkout: Path, verify_only: bool
) -> str:
    revision = validate_revision(reference)
    if verify_only:
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise ReferenceError(f"missing checkout: {checkout}")
        remote = run_git(checkout, "remote", "get-url", "origin", check=False)
        if remote.returncode != 0 or remote.stdout.strip() != reference.get("url"):
            raise ReferenceError(f"{reference['name']}: missing or mismatched origin URL")
        require_clean_checkout(checkout)
        actual = checkout_head(checkout)
        if actual != revision:
            raise ReferenceError(
                f"{reference['name']}: expected {revision}, found {actual or 'no HEAD'}"
            )
    else:
        prepare_checkout(reference, checkout)
        actual = checkout_head(checkout)
        if actual != revision:
            present = run_git(
                checkout, "cat-file", "-e", f"{revision}^{{commit}}", check=False
            ).returncode == 0
            if not present:
                fetch_commit(checkout, revision)
            # Cleanliness was checked before network activity; check again before checkout.
            require_clean_checkout(checkout)
            run_git(checkout, "checkout", "--quiet", "--detach", revision)
        actual = checkout_head(checkout)
        if actual != revision:
            raise ReferenceError(
                f"{reference['name']}: post-acquisition revision is {actual or 'no HEAD'}"
            )
        require_clean_checkout(checkout)

    pointers = find_lfs_pointers(checkout)
    if pointers:
        raise ReferenceError(
            f"{reference['name']}: checkout contains Git LFS pointer(s), not payloads: "
            + ", ".join(pointers)
        )
    return revision


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(artifact: dict[str, Any], path: Path) -> str:
    name = artifact.get("name", path.name)
    if not path.is_file():
        raise ReferenceError(f"missing payload for {name}: {path}")
    if path.is_symlink():
        raise ReferenceError(f"payload for {name} must not be a symlink: {path}")
    if is_lfs_pointer(path):
        raise ReferenceError(f"payload for {name} is a Git LFS pointer: {path}")
    expected = artifact.get("sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ReferenceError(f"{name}: sha256 must be an exact lowercase SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise ReferenceError(
            f"payload checksum mismatch for {name}: expected {expected}, got {actual}"
        )
    expected_size = artifact.get("size_bytes")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ReferenceError(
            f"payload size mismatch for {name}: expected {expected_size}, "
            f"got {path.stat().st_size}"
        )
    return actual


def download_artifact(artifact: dict[str, Any], path: Path) -> None:
    url = artifact.get("url")
    if not isinstance(url, str) or not url:
        raise ReferenceError(f"{artifact.get('name', path.name)}: missing payload URL")
    expected = artifact.get("sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        raise ReferenceError(
            f"{artifact.get('name', path.name)}: refusing to download without a pinned SHA-256"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.download-{os.getpid()}")
    if temporary.exists():
        raise ReferenceError(f"stale download temporary preserved: {temporary}")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("xb") as stream:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                stream.write(block)
        verify_artifact(artifact, temporary)
        if path.exists():
            raise ReferenceError(f"payload appeared during download; refusing to overwrite: {path}")
        temporary.rename(path)
    except Exception as error:
        # Only remove the unique temporary file created by this invocation.
        if temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()
        if isinstance(error, ReferenceError):
            raise
        raise ReferenceError(
            f"payload download failed for {artifact.get('name', path.name)}: {error}"
        ) from error


def acquire_artifact(artifact: dict[str, Any], path: Path, verify_only: bool) -> str:
    if path.exists():
        # An invalid existing payload is never replaced implicitly.
        return verify_artifact(artifact, path)
    if verify_only:
        raise ReferenceError(f"missing payload in offline verification: {path}")
    download_artifact(artifact, path)
    return verify_artifact(artifact, path)


def process_references(
    references: Iterable[dict[str, Any]], root: Path, verify_only: bool
) -> int:
    failures = 0
    for reference in references:
        name = reference["name"]
        try:
            checkout = rooted_path(root, reference.get("checkout"), f"{name}.checkout")
            revision = acquire_source(reference, checkout, verify_only)
            mode = "verified" if verify_only else "acquired"
            report = [f"OK {name} revision={revision} checkout={checkout} status={mode}"]
            report.append(f"   {license_message(reference, checkout)}")
            artifacts = reference.get("artifacts", [])
            if not isinstance(artifacts, list):
                raise ReferenceError(f"{name}: artifacts must be a list")
            if not artifacts:
                report.append("   payloads=none-acquired")
            for artifact in artifacts:
                relative_path = artifact.get("path")
                path = rooted_path(root, relative_path, f"{name}.artifacts.path")
                digest = acquire_artifact(artifact, path, verify_only)
                report.append(f"   payload={path} sha256={digest}")
            reproduction = reference.get("reproduction", {})
            report.append(
                f"   reproduction={reproduction.get('status', 'missing/unclear')}"
            )
            print("\n".join(report))
        except (ReferenceError, OSError, json.JSONDecodeError) as error:
            failures += 1
            print(f"ERROR {name}: {error}", file=sys.stderr)
    return 1 if failures else 0


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        help="comma-separated reference names; defaults to all non-optional references",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="offline read-only verification; never fetch, checkout, or download",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_root / "configs" / "references.json",
        help="manifest path (default: configs/references.json)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="repository root for checkout and payload paths",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    try:
        root = args.root.resolve(strict=True)
        if not root.is_dir():
            raise ReferenceError(f"root is not a directory: {root}")
        manifest = load_manifest(args.manifest.resolve(strict=True))
        references = select_references(manifest, args.subset)
    except (OSError, ReferenceError, json.JSONDecodeError) as error:
        print(f"ERROR configuration: {error}", file=sys.stderr)
        return 2
    return process_references(references, root, args.verify_only)


if __name__ == "__main__":
    raise SystemExit(main())
