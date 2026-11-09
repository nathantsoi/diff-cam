"""Tests for pinned reference acquisition; all remotes and payloads are local/mocked."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pull_references.py"
SPEC = importlib.util.spec_from_file_location("diffcam_pull_references", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pull = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pull)


class ReferenceAcquisitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="diffcam-references-test-")
        self.root = Path(self.temporary.name)
        self.origin = self.root / "origin"
        self.origin.mkdir()
        (self.origin / "README.md").write_text("original\n", encoding="utf-8")
        (self.origin / "LICENSE").write_text("test license\n", encoding="utf-8")
        self.git(self.origin, "init", "--quiet")
        self.revision = self.commit_origin()
        self.manifest_path = self.root / "references.json"
        self.reference = {
            "name": "local-reference",
            "url": self.origin.as_uri(),
            "revision": self.revision,
            "checkout": "external/local-reference",
            "optional": False,
            "status": "not-acquired",
            "license": {"path": "LICENSE", "spdx": "BSD-3-Clause"},
            "artifacts": [],
            "reproduction": {"status": "not-attempted"},
        }
        self.optional = {
            "name": "optional-reference",
            "url": "https://example.invalid/optional.git",
            "revision": None,
            "checkout": "external/optional-reference",
            "optional": True,
            "status": "not-acquired",
            "license": {"path": None, "spdx": "missing/unclear"},
            "artifacts": [],
            "reproduction": {"status": "not-attempted"},
        }
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def git(checkout: Path, *arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"test git command failed: {arguments}\n{result.stdout}{result.stderr}"
            )
        return result.stdout.strip()

    def commit_origin(self) -> str:
        self.git(self.origin, "add", "--all")
        self.git(
            self.origin,
            "-c",
            "user.name=DiffCAM Tests",
            "-c",
            "user.email=diffcam-tests@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture revision",
        )
        return self.git(self.origin, "rev-parse", "HEAD")

    def write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "references": [self.reference, self.optional],
                }
            ),
            encoding="utf-8",
        )

    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = pull.main(
                [
                    "--root",
                    str(self.root),
                    "--manifest",
                    str(self.manifest_path),
                    *arguments,
                ]
            )
        return result, stdout.getvalue(), stderr.getvalue()

    @property
    def checkout(self) -> Path:
        return self.root / "external" / "local-reference"

    def head(self) -> str:
        return self.git(self.checkout, "rev-parse", "HEAD")

    def add_artifact(self, payload: bytes = b"real payload\n") -> dict[str, object]:
        artifact: dict[str, object] = {
            "name": "local-payload",
            "path": "data/external/local-reference/payload.bin",
            "url": "https://payload.invalid/payload.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        self.reference["artifacts"] = [artifact]
        self.write_manifest()
        return artifact

    def test_rerun_keeps_pin_when_origin_advances_and_offline_verifies(self) -> None:
        code, _, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertEqual(self.head(), self.revision)

        (self.origin / "README.md").write_text("moving upstream\n", encoding="utf-8")
        newer_revision = self.commit_origin()
        self.assertNotEqual(newer_revision, self.revision)

        # A same-pin rerun must not consult the remote at all.
        with mock.patch.object(
            pull, "fetch_commit", side_effect=AssertionError("unexpected fetch")
        ), mock.patch.object(pull, "urlopen", side_effect=AssertionError("unexpected URL access")):
            code, _, errors = self.invoke()
            self.assertEqual(code, 0, errors)
            code, _, errors = self.invoke("--verify-only")
            self.assertEqual(code, 0, errors)
        self.assertEqual(self.head(), self.revision)
        self.assertEqual(
            (self.checkout / "README.md").read_text(encoding="utf-8"), "original\n"
        )

    def test_missing_checkout_fails_offline_without_fetching(self) -> None:
        with mock.patch.object(
            pull, "fetch_commit", side_effect=AssertionError("unexpected fetch")
        ):
            code, _, errors = self.invoke("--verify-only")
        self.assertEqual(code, 1)
        self.assertIn("missing checkout", errors)
        self.assertFalse(self.checkout.exists())

    def test_network_failure_is_explicit_and_partial_clone_can_resume(self) -> None:
        with mock.patch.object(
            pull,
            "fetch_commit",
            side_effect=pull.ReferenceError("simulated network unavailable"),
        ):
            code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("simulated network unavailable", errors)
        self.assertFalse((self.checkout / "README.md").exists())

        code, output, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        self.assertIn(self.revision, output)
        self.assertEqual(self.head(), self.revision)

    def test_dirty_checkout_is_preserved_and_blocks_acquisition(self) -> None:
        code, _, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        readme = self.checkout / "README.md"
        readme.write_text("local work must survive\n", encoding="utf-8")

        with mock.patch.object(
            pull, "fetch_commit", side_effect=AssertionError("unexpected fetch")
        ):
            code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("dirty checkout preserved", errors)
        self.assertEqual(readme.read_text(encoding="utf-8"), "local work must survive\n")
        self.assertEqual(self.head(), self.revision)

    def test_verify_only_never_repairs_wrong_revision(self) -> None:
        code, _, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        (self.origin / "README.md").write_text("new upstream\n", encoding="utf-8")
        newer_revision = self.commit_origin()
        self.git(self.checkout, "fetch", "--quiet", "origin", newer_revision)
        self.git(self.checkout, "checkout", "--quiet", "--detach", newer_revision)

        with mock.patch.object(
            pull, "fetch_commit", side_effect=AssertionError("unexpected fetch")
        ):
            code, _, errors = self.invoke("--verify-only")
        self.assertEqual(code, 1)
        self.assertIn("expected", errors)
        self.assertEqual(self.head(), newer_revision)

    def test_checked_payload_download_then_offline_verification(self) -> None:
        payload = b"not an LFS pointer\x00\x01\n"
        artifact = self.add_artifact(payload)
        with mock.patch.object(pull, "urlopen", return_value=io.BytesIO(payload)) as opener:
            code, output, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        opener.assert_called_once()
        path = self.root / str(artifact["path"])
        self.assertEqual(path.read_bytes(), payload)
        self.assertIn(str(artifact["sha256"]), output)

        with mock.patch.object(
            pull, "urlopen", side_effect=AssertionError("unexpected network access")
        ), mock.patch.object(
            pull, "fetch_commit", side_effect=AssertionError("unexpected fetch")
        ):
            code, _, errors = self.invoke("--verify-only")
        self.assertEqual(code, 0, errors)

    def test_missing_payload_fails_offline(self) -> None:
        code, _, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        self.add_artifact()
        with mock.patch.object(
            pull, "urlopen", side_effect=AssertionError("unexpected network access")
        ):
            code, _, errors = self.invoke("--verify-only")
        self.assertEqual(code, 1)
        self.assertIn("missing payload in offline verification", errors)

    def test_corrupt_existing_payload_is_not_overwritten(self) -> None:
        artifact = self.add_artifact(b"expected payload\n")
        path = self.root / str(artifact["path"])
        path.parent.mkdir(parents=True)
        path.write_bytes(b"locally modified payload\n")
        with mock.patch.object(
            pull, "urlopen", side_effect=AssertionError("unexpected network access")
        ):
            code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("payload checksum mismatch", errors)
        self.assertEqual(path.read_bytes(), b"locally modified payload\n")

    def test_lfs_pointer_is_rejected_even_with_its_matching_hash(self) -> None:
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\n"
            b"oid sha256:" + b"a" * 64 + b"\nsize 12345\n"
        )
        artifact = self.add_artifact(pointer)
        path = self.root / str(artifact["path"])
        path.parent.mkdir(parents=True)
        path.write_bytes(pointer)
        code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("Git LFS pointer", errors)
        self.assertEqual(path.read_bytes(), pointer)

    def test_optional_entries_are_skipped_by_default_and_explicitly_unpinned(self) -> None:
        manifest = pull.load_manifest(self.manifest_path)
        selected = pull.select_references(manifest, None)
        self.assertEqual([item["name"] for item in selected], ["local-reference"])
        code, _, errors = self.invoke("--subset", "optional-reference")
        self.assertEqual(code, 1)
        self.assertIn("optional reference has no pin", errors)
        self.assertFalse((self.root / "external" / "optional-reference").exists())

    def test_missing_declared_license_evidence_is_explicit(self) -> None:
        (self.origin / "LICENSE").unlink()
        self.origin.joinpath("README.md").write_text("without license\n", encoding="utf-8")
        self.reference["revision"] = self.commit_origin()
        self.write_manifest()
        code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("declared license evidence is absent", errors)


if __name__ == "__main__":
    unittest.main()
