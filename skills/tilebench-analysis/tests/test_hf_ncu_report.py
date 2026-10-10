import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hf_ncu_report.py"
spec = importlib.util.spec_from_file_location("hf_ncu_report", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.data = b"fake report bytes"
        self.path = "NVIDIA_B200/streamk_matmul/tilelang_fp16.ncu-rep"
        self.calls = []
        self.entry = SimpleNamespace(path=self.path, size=len(self.data),
                                     lfs=SimpleNamespace(sha256=hashlib.sha256(self.data).hexdigest()))
        self.api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha="fixed-commit"),
                                   get_paths_info=self.get_paths)

    def get_paths(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [self.entry]

    def download(self, repo, path, **kwargs):
        self.calls.append(((repo, path), kwargs))
        target = Path(kwargs["local_dir"]) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.data)
        return str(target)

    def test_pinned_single_file_and_receipt(self):
        with TemporaryDirectory() as directory:
            local, receipt = module.fetch(self.api, self.download, "B200", "streamk_matmul",
                                           "tilelang", "fp16", "main", directory)
            self.assertEqual(local.read_bytes(), self.data)
            self.assertTrue(receipt["remote_hash_verified"])
            self.assertEqual(receipt["revision"], "fixed-commit")
            self.assertTrue(all(k["revision"] == "fixed-commit" for a, k in self.calls))
            self.assertEqual(self.calls[1][0][1], self.path)
            self.assertEqual(json.loads((Path(directory) / "download.json").read_text()), receipt)

    def test_missing_report_does_not_download(self):
        self.api.get_paths_info = lambda *a, **k: []
        with TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                module.fetch(self.api, self.download, "B200", "streamk_matmul", "tilelang", "fp16", "main", directory)
            self.assertEqual(self.calls, [])

    def test_checksum_failure_has_no_verified_receipt(self):
        self.entry.lfs.sha256 = "wrong"
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                module.fetch(self.api, self.download, "B200", "streamk_matmul", "tilelang", "fp16", "main", directory)
            self.assertFalse((Path(directory) / "download.json").exists())

    def test_refuse_cross_case_or_revision_reuse(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "download.json").write_text(json.dumps({"dataset": module.REPO,
                "revision": "old", "remote_path": self.path}))
            with self.assertRaisesRegex(ValueError, "another case/revision"):
                module.fetch(self.api, self.download, "B200", "streamk_matmul", "tilelang", "fp16", "main", directory)
            self.assertEqual(self.calls, [])

    def test_platform_alias_and_path_validation(self):
        self.assertEqual(module.remote_path("GH200", "matmul_fp32_fp16_fp8", "triton", "fp8_e4m3fn"),
                         "NVIDIA_GH200/matmul_fp32_fp16_fp8/triton_fp8_e4m3fn.ncu-rep")
        self.assertEqual(module.remote_path("NVIDIA_B200", "streamk_matmul", "tilelang", "fp16"), self.path)
        for hardware, operator in [("MI300X", "argmax"), ("B200", "../argmax")]:
            with self.assertRaises(ValueError):
                module.remote_path(hardware, operator, "triton", "fp16")


if __name__ == "__main__":
    unittest.main()
