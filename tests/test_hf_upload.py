"""scripts/profiling/hf_upload.py: --gpu names the local source, --hf-folder the
dataset destination, and neither is derived from the other.

HfApi is replaced by a recorder, so nothing here talks to Hugging Face.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

import tilebench.paths as paths

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "profiling" / "hf_upload.py"


def load():
    spec = importlib.util.spec_from_file_location("_hf_upload_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RecordingApi:
    instances = []

    def __init__(self, token=None):
        self.token, self.calls = token, []
        RecordingApi.instances.append(self)

    def create_repo(self, *args, **kwargs):
        self.calls.append(("create_repo", args, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", (), kwargs))
        return "ok"


@pytest.fixture
def uploader(tmp_path, monkeypatch):
    """Run main() with argv; outputs/ncu/ is a temporary tree holding GH200 and B200 reports."""
    ncu = tmp_path / "outputs" / "ncu"
    for gpu in ("GH200", "B200", "TESTGPU"):
        (ncu / gpu / "1d_conv").mkdir(parents=True)
        (ncu / gpu / "1d_conv" / "triton_fp16.ncu-rep").write_bytes(b"rep")
    monkeypatch.setattr(paths, "NCU_OUTPUT_ROOT", ncu)
    monkeypatch.setenv("HUGGING_FACE", "hf_test_token")
    RecordingApi.instances = []
    mod = load()
    monkeypatch.setattr(mod, "HfApi", RecordingApi)

    def run(*argv):
        monkeypatch.setattr(sys, "argv", ["hf_upload.py", *argv])
        mod.main()
        (api,) = RecordingApi.instances
        (upload,) = [kw for name, _, kw in api.calls if name == "upload_folder"]
        return upload
    run.ncu = ncu
    run.module = mod
    return run


def test_hf_folder_is_required(uploader, capsys):
    with pytest.raises(SystemExit) as e:
        uploader("--gpu", "GH200")
    assert e.value.code == 2 and "--hf-folder" in capsys.readouterr().err
    assert RecordingApi.instances == []                       # nothing contacted


def test_gh200_all_operators(uploader):
    up = uploader("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200")
    assert up["folder_path"] == str(uploader.ncu / "GH200")
    assert up["path_in_repo"] == "NVIDIA_GH200"
    assert up["repo_id"] == "bcui2/NCU_report" and up["repo_type"] == "dataset"


def test_gh200_one_operator(uploader):
    up = uploader("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200", "1d_conv")
    assert up["folder_path"] == str(uploader.ncu / "GH200" / "1d_conv")
    assert up["path_in_repo"] == "NVIDIA_GH200/1d_conv"


def test_b200_reads_the_local_label_not_the_remote_folder(uploader):
    up = uploader("--gpu", "B200", "--hf-folder", "NVIDIA_B200")
    assert up["folder_path"] == str(uploader.ncu / "B200")
    assert not (uploader.ncu / "NVIDIA_B200").exists()
    assert up["path_in_repo"] == "NVIDIA_B200"


def test_no_old_layout_is_ever_produced(uploader):
    for argv in (("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200"),
                 ("--gpu", "B200", "--hf-folder", "NVIDIA_B200", "1d_conv")):
        RecordingApi.instances = []
        path = uploader(*argv)["path_in_repo"]
        assert not path.startswith(("ncu_report_main", "GH200", "B200"))


@pytest.mark.parametrize("bad", ["", ".", "..", "../x", "a/b", "/tmp/x", "x/../y", ".hidden",
                                 "a\\b", "NVIDIA GH200"])
def test_an_unsafe_hf_folder_is_rejected(uploader, bad, capsys):
    with pytest.raises(SystemExit) as e:
        uploader("--gpu", "GH200", "--hf-folder", bad)
    assert e.value.code == 2 and "--hf-folder" in capsys.readouterr().err
    assert RecordingApi.instances == []


@pytest.mark.parametrize("bad", ["..", "../B200", "a/b", "/tmp/x"])
def test_the_operator_cannot_reach_outside_the_hf_folder(uploader, bad, capsys):
    with pytest.raises(SystemExit) as e:
        uploader("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200", bad)
    assert e.value.code == 2 and RecordingApi.instances == []


def test_there_is_no_gpu_to_folder_mapping(uploader):
    """Any valid local label with any valid remote folder: nothing is looked up."""
    up = uploader("--gpu", "TESTGPU", "--hf-folder", "SOME_REMOTE_NAMESPACE")
    assert up["folder_path"] == str(uploader.ncu / "TESTGPU")
    assert up["path_in_repo"] == "SOME_REMOTE_NAMESPACE"
    assert uploader.module.upload_plan("GH200", "AMD_MI300X", "op") == \
        (uploader.ncu / "GH200" / "op", "AMD_MI300X/op")       # names are taken as given


def test_only_ncu_reports_are_uploaded(uploader):
    up = uploader("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200")
    assert up["allow_patterns"] == ["*.ncu-rep", "**/*.ncu-rep"]


def test_a_missing_local_folder_stops_before_contacting_the_hub(uploader):
    with pytest.raises(SystemExit) as e:
        uploader("--gpu", "MI300X", "--hf-folder", "AMD_MI300X")
    assert "is not a directory" in str(e.value) and RecordingApi.instances == []


def test_a_missing_token_stops_before_contacting_the_hub(uploader, monkeypatch):
    monkeypatch.delenv("HUGGING_FACE")
    with pytest.raises(SystemExit) as e:
        uploader("--gpu", "GH200", "--hf-folder", "NVIDIA_GH200")
    assert "HUGGING_FACE" in str(e.value) and RecordingApi.instances == []


def test_importing_the_module_has_no_hub_side_effect(monkeypatch):
    import huggingface_hub
    RecordingApi.instances = []
    monkeypatch.setattr(huggingface_hub, "HfApi", RecordingApi)
    mod = load()
    assert mod.HfApi is RecordingApi and RecordingApi.instances == []
