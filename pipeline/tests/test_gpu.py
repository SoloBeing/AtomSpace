import pytest

from atomspace import gpu


@pytest.mark.parametrize(
    ("vram", "profile"), [(None, "vram_8gb"), (4.0, "vram_4gb"), (7.9, "vram_8gb"), (12.0, "vram_12gb"), (24.0, "vram_24gb")]
)
def test_profile_follows_detected_vram(monkeypatch, vram, profile):
    monkeypatch.setattr(gpu, "vram_gib", lambda: vram)
    assert gpu.resolve_profile().name == profile


def test_below_the_recommended_size_warns_but_runs(monkeypatch, capsys):
    monkeypatch.setattr(gpu, "vram_gib", lambda: 4.0)
    gpu.resolve_profile()
    gpu.require_gpu()
    assert "below the recommended 8 GB" in capsys.readouterr().out


def test_below_the_minimum_is_refused(monkeypatch):
    monkeypatch.setattr(gpu, "vram_gib", lambda: 2.0)
    with pytest.raises(RuntimeError, match="at least 4 GB"):
        gpu.resolve_profile()
    with pytest.raises(RuntimeError, match="at least 4 GB"):
        gpu.require_gpu()


def test_gpu_stages_refuse_a_machine_without_a_gpu(monkeypatch):
    monkeypatch.setattr(gpu, "vram_gib", lambda: None)
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        gpu.require_gpu()
