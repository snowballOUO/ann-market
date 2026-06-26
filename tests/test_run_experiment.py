from scripts import run_experiment


def test_progress_iter_uses_dynamic_single_line_tqdm(monkeypatch):
    calls = {}

    def fake_tqdm(iterable, **kwargs):
        calls.update(kwargs)
        return iterable

    monkeypatch.setattr(run_experiment, "tqdm", fake_tqdm)

    assert list(run_experiment.progress_iter(10000))[:3] == [0, 1, 2]
    assert calls["dynamic_ncols"] is True
    assert calls["mininterval"] == 0.5
    assert calls["leave"] is True
