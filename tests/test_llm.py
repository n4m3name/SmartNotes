from smartnotes.llm.factory import get_provider


def test_llm_local_provider_default(monkeypatch):
    # Ensure remote_allowed is false -> forces local
    from smartnotes import config as _cfg
    s = _cfg.load_settings()
    monkeypatch.setenv("SMARTNOTES_CONFIG", str(_cfg._resolve_config_path(None)))

    prov = get_provider()
    summary = prov.generate_summary("Hello world. This is a test! More text?", max_sentences=2)
    assert "Hello world." in summary
    tags = prov.generate_tags("Alpha beta beta gamma gamma gamma")
    assert tags[0] in {"gamma", "beta"}
