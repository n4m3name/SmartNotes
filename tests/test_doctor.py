from smartnotes.doctor import run_checks


def test_doctor_basic(run_async):
    res = run_async(run_checks())
    assert 'config.notes_dir' in res.details
    assert 'db.tables' in res.details
    assert 'vecstore.dir' in res.details
    # OK may be true or false depending on fresh env, but result should be structured
    assert isinstance(res.warnings, list)
    assert isinstance(res.errors, list)
