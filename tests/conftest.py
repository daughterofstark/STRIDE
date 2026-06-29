import os, pytest

def _find_v5():
    env = os.environ.get("MECHANISM_V5_SOURCE")
    cands = ([env] if env else []) + [
        os.path.join(os.path.dirname(__file__), "..", "reference", "v5_final_code_piece.py"),
        "final_code_piece.py", "../final_code_piece.py",
    ]
    for c in cands:
        if c and os.path.exists(c):
            return os.path.abspath(c)
    return None

@pytest.fixture(scope="session")
def v5_source_path():
    p = _find_v5()
    if p is None:
        pytest.skip("v5 source not found; set MECHANISM_V5_SOURCE")
    return p

@pytest.fixture(scope="session")
def golden_dir():
    d = os.path.join(os.path.dirname(__file__), "golden")
    if not any(os.scandir(d)) if os.path.isdir(d) else True:
        pytest.skip("no golden fixtures; run scripts/capture_golden.py on a machine with data+POVME")
    return d
