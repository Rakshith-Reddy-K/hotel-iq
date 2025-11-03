from pathlib import Path

def _ensure_output_dir(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _resolve_project_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    project_root = Path(__file__).resolve().parents[2]
    return str((project_root / path).resolve())
