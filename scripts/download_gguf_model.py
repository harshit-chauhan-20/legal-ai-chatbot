"""Download small Q4_K_M instruct/chat GGUF (TinyLlama) to D: or project models dir."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from huggingface_hub import hf_hub_download

REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def main() -> None:
    out_dir = Path(
        sys.argv[1] if len(sys.argv) > 1 else r"D:\rag_legal_run\models"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO,
        filename=FILENAME,
        local_dir=str(out_dir),
    )
    print(f"OK: {path}")


if __name__ == "__main__":
    main()
