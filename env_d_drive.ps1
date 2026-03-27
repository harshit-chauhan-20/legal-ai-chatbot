# Run before pip / python / streamlit so caches and temp files stay on D: (not C:).
$Root = "D:\rag_legal_run"
$env:TMP = "$Root\tmp"
$env:TEMP = "$Root\tmp"
$env:PIP_CACHE_DIR = "$Root\.cache\pip"
$env:HF_HOME = "$Root\.cache\huggingface"
$env:HF_HUB_CACHE = "$Root\.cache\huggingface\hub"
$env:HUGGINGFACE_HUB_CACHE = "$Root\.cache\huggingface\hub"
# Ollama model blobs (when you install Ollama; point here so downloads stay on D:)
$env:OLLAMA_MODELS = "$Root\ollama_models"

Write-Host "D: drive env active. TMP/TEMP/PIP/HF/Ollama models -> $Root"
