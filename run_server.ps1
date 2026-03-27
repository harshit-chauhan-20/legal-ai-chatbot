# Start Streamlit with D: caches (no large writes to C:).
Set-Location $PSScriptRoot
. "$PSScriptRoot\env_d_drive.ps1"
& "$PSScriptRoot\.venv\Scripts\Activate.ps1"
$env:STREAMLIT_BROWSER = "true"
streamlit run app.py --server.headless false
