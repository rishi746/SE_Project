$ErrorActionPreference = 'Stop'
$projectDirectory = $PSScriptRoot
$projectPython = Join-Path $projectDirectory '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Set up the CPU environment using README.md first. No model runs on this laptop.'
}
Set-Location -LiteralPath $projectDirectory
& $projectPython -m streamlit run (Join-Path $projectDirectory 'app.py') --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
