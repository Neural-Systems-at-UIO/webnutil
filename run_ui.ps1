

try {
    Write-Host "Activating venv" -ForegroundColor Green
    .\.venv\Scripts\Activate.ps1
} catch {
    Write-Information "Couldn't find a venv, create a new one"
    python -m venv .\.venv
    .\.venv\Scripts\Activate.ps1
    pip install -r .\ui\requirements.txt
}

try {
    Write-Host "Launching the UI" -ForegroundColor Green
    python .\ui\app.py
} catch {
    Write-Information "Ran into an issue {$($_.Exception.Message)}"
}


