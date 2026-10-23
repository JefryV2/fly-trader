# Start the FlyTrader dashboard from the repository root.
Set-Location -Path $PSScriptRoot
python -m uvicorn web-app.server:app --host 127.0.0.1 --port 8000
