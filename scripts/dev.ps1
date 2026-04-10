param(
    [Parameter(Position = 0)]
    [ValidateSet("setup", "test", "app", "api", "tiers", "help")]
    [string]$Task = "help"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

function Require-Python {
    if (-not (Test-Path $python)) {
        throw "Virtual environment not found. Run: python -m venv .venv"
    }
}

switch ($Task) {
    "setup" {
        Push-Location $root
        try {
            python -m venv .venv
            & $python -m pip install -r requirements.txt
        }
        finally {
            Pop-Location
        }
    }
    "test" {
        Require-Python
        Push-Location $root
        try {
            & $python -m pytest tests\ -q
        }
        finally {
            Pop-Location
        }
    }
    "app" {
        Require-Python
        Push-Location $root
        try {
            & $python -m streamlit run app.py
        }
        finally {
            Pop-Location
        }
    }
    "api" {
        Require-Python
        Push-Location $root
        try {
            & $python -m deuce.api.server
        }
        finally {
            Pop-Location
        }
    }
    "tiers" {
        Require-Python
        Push-Location $root
        try {
            & $python -m deuce.cli feature-tiers
        }
        finally {
            Pop-Location
        }
    }
    default {
        Write-Host "Usage:"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 setup"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 test"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 app"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 api"
        Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1 tiers"
    }
}
