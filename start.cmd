@echo off
rem Prepares what is missing, then starts Dance Studio Bricks.
chcp 65001 >nul
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup\bootstrap.ps1"
if errorlevel 1 (
  pause
  exit /b 1
)
cd /d "%~dp0electron"
call npm start
