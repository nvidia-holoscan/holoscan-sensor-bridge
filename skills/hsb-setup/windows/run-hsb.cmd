@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0
setlocal
if "%~1"=="" (
    echo Usage: run-hsb.cmd ^<profile^>
    echo.
    echo Available profiles:
    powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0set-hsb-env.ps1'"
    exit /b 1
)
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ". '%~dp0set-hsb-env.ps1' -Profile '%~1'; claude"
endlocal
