@echo off
setlocal
cd /d "%~dp0"
echo Existing human annotations are preserved. Optional screening.json only changes the review queue.
if exist ".venv-sitting-posture\Scripts\python.exe" (
    ".venv-sitting-posture\Scripts\python.exe" -X utf8 "scripts\review_candidates.py" --open-browser %*
) else (
    python -X utf8 "scripts\review_candidates.py" --open-browser %*
)
set "review_exit=%errorlevel%"
echo.
echo Review server stopped. Any error is shown above.
pause
exit /b %review_exit%
