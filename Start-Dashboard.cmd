@echo off
REM ===========================================================================
REM  Start the local-llm stack: the model server, the API, and the dashboard.
REM
REM    Start-Dashboard.cmd          development - LIVE RELOAD on both halves
REM    Start-Dashboard.cmd prod     production  - faster, no reload
REM
REM  Development is the default because this is a tool you work on, and the
REM  earlier version of this script only ever ran production builds. That is
REM  why editing a component appeared to do nothing until the whole thing was
REM  rebuilt and restarted by hand: `next start` serves a compiled bundle and
REM  has no file watcher at all.
REM
REM  What "live reload" means on each side:
REM
REM    Dashboard  `next dev` enables Fast Refresh. Saving a .tsx re-renders the
REM               changed component in place, usually keeping component state,
REM               so a panel you are watching does not reset.
REM
REM    API        `uvicorn --reload` watches the Python source and restarts the
REM               server when it changes. State is NOT preserved - it is a
REM               process restart - but the browser reconnects on its own,
REM               because the dashboard's event stream retries automatically.
REM
REM  The trade is speed: `next dev` compiles each route on first request, so the
REM  first page load is noticeably slower and pages are not pre-optimised. Use
REM  `prod` when you want to see real performance rather than edit the code.
REM ===========================================================================

setlocal
set ROOT=%~dp0
set TOOLKIT=%ROOT%toolkit
set DASHBOARD=%ROOT%dashboard
set LMS=%USERPROFILE%\.lmstudio\bin\lms.exe

REM Default to dev; accept "prod" as the first argument.
set MODE=dev
if /i "%~1"=="prod" set MODE=prod

echo.
if "%MODE%"=="dev" (
    echo === MODE: development - live reload enabled ===
) else (
    echo === MODE: production - no live reload ===
)

echo.
echo === 1/3  Local model server ===
REM Called twice on purpose. From cold the first invocation reports
REM "Timed out waiting for LM Studio daemon to start" even with Bionic already
REM running, and the second succeeds immediately. Treating the first failure as
REM fatal would stop the script when nothing is actually wrong.
"%LMS%" server start >nul 2>&1
"%LMS%" server start
if errorlevel 1 (
    echo   WARNING: could not start the model server.
    echo   The dashboard will still run and will show "model server down".
)

REM Which Python starts the API.
REM
REM `py` is the launcher that a python.org install puts on Windows, and it is
REM tried first. The bare word `python` is only the fallback. On Windows that
REM word can be taken over by a stub that does nothing except offer to install
REM Python from the Microsoft Store (Settings, Apps, Advanced app settings, App
REM execution aliases). The stub prints one line and exits, so the API window
REM opened and closed at once and the dashboard showed "Cannot reach the API"
REM with nothing on screen to say why. An update of the Python installer can
REM switch the stub back on without asking, which is how this broke after
REM working for weeks.
REM
REM `where` prints the full path of a command if Windows can find it and sets
REM errorlevel 1 if it cannot. `&&` runs the second command only when the first
REM succeeded, so PY becomes `py` only where that launcher exists.
set PY=python
where py >nul 2>&1 && set PY=py

echo.
echo === 2/3  API on http://127.0.0.1:7878 ===
cd /d "%TOOLKIT%"
REM `start "title" /min` launches this in its own minimised window so the script
REM can continue. Without it the API would block here and the dashboard would
REM never start.
REM
REM --reload-dir limits the watcher to the package source. Pointed at the whole
REM working directory it would also watch .local-llm-data, where the live
REM progress files are rewritten several times a second during a generation -
REM so the API would restart continuously while the model was working.
if "%MODE%"=="dev" (
    start "local-llm API [reload]" /min cmd /c "%PY% -m uvicorn local_llm.api:app --port 7878 --reload --reload-dir src"
) else (
    start "local-llm API" /min cmd /c "%PY% -m uvicorn local_llm.api:app --port 7878"
)

REM The dashboard's first request fires as soon as the page loads, so give the
REM API a moment to bind its port. Otherwise the UI opens, fails its first
REM fetch, and shows "cannot reach the API" until the next refresh.
timeout /t 4 /nobreak >nul

echo.
echo === 3/3  Dashboard on http://localhost:3000 ===
cd /d "%DASHBOARD%"
if not exist "node_modules" (
    echo   node_modules missing - installing dependencies first...
    call npm install --no-audit --no-fund
)

REM Invoked through node rather than the npx shim: `npx next` launched via
REM `start` does not reliably attach on Windows, and fails silently when it does
REM not. Calling the binary directly is dependable.
if "%MODE%"=="dev" (
    start "local-llm dashboard [Fast Refresh]" /min cmd /c "node node_modules\next\dist\bin\next dev -p 3000"
    REM Dev compiles the first route on demand, so it needs longer before the
    REM page will answer than a production server does.
    timeout /t 9 /nobreak >nul
) else (
    if not exist ".next" (
        echo   No production build found - building...
        call npm run build
    )
    start "local-llm dashboard" /min cmd /c "node node_modules\next\dist\bin\next start -p 3000"
    timeout /t 5 /nobreak >nul
)

start "" http://localhost:3000

echo.
echo Running. Close the two minimised windows to stop the API and dashboard.
echo   API       http://127.0.0.1:7878
echo   Dashboard http://localhost:3000
if "%MODE%"=="dev" (
    echo.
    echo Live reload is ON:
    echo   - editing dashboard\**\*.tsx  re-renders in place
    echo   - editing toolkit\src\**\*.py restarts the API
)
echo.
endlocal
