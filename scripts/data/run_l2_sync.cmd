@echo off
REM Weekly puller for the Polymarket L2 collector output (GPUHub -> local).
REM Registered as Windows Scheduled Task "PolymarketL2WeeklySync". Sets cwd to the
REM project root so the sync script resolves .env (GPUHUB_PASSWORD) and its relative
REM LOCAL_DIR (data/polymarket_updown/l2) correctly. Appends timestamped output to a
REM gitignored log under data/.
cd /d C:\FinRL\FinRL-Pro_DS
echo ==== sync run %DATE% %TIME% ==== >> data\polymarket_updown\l2_sync.log
"~\AppData\Local\Programs\Python\Python311\python.exe" scripts\data\sync_polymarket_l2_from_remote.py --instance gpuhub-2 >> data\polymarket_updown\l2_sync.log 2>&1
echo ==== exit=%ERRORLEVEL% ==== >> data\polymarket_updown\l2_sync.log
