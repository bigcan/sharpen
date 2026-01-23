import threading
import time
import os
import signal
import sys
import logging

class TrainingWatchdog:
    """
    Monitors the training loop for liveness.
    If the training step does not advance within the timeout period,
    it assumes a deadlock and forces the process to terminate.
    """
    def __init__(self, callback_get_step, timeout_seconds=300, check_interval=30):
        """
        Args:
            callback_get_step: Function returning the current step (int)
            timeout_seconds: Max seconds allowed without step increment
            check_interval: How often to check
        """
        self.get_step = callback_get_step
        self.timeout = timeout_seconds
        self.interval = check_interval
        self._stop_event = threading.Event()
        self._thread = None
        self._last_step = -1
        self._last_change_time = time.time()
        self.logger = logging.getLogger("TrainingWatchdog")

    def start(self):
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"TrainingWatchdog started. Timeout: {self.timeout}s")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            current_step = self.get_step()
            now = time.time()

            # Initialize on first check effectively
            if self._last_step == -1:
                 self._last_step = current_step
                 self._last_change_time = now

            if current_step > self._last_step:
                self._last_step = current_step
                self._last_change_time = now
            else:
                elapsed = now - self._last_change_time
                if elapsed > self.timeout:
                    self._trigger_timeout(elapsed, current_step)

            time.sleep(self.interval)

    def _trigger_timeout(self, elapsed, step):
        msg = f"CRITICAL: Training HANG detected! Step {step} stalled for {elapsed:.1f}s (Timeout: {self.timeout}s). Terminating process."
        print(msg, file=sys.stderr)
        self.logger.critical(msg)
        
        # Flush logs
        sys.stderr.flush()
        sys.stdout.flush()
        
        # Force Kill
        # Try graceful exit first? No, deadlock means likely hung IO. Hard Kill.
        try:
             os.kill(os.getpid(), signal.SIGTERM)
             time.sleep(5)
             os.kill(os.getpid(), signal.SIGKILL)
        except Exception as e:
             # Force exit via sys
             print(f"Watchdog kill failed: {e}. using sys.exit", file=sys.stderr)
             os._exit(1)
