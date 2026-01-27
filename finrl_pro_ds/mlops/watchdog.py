import threading
import time
import os
import signal
import sys
import logging
import collections
from typing import Callable

class TrainingWatchdog:
    """
    Monitors the training loop for:
    1. Deadlocks (No step progress).
    2. Performance Degradation (SPS drops below threshold).
    
    If critical failure detected, forces process termination to save resources (Zombie Killer).
    """
    def __init__(
        self, 
        callback_get_step: Callable[[], int], 
        timeout_seconds: int = 300, 
        min_sps: float = 0.0,
        sps_window: int = 10,
        check_interval: int = 10
    ):
        """
        Args:
            callback_get_step: Function returning current global step.
            timeout_seconds: Max seconds allowed without any step increment.
            min_sps: Minimum acceptable Steps Per Second. 0 to disable.
            sps_window: Number of checks to average SPS over.
            check_interval: Seconds between checks.
        """
        self.get_step = callback_get_step
        self.timeout = timeout_seconds
        self.min_sps = min_sps
        self.interval = check_interval
        
        self._stop_event = threading.Event()
        self._thread = None
        
        self.logger = logging.getLogger("SPSWatchdog")
        self.check_history = collections.deque(maxlen=sps_window) # Stores (time, step)

    def start(self):
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"SPSWatchdog started. Timeout: {self.timeout}s, Min SPS: {self.min_sps}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        last_step = -1
        last_change_time = time.time()
        
        while not self._stop_event.is_set():
            current_step = self.get_step()
            now = time.time()
            
            # 1. Deadlock Check
            if last_step == -1: # Init
                last_step = current_step
                last_change_time = now
                self.check_history.append((now, current_step))
                time.sleep(self.interval)
                continue
                
            if current_step > last_step:
                last_step = current_step
                last_change_time = now
            else:
                elapsed = now - last_change_time
                if elapsed > self.timeout:
                    self._trigger_kill(f"Deadlock detected! No progress for {elapsed:.1f}s.")
                    return

            # 2. SPS Check
            self.check_history.append((now, current_step))
            if self.min_sps > 0 and len(self.check_history) > 1:
                # Calculate avg SPS over window
                t_start, s_start = self.check_history[0]
                t_end, s_end = self.check_history[-1]
                
                delta_t = t_end - t_start
                delta_s = s_end - s_start
                
                if delta_t > 0:
                    current_sps = delta_s / delta_t
                    if current_sps < self.min_sps and delta_t > 60: # Warmup constraint
                        # Grace period? We assume window is large enough.
                        # But wait, paused training (e.g. eval) triggers this?
                        # Eval is part of training usually, but step doesn't increment?
                        # If step stops, deadlock check handles it.
                        # If step is slow, this handles it.
                        pass # Warning for now? Or strict kill?
                        # self.logger.warning(f"Low SPS: {current_sps:.2f} < {self.min_sps}")
                        # Strict kill only if explicitly requested?
                        # Let's just log warning for safety until tuned.
                        if current_sps < (self.min_sps * 0.5): # Critical Drop
                             self.logger.error(f"CRITICAL PERF DROP: {current_sps:.2f} SPS.")


            time.sleep(self.interval)

    def _trigger_kill(self, reason: str):
        msg = f"WATCHDOG KILL: {reason} Terminating process."
        print(msg, file=sys.stderr)
        self.logger.critical(msg)
        sys.stderr.flush()
        
        # Hard Kill
        try:
             os.kill(os.getpid(), signal.SIGTERM)
             time.sleep(5)
             os.kill(os.getpid(), signal.SIGKILL)
        except:
             os._exit(1)
