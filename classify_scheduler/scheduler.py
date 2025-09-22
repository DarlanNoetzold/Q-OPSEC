from typing import Callable
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

def schedule_periodic_training(cron_expr: str, cfg, job_fn: Callable):
    sched = BackgroundScheduler()
    trigger = CronTrigger.from_crontab(cron_expr)
    sched.add_job(job_fn, trigger=trigger, args=[cfg], id="periodic_training", replace_existing=True)
    sched.start()
    print(f"[SCHEDULER] Started with CRON '{cron_expr}'. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("[SCHEDULER] Stopping...")
        sched.shutdown()