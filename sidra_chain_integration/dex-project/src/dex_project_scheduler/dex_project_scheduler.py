# dex_project_scheduler.py
import schedule
import time

class DexProjectScheduler:
    def __init__(self):
        pass

    def schedule_sync(self, sync_function):
        # Schedule sync function to run every 1 minute
        schedule.every(1).minutes.do(sync_function)

    def run_scheduler(self):
        # Run the scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)
