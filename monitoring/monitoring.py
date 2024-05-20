import time
import logging
import requests
import psutil

logging.basicConfig(filename='monitoring.log', level=logging.INFO)

def monitor_transactions():
    logging.info('Monitoring transactions...')
    # Add your monitoring code here

def monitor_system_performance():
    logging.info('Monitoring system performance...')
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent

    logging.info(f'CPU usage: {cpu_percent}%')
    logging.info(f'Memory usage: {memory_percent}%')
    logging.info(f'Disk usage: {disk_percent}%')

    # Send system performance data to a monitoring service
    url = 'https://your-monitoring-service.com/api/system-performance'
    data = {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_percent,
        'disk_percent': disk_percent
    }
    response = requests.post(url, json=data)

    if response.status_code != 200:
        logging.error(f'Failed to send system performance data to monitoring service: {response.text}')

if __name__ == '__main__':
    while True:
        monitor_transactions()
        monitor_system_performance()
        time.sleep(60)
