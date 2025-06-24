import time
import random
from prometheus_client import start_http_server, Summary, Counter, REGISTRY, ProcessCollector, PlatformCollector


REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNTER = Counter('model_requests_total', 'Total number of prediction requests')

REGISTRY.register(ProcessCollector())
try:
    REGISTRY.register(PlatformCollector())
except ValueError:
    pass


@REQUEST_TIME.time()
def process_request():
    REQUEST_COUNTER.inc()
    time.sleep(random.random())  

if __name__ == '__main__':
    start_http_server(8001)  
    while True:
        process_request()
        time.sleep(2)