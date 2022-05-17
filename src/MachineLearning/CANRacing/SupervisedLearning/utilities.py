# some decorator magic
import time

def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def wait_forever(msg: str = "Waiting forever."):
    print(msg, end="", flush=True)
    while True:
        time.sleep(1)
        print(".", end="", flush=True)