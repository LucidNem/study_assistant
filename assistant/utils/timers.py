import time
import functools

def timed_function(func):
 
#  Decorator that measures and prints the execution time of any function it wraps.
    
# Usage:
# @timed_function
# def your_function(...):
#     ...
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"[TIMER] Function `{func.__name__}` took {duration:.4f} seconds.")
        return result
    return wrapper
