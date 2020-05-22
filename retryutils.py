import asyncio


def retry(n, f, *args, **kwargs):
    error = None
    for _ in range(n):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error = e
    msg = "Could not get (%s, %s, %s) after %s tries with error %s"
    raise Exception(msg % (f,
                           args,
                           kwargs,
                           n,
                           error))

async def async_retry(n, f, *args, **kwargs):
    error = None
    for _ in range(n):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            error = e
    msg = "Could not get (%s, %s, %s) after %s tries with error %s"
    raise Exception(msg % (f,
                           args,
                           kwargs,
                           n,
                           error))

async def async_retry_sleep(n, f, sleep, *args, **kwargs):
    error = None
    for _ in range(n):
        try:
            return await f(*args, **kwargs)
        except Exception as e:
            error = e
        if sleep:
            await asyncio.sleep(sleep)
    msg = "Could not get (%s, %s, %s) after %s tries with error %s"
    raise Exception(msg % (f,
                           args,
                           kwargs,
                           n,
                           error))
