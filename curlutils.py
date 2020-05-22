from io import BytesIO
from urllib.parse import urlencode
import pycurl


def curl(url,
         post_data=None,
         user_pwd=None,
         headers=None,
         custom_request=None,
         cookies=None):
    b = BytesIO()
    c = pycurl.Curl()
    c.setopt(c.WRITEDATA, b)
    c.setopt(c.URL, url)
    c.setopt(pycurl.TIMEOUT, 6000)
    if user_pwd:
        c.setopt(c.USERPWD, user_pwd)
    if post_data:
        postfields = urlencode(post_data)
        c.setopt(c.POSTFIELDS, postfields)
    if headers:
        c.setopt(c.HTTPHEADER, headers)
    if custom_request:
        c.setopt(pycurl.CUSTOMREQUEST, custom_request)
    if cookies:
        c.setopt(pycurl.COOKIE, cookies)
    c.setopt(c.VERBOSE, 0)
    c.perform()
    c.close()
    body = b.getvalue()
    return body.decode('iso-8859-1')


def get_run_args(url, headers=None, max_time=5):
    headers = headers or {}
    run_args = ["curl"]
    for k, v in headers.items():
        run_args.append('-H')
        run_args.append(k + ": " + v)
    run_args.append("-s")
    run_args.append(url)
    run_args.append("--max-time")
    run_args.append("%s" % max_time)
    return run_args
