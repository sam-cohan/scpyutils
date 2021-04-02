import logging
import os
import pathlib
import random
import re

import numpy as np
import pandas as pd
import requests

import geoip2.database
import miniupnpc

from . import cacheutils as chu
from .logutils import setup_logger

HOME_DIR = pathlib.Path.home()
MAXMIND_DBS_ROOT_DIR = os.path.join(HOME_DIR, "data/maxmind_dbs")

MAXMIND_DB_PATHS = {
    "country": {
        "local_path": os.path.join(MAXMIND_DBS_ROOT_DIR, "GeoLite2-Country.mmdb"),
    },
    "city": {
        "local_path": os.path.join(MAXMIND_DBS_ROOT_DIR, "GeoLite2-City.mmdb"),
    },
}


LOGGER = setup_logger(__name__, log_level=logging.INFO)


def load_mmdb(db_name, force_refresh_db=False):
    db_paths = MAXMIND_DB_PATHS[db_name]
    local_path = db_paths["local_path"]
    if local_path:
        return geoip2.database.Reader(local_path)


MM_COUNTRY_DB = load_mmdb("country")
MM_CITY_DB = load_mmdb("city")

IS_IP_COMPILED_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
EXTRACT_IP_COMPILED_RE = re.compile(r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}")


def get_external_ip_without_external_service():
    """
    Note: This has the advantage that it does not rely on external service,
    but is not compatible with VPNs.
    """
    u = miniupnpc.UPnP()
    u.discoverdelay = 100
    u.discover()
    u.selectigd()
    return u.externalipaddress()


def get_external_ip_from_aws():
    return requests.get("https://checkip.amazonaws.com").text.strip()


def get_external_ip_from_checkmyip():
    txt = requests.get("http://checkip.dyndns.org").text
    return EXTRACT_IP_COMPILED_RE.findall(txt)[0]


def get_external_ip_from_hostip():
    """Note that this service does not give same result as other services."""
    return requests.get("http://api.hostip.info/get_json.php").json()["ip"]


def get_external_ip_from_icanhazip():
    return requests.get("http://icanhazip.com/").text.strip("\n")


def get_external_ip_from_ipify():
    return requests.get("https://api.ipify.org").text.strip()


def get_external_ip_from_jsonip():
    return requests.get("http://jsonip.com").json()["ip"]


def get_external_ip_from_networksecuritytoolkit():
    return requests.get(
        "http://www.networksecuritytoolkit.org/nst/tools/ip.shtml"
    ).text.strip("\n")


def get_external_ip_from_whatsmyip():
    txt = requests.get("http://whatismyip.org").text
    return EXTRACT_IP_COMPILED_RE.findall(txt)[0]


def get_external_ip_from_wtfismyip():
    return requests.get("http://wtfismyip.com/json").json()["YourFuckingIPAddress"]


EXTERNAL_IP_SERVICES = [
    get_external_ip_from_aws,
    get_external_ip_from_checkmyip,
    # get_external_ip_from_hostip,  # This service does not provide same result as others.
    get_external_ip_from_icanhazip,
    get_external_ip_from_ipify,
    get_external_ip_from_jsonip,
    get_external_ip_from_networksecuritytoolkit,
    get_external_ip_from_whatsmyip,
    get_external_ip_from_wtfismyip,
]


@chu.memoize_with_hashable_args
def is_valid_ip(ip):
    return not pd.isnull(ip) and ip != "" and bool(IS_IP_COMPILED_RE.match(ip))


def get_external_ip(max_attempts=10):
    attempt = 0
    tried_idxs = set()
    while attempt < max_attempts:
        idx = int(random.uniform(0, len(EXTERNAL_IP_SERVICES)))
        if idx in tried_idxs:
            if len(tried_idxs) == len(EXTERNAL_IP_SERVICES):
                break
            continue
        tried_idxs.add(idx)
        ip_getter = EXTERNAL_IP_SERVICES[idx]
        try:
            ip = ip_getter()
            if not is_valid_ip(ip):
                raise Exception(f"{ip_getter} gave invalid ip: {ip}")
            return ip
        except Exception as e:
            LOGGER.exception(
                "Failed to get external IP from external service=%s %s", ip_getter, e
            )
            attempt += 1
    raise Exception("Unable to get external IP")


@chu.memoize_with_hashable_args
def get_ip_country(ip):
    try:
        return MM_COUNTRY_DB.country(ip).country.iso_code if is_valid_ip(ip) else np.NaN
    except geoip2.errors.AddressNotFoundError:
        return np.NaN
    except ValueError as e:
        LOGGER.warning(e)
        return np.NaN


@chu.memoize_with_hashable_args
def get_ip_city(ip):
    try:
        return MM_CITY_DB.city(ip).city.name if is_valid_ip(ip) else np.NaN
    except geoip2.errors.AddressNotFoundError:
        return np.NaN
    except ValueError as e:
        LOGGER.warning(e)
        return np.NaN


@chu.memoize_with_hashable_args
def get_ip_city_country(ip):
    try:
        if not is_valid_ip(ip):
            return np.NaN
        city = MM_CITY_DB.city(ip)
        return (city.city.name, city.country.iso_code)
    except geoip2.errors.AddressNotFoundError:
        return np.NaN
    except ValueError as e:
        LOGGER.warning(e)
        return np.NaN


def clear_iputils_caches():
    for func in [is_valid_ip, get_ip_country, get_ip_city, get_ip_city_country]:
        print(
            "{} cache_size is {:,.0f}... clearing it".format(
                func.__name__, len(func._cached_results_)
            )
        )
        func._cached_results_.clear()
