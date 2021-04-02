import psutil


def get_vitals():
    return {"memory": get_memory_vitals(), "cpu": get_cpu_vitals()}


def get_memory_vitals():
    virtual_memory = psutil.virtual_memory()
    return {
        item: getattr(virtual_memory, item, None)
        for item in [
            "total",
            "available",
            "percent",
            "used",
            "free",
            "active",
            "inactive",
            "wired",
        ]
    }


def get_cpu_vitals():
    return {"percent": psutil.cpu_percent()}
