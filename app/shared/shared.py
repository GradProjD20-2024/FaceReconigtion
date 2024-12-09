def exclude_empty(data: dict):
    return {k: v for k, v in data.items() if v is not None}


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance