from .config import Config


def t(key, value):
    return {"key": key, "value": value}

def detection_to_tags(detection, config: Config):
    """Change the detections to Panoramax semantic tags"""
    res = []
    model_full_name = f"{config.api_name}-{detection['model']['name']}/{detection['model']['version']}"
    for info in detection["info"]:
        sem = []
        if info["class"] == "sign":
            sem.append(t("osm|traffic_sign", "yes"))
            sem.append(t("detection_model[osm|traffic_sign=yes]", model_full_name))
            sem.append(t("detection_confidence[osm|traffic_sign=yes]", info["confidence"]))
        else:
            continue

        res.append({
            "shape": info["xywh"],
            "semantics": sem
        })
    return res
