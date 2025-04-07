from .config import Config


def t(key, value):
    return {"key": key, "value": value}

def detection_to_tags(detection, config: Config):
    """Change the detections to Panoramax semantic tags"""
    res = []
    for info in detection["info"]:
        sem = []
        if info["class"] == "sign":
            sem.append(t("osm:traffic_sign", "yes"))
            sem.append(t("osm:traffic_sign:model", config.model_name))
            sem.append(t("osm:traffic_sign:confidence", info["confidence"]))
            if "model_version" in detection:
                sem.append(t("osm:traffic_sign:model:version", detection["model_version"]))
        else:
            continue

        res.append({
            "shape": info["xywh"],
            "semantics": sem
        })
    return res