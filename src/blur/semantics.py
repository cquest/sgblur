from .config import Config


def t(key, value):
    return {"key": key, "value": value}

def detection_to_tags(detection, config: Config):
    """Change the detections to Panoramax semantic tags"""
    res = {'annotations': []}
    salt = detection.get('salt')
    if salt:
        res['blurring_id'] = salt
    model_full_name = f"{config.api_name}-{detection['model']['name']}/{detection['model']['version']}"
    for info in detection["info"]:
        sem = []
        if info["class"] == "sign":
            sem.append(t("osm|traffic_sign", "yes"))
            sem.append(t("detection_model[osm|traffic_sign=yes]", model_full_name))
            sem.append(t("detection_confidence[osm|traffic_sign=yes]", str(info["confidence"])))
        else:
            continue

        res['annotations'].append({
            "shape": info["xywh"],
            "semantics": sem
        })

    return res
