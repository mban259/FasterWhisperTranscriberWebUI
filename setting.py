import json
import os

if os.path.isfile("setting.json"):
    with open("setting.json") as f:
        js = json.load(f)
        val_threshold = js["threshold"]
        val_volume = js["volume"]
        val_language = js["language"] 
else:
    val_threshold = 0.1
    val_volume = 0.8
    val_language = "None"


def save(threshold: float, volume: float, language: str):
    js = {}
    js["threshold"] = threshold
    js["volume"] = volume
    js["language"] = language

    with open("setting.json", mode="wt") as f:
        json.dump(js, f)