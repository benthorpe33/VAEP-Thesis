from kloppy.wyscout import load
from socceraction.spadl.kloppy import *
import json


def save_events(data, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file)


def get_events(filepath):
    with open(filepath, 'r') as json_file:
        return json.load(json_file)


def wyscout_to_kloppy(file_path):
    kloppy_event_data = load(
        event_data=file_path,
        # Optional arguments
        coordinates="wyscout",
        data_version="V3"
    )
    return kloppy_event_data


def kloppy_to_spadl(kloppy_event_data, game_id=None):
    spadl_event_data = convert_to_actions(kloppy_event_data, game_id=game_id)
    return spadl_event_data
