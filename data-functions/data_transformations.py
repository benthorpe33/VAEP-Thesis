import pandas as pd

from kloppy.wyscout import load
from socceraction.spadl.kloppy import *
import json
import requests
from socceraction.vaep import VAEP

username = "2wsgzl0-vwx2hipkj-yxme0ub-wgjenczp83"
password = "nQv7vWe@FtES5_Z%4@F^GbtZrA*&Rh"


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


if __name__ == '__main__':
    match_id = 5272925
    file_path_out = f'C:/Users/thorp/Downloads/{match_id}_events.json'

    match = requests.get(f"https://apirest.wyscout.com/v3/matches/{match_id}?useSides=1", auth=(username, password))
    match_json = match.json()

    events = requests.get(f'https://apirest.wyscout.com/v3/matches/{match_id}/events', auth=(username, password))
    events_json = events.json()
    events_json['teams'] = {61493: "Duke Blue Devils", 61485: "Wake Forest Demon Deacon"}
    players_dict = {}
    for wyId, teamName in events_json['teams'].items():
        players_req = requests.get('https://apirest.wyscout.com/v3/teams/%s/squad' % wyId, auth=(username, password))
        players = players_req.json()['squad']
        players_dict[wyId] = dict(
            [
                (wyId, {player['wyId']: player for player in players})
            ]
        )
    events_json['players'] = players_dict
    save_events(events_json, file_path_out)
    kloppy_data = wyscout_to_kloppy(file_path_out)
    spadl_data = kloppy_to_spadl(kloppy_data)

    df_games = pd.Series()
    df_games['game_id'] = match_json['wyId']
    df_games['competition_id'] = match_json['competitionId']
    df_games['season_id'] = match_json['seasonId']
    df_games['game_date'] = match_json['date']
    df_games['game_day'] = match_json['gameweek']
    df_games['home_team_id'] = match_json['teamsData']['home']
    df_games['away_team_id'] = match_json['teamsData']['away']
    # df_games = df_games.set_index('game_id')

    VAEP_model = VAEP(nb_prev_actions=5)
    features = VAEP_model.compute_features(df_games, spadl_data)
    labels = VAEP_model.compute_labels(df_games, spadl_data)
    VAEP_model.fit(features, labels)

    hi = 1
