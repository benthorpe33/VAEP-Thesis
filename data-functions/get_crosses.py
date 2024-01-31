import requests
import pandas as pd
from vaep_spadl.wyscout import *
from vaep_model import VAEP

VAEP_model = VAEP(nb_prev_actions=3)

username = "2wsgzl0-vwx2hipkj-yxme0ub-wgjenczp83"
password = "nQv7vWe@FtES5_Z%4@F^GbtZrA*&Rh"
match_id = 5517622

match_data = requests.get(f"https://apirest.wyscout.com/v3/matches/{match_id}?useSides=1",
                          auth=(username, password))

match_df = pd.DataFrame(match_data.json()['teamsData'])
events = requests.get(f'https://apirest.wyscout.com/v2/matches/{match_id}/events', auth=(username, password))
events_df_v2 = pd.DataFrame(events.json()['events'])

events = requests.get(f'https://apirest.wyscout.com/v3/matches/{match_id}/events', auth=(username, password))
events_df_v3 = pd.DataFrame(events.json()['events'])

events_df_v2.columns = ['event_id', 'player_id', 'team_id', 'game_id', 'period_id', 'milliseconds',
                     'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']

cols = ['event_id', 'game_id', 'period_id', 'milliseconds', 'team_id', 'player_id',
        'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']
events_df = events_df_v2[cols]

events_df['period_id'] = events_df['period_id'].str.replace("H", "")
events_df['period_id'] = events_df['period_id'].str.replace("E", "")
events_df['period_id'] = events_df['period_id'].astype('int')

game_events = events_df

game_actions = convert_to_actions(game_events, match_df['home']['teamId'])


events_df[['id', 'name', 'position']] = events_df['player'].apply(lambda x: pd.Series(x))
events_df[['primary', 'secondary']] = events_df['type'].apply(lambda x: pd.Series(x))
events_df[['teamId', 'teamName', 'teamFormation']] = events_df['team'].apply(lambda x: pd.Series(x))
events_df[['opponentId', 'opponentName', 'opponentFormation']] = events_df['opponentTeam'].apply(
    lambda x: pd.Series(x))
events_df = events_df[events_df['primary'] == 'pass']
events_df = events_df[events_df['secondary'].apply(lambda x: 'progressive_pass' in x)]
events_df[['accurate', 'angle', 'height', 'length', 'recipient', 'endLocation']] \
    = events_df['pass'].apply(lambda x: pd.Series(x))
events_df[['id', 'recipientName', 'position']] \
    = events_df['recipient'].apply(lambda x: pd.Series(x))
events_df['text'] = events_df['name'] + ' to ' + events_df['recipientName'] + " at " + \
                    events_df['minute'].astype(str) + ':' + events_df['second'].astype(str)
events_df[['startLocation_x', 'startLocation_y']] \
    = events_df['location'].apply(lambda x: pd.Series(x))
events_df[['endLocation_x', 'endLocation_y']] \
    = events_df['endLocation'].apply(lambda x: pd.Series(x))
events_df = events_df[events_df['accurate']]
events_df.dropna(subset=['recipientName'], inplace=True)
events_df = events_df[events_df['secondary'].apply(lambda x: 'cross' in x)]
hi=1
