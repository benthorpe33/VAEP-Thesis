import requests
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from socceraction.spadl.wyscout import *
from socceraction.vaep import VAEP
from xgboost import plot_importance
import json
from tqdm import tqdm
from kloppy.wyscout import load, load_open_data

seasonId = '190238'
matchId = '5272925'
dukeId = 61485

username = "2wsgzl0-vwx2hipkj-yxme0ub-wgjenczp83"
password = "nQv7vWe@FtES5_Z%4@F^GbtZrA*&Rh"

duke_matches = requests.get('https://apirest.wyscout.com/v3/teams/61485/matches?seasonId=190238', auth=(username, password))
df_duke = pd.DataFrame(duke_matches.json()['matches'])

match_list = df_duke['matchId'].tolist()
home = []
away = []

for match in tqdm(match_list):

    match_call = requests.get('https://apirest.wyscout.com/v3/matches/%s?useSides=1' % match, auth=(username, password))
    df_match = pd.DataFrame(match_call.json()['teamsData'])
    home_team = df_match['home'].tolist()[0]
    away_team = df_match['away'].tolist()[0]

    home.append(home_team)
    away.append(away_team)

df_games = pd.DataFrame()

df_games['game_id'] = df_duke['matchId']
df_games['competition_id'] = df_duke['competitionId']
df_games['season_id'] = df_duke['seasonId']
df_games['game_date'] = df_duke['date']
df_games['game_day'] = df_duke['gameweek']
df_games['home_team_id'] = home
df_games['away_team_id'] = away
df_games = df_games.set_index('game_id')

VAEP_model = VAEP(nb_prev_actions=5)

# compute features and labels for each game
all_features, all_labels = [], []
for game_id, game in tqdm(list(df_games.iterrows())):
    # load the game's events
    # game_events = loaded.events(game_id)
    events = requests.get('https://apirest.wyscout.com/v2/matches/%s/events' % game_id, auth=(username, password))
    if events.status_code == 400:
        print('Match {} has no data'.format(game_id))
        continue

    events_df = pd.DataFrame(events.json()['events'])
    events_df.columns = ['event_id', 'player_id', 'team_id', 'game_id', 'period_id', 'milliseconds',
                         'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']

    cols = ['event_id', 'game_id', 'period_id', 'milliseconds', 'team_id', 'player_id',
            'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']
    events_df = events_df[cols]

    events_df['period_id'] = events_df['period_id'].str.replace("H", "")
    events_df['period_id'] = events_df['period_id'].str.replace("E", "")
    events_df['period_id'] = events_df['period_id'].astype('int')

    game_events = events_df

    # convert the events to actions
    game_home_team_id = df_games.at[game_id, "home_team_id"]
    game_actions = convert_to_actions(game_events, game_home_team_id)
    # compute features and labels
    all_features.append(VAEP_model.compute_features(game, game_actions))
    all_labels.append(VAEP_model.compute_labels(game, game_actions))
# combine all features and labels in a single dataframe
all_features = pd.concat(all_features)
all_labels = pd.concat(all_labels)

# fit the model
VAEP_model.fit(all_features, all_labels)

# Gets VAEP values for each action in a game
ratings = VAEP_model.rate(df_games.loc[5517560], game_actions)
# Gets AUROC and Brier score for scoring and conceding
scores = VAEP_model.score(all_features, all_labels)



game_actions['vaep_value'] = ratings['vaep_value'].tolist()
game_actions['bodypart_id'] = game_actions['bodypart_id'].astype('category')
game_actions['type_id'] = game_actions['type_id'].astype('category')
game_actions['result_id'] = game_actions['result_id'].astype('category')
game_actions['action_id'] = game_actions['action_id'].astype('category')

X = all_features
y = game_actions['vaep_value']

model = LinearRegression()
model.fit(X, y)

model2 = LogisticRegression()
y = all_labels['scores']
model2.fit(X, y)




###############################################

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

df_games = pd.DataFrame()
df_games['game_id'] = df_duke['matchId']
df_games['competition_id'] = df_acc['competitionId']
df_games['season_id'] = df_acc['seasonId']
df_games['game_date'] = df_acc['date']
df_games['game_day'] = df_acc['gameweek']
df_games['home_team_id'] = home
df_games['away_team_id'] = away
df_games = df_games.set_index('game_id')

features = VAEP_model.compute_features(game, game_actions)
labels = VAEP_model.compute_labels(game, game_actions)
VAEP_model.fit(features, labels)

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
hi = 1
