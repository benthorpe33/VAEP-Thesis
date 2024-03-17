import pandas as pd
import requests
from data_transformations import wyscout_to_kloppy, kloppy_to_spadl, save_events
from sklearn.linear_model import LinearRegression, LogisticRegression
from socceraction.spadl.wyscout import *
from socceraction.vaep import VAEP
import glob
from tqdm import tqdm
import json
from kloppy.wyscout import load, load_open_data

seasonId = '190238'
matchId = '5272925'
dukeId = 61485

username = "2wsgzl0-vwx2hipkj-yxme0ub-wgjenczp83"
password = "nQv7vWe@FtES5_Z%4@F^GbtZrA*&Rh"

folder_path = 'C:/Users/thorp/Documents/ACC_spadl/*.csv'
csv_files = glob.glob(folder_path)
file_path_mapping = {file_path.split('\\')[-1].replace('.csv', ''): file_path for file_path in csv_files}


def make_acc_teams_dict():
    acc_teams = requests.get("https://apirest.wyscout.com/v3/seasons/190238/teams", auth=(username, password))
    acc_dict = {}
    for team in acc_teams.json()["teams"]:
        acc_dict[team["wyId"]] = team["name"]
    save_events(acc_dict, f'C:/Users/thorp/Documents/ACC_teams.json')


def make_acc_matches():
    matches = requests.get('https://apirest.wyscout.com/v3/competitions/43231/matches',
                           auth=(username, password))
    df_acc = pd.DataFrame(matches.json()['matches'])

    match_list = df_acc['matchId'].tolist()
    home = []
    away = []

    for match in tqdm(match_list):
        match_call = requests.get('https://apirest.wyscout.com/v3/matches/%s?useSides=1' % match,
                                  auth=(username, password))
        df_match = pd.DataFrame(match_call.json()['teamsData'])
        home_team = df_match['home'].tolist()[0]
        away_team = df_match['away'].tolist()[0]

        home.append(home_team)
        away.append(away_team)

    df_games = pd.DataFrame()

    df_games['game_id'] = df_acc['matchId']
    df_games['competition_id'] = df_acc['competitionId']
    df_games['season_id'] = df_acc['seasonId']
    df_games['game_date'] = df_acc['date']
    df_games['game_day'] = df_acc['gameweek']
    df_games['home_team_id'] = home
    df_games['away_team_id'] = away
    df_games = df_games.set_index('game_id')

    df_games.to_csv(r'C:/Users/thorp/Documents/ACC_games.csv')


def acc_v3_to_spadl():
    filepath = "C:/Users/thorp/Documents/ACC_teams.json"
    with open(filepath, 'r') as json_file:
        acc_dict = json.load(json_file)

    df_games = pd.read_csv(r'C:/Users/thorp/Documents/ACC_games.csv', index_col=0)

    for game_id, game in tqdm(list(df_games.iterrows())):
        file_path_out = f'C:/Users/thorp/Documents/ACC_games/{game_id}_events.json'
        events = requests.get('https://apirest.wyscout.com/v3/matches/%s/events' % game_id, auth=(username, password))
        if events.status_code == 400:
            print('Match {} has no data'.format(game_id))
            continue
        events_json = events.json()
        events_json['teams'] = {game['home_team_id']: acc_dict[f"{game['home_team_id']}"],
                                game['away_team_id']: acc_dict[f"{game['away_team_id']}"]}
        players_dict = {}
        for wyId, teamName in events_json['teams'].items():
            players_req = requests.get('https://apirest.wyscout.com/v3/teams/%s/squad' % wyId,
                                       auth=(username, password))
            players = players_req.json()['squad']
            players_dict[wyId] = dict(
                [
                    (wyId, {player['wyId']: player for player in players})
                ]
            )
        events_json['players'] = players_dict
        save_events(events_json, file_path_out)
        try:
            kloppy_data = wyscout_to_kloppy(file_path_out)
        except KeyError as e:
            print(e)
            continue
        spadl_data = kloppy_to_spadl(kloppy_data, game_id=game_id)
        spadl_data.to_csv(fr'C:/Users/thorp/Documents/ACC_spadl/{game_id}.csv', index=False)


def train_test_data():
    df_games = pd.read_csv(r'C:/Users/thorp/Documents/ACC_games.csv', index_col=0)
    df_games["game_date2"] = pd.to_datetime(df_games["game_date"])
    df_games_sorted = df_games.sort_values(by="game_date2")
    df_games_sorted.drop("game_date2", axis=1, inplace=True)
    cutoff_index = int(len(df_games_sorted) * 0.8)
    earliest_80_percent_df = df_games_sorted.iloc[:cutoff_index]
    most_recent_20_percent_df = df_games_sorted.iloc[cutoff_index:]
    return earliest_80_percent_df, most_recent_20_percent_df


def v2_acc_VAEP_features_and_labels(data, k, test="train"):
    train_data = data
    VAEP_model = VAEP(nb_prev_actions=k)

    # compute features and labels for each game
    all_features, all_labels = [], []
    for game_id, game in tqdm(list(train_data.iterrows())):
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
        try:
            events_df['period_id'] = events_df['period_id'].astype('int')
        except ValueError:
            events_df = events_df[events_df['period_id'] != "P"]

        game_events = events_df

        # convert the events to actions
        game_home_team_id = train_data.at[game_id, "home_team_id"]
        game_actions = convert_to_actions(game_events, game_home_team_id)
        # compute features and labels
        all_features.append(VAEP_model.compute_features(game, game_actions))
        all_labels.append(VAEP_model.compute_labels(game, game_actions))
    # combine all features and labels in a single dataframe
    all_features = pd.concat(all_features)
    all_labels = pd.concat(all_labels)

    all_features.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V2/{test}/{k}.csv")
    all_labels.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V2/{test}/{k}.csv")

    return all_features, all_labels


def v3_acc_VAEP_features_and_labels(data, k, test="train"):
    train_data = data
    VAEP_model = VAEP(nb_prev_actions=k)
    train_features, train_labels = [], []
    for game_id, game in tqdm(train_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        train_features.append(VAEP_model.compute_features(game, spadl_data))
        train_labels.append(VAEP_model.compute_labels(game, spadl_data))

    # combine all features and labels in a single dataframe
    train_features = pd.concat(train_features)
    train_labels = pd.concat(train_labels)

    train_features.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V3/{test}/{k}.csv")
    train_labels.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V3/{test}/{k}.csv")

    return train_features, train_labels


def get_test_score(test_data, VAEP_model):
    scores = []
    test_features, test_labels = [], []
    for game_id, game in tqdm(test_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        test_features.append(VAEP_model.compute_features(game, spadl_data))
        test_labels.append(VAEP_model.compute_labels(game, spadl_data))

    test_features = pd.concat(test_features)
    test_labels = pd.concat(test_labels)

    # For testing:
    # test_features.drop("actiontype_shot_result_success_a0", axis=1, inplace=True)

    scores.append(VAEP_model.score(test_features, test_labels))
    return scores


def v2_model_train(k):
    try:
        all_features = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V2/train/{k}.csv")
        all_labels = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V2/train/{k}.csv")
    except FileNotFoundError:
        all_features, all_labels = v2_acc_VAEP_features_and_labels(earliest_80_percent_df, k)

    VAEP_model = VAEP(nb_prev_actions=k)

    # fit the model
    all_features = all_features.loc[:, ~all_features.columns.str.contains('^Unnamed')]
    all_labels = all_labels.loc[:, ~all_labels.columns.str.contains('^Unnamed')]
    VAEP_model.fit(all_features, all_labels)
    return VAEP_model


def v3_model_train(k):
    try:
        train_features = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V3/train/{k}.csv")
        train_labels = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V3/train/{k}.csv")
    except FileNotFoundError:
        train_features, train_labels = v3_acc_VAEP_features_and_labels(k)
    train_features = train_features.loc[:, ~train_features.columns.str.contains('^Unnamed')]
    train_labels = train_labels.loc[:, ~train_labels.columns.str.contains('^Unnamed')]

    VAEP_model = VAEP(nb_prev_actions=k)

    # Testing removal of features
    # train_features = train_features.drop("actiontype_shot_result_success_a0", axis=1)

    VAEP_model.fit(train_features, train_labels)

    return VAEP_model


def logistic_goal_model(k):
    VAEP_model = VAEP(nb_prev_actions=k)
    train_data, test_data = train_test_data()
    train_features, train_labels = [], []
    for game_id, game in tqdm(train_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        train_features.append(VAEP_model.compute_features(game, spadl_data))
        train_labels.append(VAEP_model.compute_labels(game, spadl_data))
    train_features = pd.concat(train_features)
    train_labels = pd.concat(train_labels)
    VAEP_model.fit(train_features, train_labels)

    test_features, test_labels = [], []
    for game_id, game in tqdm(test_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        test_features.append(VAEP_model.compute_features(game, spadl_data))
        test_labels.append(VAEP_model.compute_labels(game, spadl_data))

    test_features = pd.concat(test_features)
    test_labels = pd.concat(test_labels)

    # Gets AUROC and Brier score for scoring and conceding
    # scores = VAEP_model.score(test_features, test_labels)

    model = LogisticRegression()
    feature_indices = [154, 160, 130, 94, 11, 70, 406, 82, 528, 517, 565, 540, 556, 520, 279, 548, 522, 518, 537, 417]
    feature_indices = [154, 160, 130, 94, 11, 70, 406, 82]
    feature_set = train_features.iloc[:, feature_indices]
    X = feature_set
    y = train_labels['scores']
    model.fit(X, y)
    X_test = test_features.iloc[:, feature_indices]
    preds = model.predict(X_test)
    hi = 1


###############################################

# match_df = pd.DataFrame(match_data.json()['teamsData'])
# events = requests.get(f'https://apirest.wyscout.com/v2/matches/{match_id}/events', auth=(username, password))
# events_df_v2 = pd.DataFrame(events.json()['events'])
#
# events = requests.get(f'https://apirest.wyscout.com/v3/matches/{match_id}/events', auth=(username, password))
# events_df_v3 = pd.DataFrame(events.json()['events'])
#
# events_df_v2.columns = ['event_id', 'player_id', 'team_id', 'game_id', 'period_id', 'milliseconds',
#                      'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']
#
# cols = ['event_id', 'game_id', 'period_id', 'milliseconds', 'team_id', 'player_id',
#         'type_id', 'type_name', 'subtype_id', 'subtype_name', 'positions', 'tags']
# events_df = events_df_v2[cols]
#
# events_df['period_id'] = events_df['period_id'].str.replace("H", "")
# events_df['period_id'] = events_df['period_id'].str.replace("E", "")
# events_df['period_id'] = events_df['period_id'].astype('int')
#
# game_events = events_df
#
# game_actions = convert_to_actions(game_events, match_df['home']['teamId'])
#
# df_games = pd.DataFrame()
# df_games['game_id'] = df_duke['matchId']
# df_games['competition_id'] = df_acc['competitionId']
# df_games['season_id'] = df_acc['seasonId']
# df_games['game_date'] = df_acc['date']
# df_games['game_day'] = df_acc['gameweek']
# df_games['home_team_id'] = home
# df_games['away_team_id'] = away
# df_games = df_games.set_index('game_id')
#
# features = VAEP_model.compute_features(game, game_actions)
# labels = VAEP_model.compute_labels(game, game_actions)
# VAEP_model.fit(features, labels)
#
# events_df[['id', 'name', 'position']] = events_df['player'].apply(lambda x: pd.Series(x))
# events_df[['primary', 'secondary']] = events_df['type'].apply(lambda x: pd.Series(x))
# events_df[['teamId', 'teamName', 'teamFormation']] = events_df['team'].apply(lambda x: pd.Series(x))
# events_df[['opponentId', 'opponentName', 'opponentFormation']] = events_df['opponentTeam'].apply(
#     lambda x: pd.Series(x))
# events_df = events_df[events_df['primary'] == 'pass']
# events_df = events_df[events_df['secondary'].apply(lambda x: 'progressive_pass' in x)]
# events_df[['accurate', 'angle', 'height', 'length', 'recipient', 'endLocation']] \
#     = events_df['pass'].apply(lambda x: pd.Series(x))
# events_df[['id', 'recipientName', 'position']] \
#     = events_df['recipient'].apply(lambda x: pd.Series(x))
# events_df['text'] = events_df['name'] + ' to ' + events_df['recipientName'] + " at " + \
#                     events_df['minute'].astype(str) + ':' + events_df['second'].astype(str)
# events_df[['startLocation_x', 'startLocation_y']] \
#     = events_df['location'].apply(lambda x: pd.Series(x))
# events_df[['endLocation_x', 'endLocation_y']] \
#     = events_df['endLocation'].apply(lambda x: pd.Series(x))
# events_df = events_df[events_df['accurate']]
# events_df.dropna(subset=['recipientName'], inplace=True)
# events_df = events_df[events_df['secondary'].apply(lambda x: 'cross' in x)]


if __name__ == "__main__":
    # make_acc_matches()
    # acc_v3_to_spadl()
    # duke_matches = requests.get('https://apirest.wyscout.com/v3/teams/61485/matches?seasonId=190238',
    #                             auth=(username, password))
    earliest_80_percent_df, most_recent_20_percent_df = train_test_data()
    # v2_model = v2_model_train(3)
    # v2_scores = get_test_score(most_recent_20_percent_df, v2_model)
    # acc_matches = requests.get('https://apirest.wyscout.com/v3/seasons/190238/matches',
    #                            auth=(username, password))
    v3_model = v3_model_train(3)
    v3_scores = get_test_score(most_recent_20_percent_df, v3_model)
    logistic_goal_model(3)
