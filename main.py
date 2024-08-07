import requests
from data_transformations import wyscout_to_kloppy, kloppy_to_spadl, save_events
from socceraction.spadl.wyscout import *
from socceraction.vaep import VAEP
import glob
from tqdm import tqdm
import json

from sensitive_vars import username, password, seasonId, accId

csv_files = glob.glob('C:/Users/thorp/Documents/ACC_spadl/*.csv')
file_path_mapping = {file_path.split('\\')[-1].replace('.csv', ''): file_path for file_path in csv_files}


def make_acc_teams_dict():
    acc_teams = requests.get(f"https://apirest.wyscout.com/v3/seasons/{seasonId}/teams", auth=(username, password))
    acc_dict = {}
    for team in acc_teams.json()["teams"]:
        acc_dict[team["wyId"]] = team["name"]
    save_events(acc_dict, f'C:/Users/thorp/Documents/ACC_teams.json')


def make_acc_matches():
    matches = requests.get(f"https://apirest.wyscout.com/v3/competitions/{accId}/matches",
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


def v2_acc_VAEP_features(data, j, test="train"):
    train_data = data
    VAEP_model = VAEP(nb_prev_actions=j)

    # compute features and labels for each game
    all_features = []
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
    # combine all features and labels in a single dataframe
    all_features = pd.concat(all_features)

    all_features.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V2/{test}/{j}.csv")

    return all_features


def v2_acc_VAEP_labels(data, k, test="train"):
    train_data = data
    VAEP_model = VAEP()

    # compute features and labels for each game
    all_labels = []
    for game_id, game in tqdm(list(train_data.iterrows())):
        # load the game's events
        # game_events = loaded.events(game_id)
        events = requests.get('https://apirest.wyscout.com/v2/matches/%s/events' % game_id, auth=(username, password))
        if events.status_code != 200:
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
        all_labels.append(VAEP_model.compute_labels(game, game_actions, k))
    # combine all labels in a single dataframe
    all_labels = pd.concat(all_labels)

    all_labels.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V2/{test}/{k}.csv")

    return all_labels


def v3_acc_VAEP_features(data, j, test="train"):
    train_data = data
    VAEP_model = VAEP(nb_prev_actions=j)
    train_features = []
    for game_id, game in tqdm(train_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        train_features.append(VAEP_model.compute_features(game, spadl_data))

    # combine all features and labels in a single dataframe
    train_features = pd.concat(train_features)

    train_features.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V3/{test}/{j}.csv")

    return train_features


def v3_acc_VAEP_labels(data, k, test="train"):
    train_data = data
    VAEP_model = VAEP()
    train_labels = []
    for game_id, game in tqdm(train_data.iterrows()):
        file_path = file_path_mapping[f"{game_id}"]
        spadl_data = pd.read_csv(file_path)
        train_labels.append(VAEP_model.compute_labels(game, spadl_data, k))

    # combine all features and labels in a single dataframe
    train_labels = pd.concat(train_labels)

    train_labels.to_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V3/{test}/{k}.csv")

    return train_labels


def get_test_score(VAEP_model, version, j, k):
    try:
        test_features = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/{version}/test/{j}.csv")
    except FileNotFoundError:
        return f"{version} {j} features not found"
    try:
        test_labels = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/{version}/test/{k}.csv")
    except FileNotFoundError:
        return f"{version} {k} labels not found"

    # For testing:
    # test_features.drop("actiontype_shot_result_success_a0", axis=1, inplace=True)

    return VAEP_model.score(test_features, test_labels)


def v2_model_train(j, k, earliest_80_percent_df):
    try:
        all_features = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V2/train/{j}.csv")
        all_labels = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V2/train/{k}.csv")
    except FileNotFoundError:
        all_features = v2_acc_VAEP_features(earliest_80_percent_df, j=j)
        all_labels = v2_acc_VAEP_labels(earliest_80_percent_df, k=k)

    VAEP_model = VAEP(nb_prev_actions=j)

    # fit the model
    all_features = all_features.loc[:, ~all_features.columns.str.contains('^Unnamed')]
    all_labels = all_labels.loc[:, ~all_labels.columns.str.contains('^Unnamed')]
    VAEP_model.fit(all_features, all_labels)
    return VAEP_model


def v3_model_train(j, k, earliest_80_percent_df):
    try:
        train_features = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_features/V3/train/{j}.csv")
        train_labels = pd.read_csv(f"C:/Users/thorp/Documents/ACC_VAEP_labels/V3/train/{k}.csv")
    except FileNotFoundError:
        train_features = v3_acc_VAEP_features(data=earliest_80_percent_df, j=j)
        train_labels = v3_acc_VAEP_labels(data=earliest_80_percent_df, k=k)
    train_features = train_features.loc[:, ~train_features.columns.str.contains('^Unnamed')]
    train_labels = train_labels.loc[:, ~train_labels.columns.str.contains('^Unnamed')]

    VAEP_model = VAEP(nb_prev_actions=j)

    # Testing removal of features
    # train_features = train_features.drop("actiontype_shot_result_success_a0", axis=1)
    import time

    start_time = time.time()
    VAEP_model.fit(train_features, train_labels)
    end_time = time.time()

    execution_time = end_time - start_time
    print("Execution time: {:.2f} seconds".format(execution_time))

    return VAEP_model


if __name__ == "__main__":
    make_acc_matches()
    acc_v3_to_spadl()
    test_j = 3
    test_k = 6
    test_earliest_80_percent_df, test_most_recent_20_percent_df = train_test_data()
    v2_acc_VAEP_features(j=test_j, data=test_most_recent_20_percent_df)
    v3_model = v3_model_train(j=test_j, k=test_j, earliest_80_percent_df=test_earliest_80_percent_df)
    v3_scores = get_test_score(VAEP_model=v3_model, version='V3', j=test_j, k=test_j)
    v2_model = v2_model_train(j=test_j, k=test_j, earliest_80_percent_df=test_earliest_80_percent_df)
    v2_scores = get_test_score(VAEP_model=v2_model, version='V2', j=test_j, k=test_j)
    acc_matches = requests.get(f"https://apirest.wyscout.com/v3/seasons/{seasonId}/matches",
                               auth=(username, password))
    for test_k in [3, 6, 10, 13]:
        v2_acc_VAEP_labels(test_earliest_80_percent_df, k=test_k)
        v3_acc_VAEP_labels(test_earliest_80_percent_df, k=test_k)
        v2_acc_VAEP_labels(test_most_recent_20_percent_df, k=test_k, test="test")
        v3_acc_VAEP_labels(test_most_recent_20_percent_df, k=test_k, test="test")

    all_scores = pd.DataFrame(columns=['j', 'k', 'version', 'model_type', 'brier', 'auroc'])
    for _ in tqdm(range(0, 5)):
        for test_j in tqdm(range(3, 10, 3)):
            for test_k in tqdm([3, 6, 10, 13]):
                v3_model = v3_model_train(j=test_j, k=test_k, earliest_80_percent_df=test_earliest_80_percent_df)
                v3_scores = get_test_score(v3_model, 'V3', test_j, test_k)
                for key, values in v3_scores.items():
                    all_scores.loc[len(all_scores)] = [test_j, test_k, 'V3', key, values['brier'], values['auroc']]
        for test_j in tqdm(range(3, 10, 3)):
            for test_k in tqdm([3, 6, 10, 13]):
                v2_model = v2_model_train(j=test_j, k=test_k, earliest_80_percent_df=test_earliest_80_percent_df)
                v2_scores = get_test_score(v2_model, 'V2', test_j, test_k)
                for key, values in v2_scores.items():
                    all_scores.loc[len(all_scores)] = [test_j, test_k, 'V2', key, values['brier'], values['auroc']]

    grouped_table = all_scores.groupby(['j', 'k', 'version', 'model_type']).mean().reset_index()
    pivot_df = grouped_table.pivot_table(index=['j', 'k', 'model_type'],
                                         columns='version',
                                         values=['brier', 'auroc'],
                                         aggfunc='mean')

    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    all_scores.to_csv(f"C:/Users/thorp/Documents/VAEP_test_scores.csv")
