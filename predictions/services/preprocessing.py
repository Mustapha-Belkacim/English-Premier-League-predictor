import numpy as np
import pandas as pd
from datetime import datetime as dt
import itertools

# Read data from the CSV into a dataframe
# url = "http://www.football-data.co.uk/mmz4281/1718/E0.csv"

def parse_date(date):
    """Parse date as time"""
    if date == '':
        return None
    else:
        return dt.strptime(date, '%d/%m/%y').date()

def get_goals_scored(playing_stat):
    """Gets the goals scored agg arranged by teams and matchweek"""
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        HTGS = playing_stat.iloc[i]['FTHG']
        ATGS = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGS)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGS)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsScored = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsScored[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsScored[i] = GoalsScored[i] + GoalsScored[i - 1]
    return GoalsScored

def get_goals_conceded(playing_stat):
    """Gets the goals conceded agg arranged by teams and matchweek"""
    # Create a dictionary with team names as keys
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match location.
    for i in range(len(playing_stat)):
        ATGC = playing_stat.iloc[i]['FTHG']
        HTGC = playing_stat.iloc[i]['FTAG']
        teams[playing_stat.iloc[i].HomeTeam].append(HTGC)
        teams[playing_stat.iloc[i].AwayTeam].append(ATGC)

    # Create a dataframe for goals scored where rows are teams and cols are matchweek.
    GoalsConceded = pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T
    GoalsConceded[0] = 0
    # Aggregate to get uptil that point
    for i in range(2, 39):
        GoalsConceded[i] = GoalsConceded[i] + GoalsConceded[i - 1]
    return GoalsConceded

def add_gs_gc(playing_stat):
    """ adds home/away team scored/conceded goals columns to the data frame"""
    GC = get_goals_conceded(playing_stat)
    GS = get_goals_scored(playing_stat)

    j = 0
    HTGS = []
    ATGS = []
    HTGC = []
    ATGC = []

    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        HTGC.append(GC.loc[ht][j])
        ATGC.append(GC.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    # adding the columns to dataframe
    #playing_stat['HTGS'] = HTGS
    playing_stat = playing_stat.assign(HTGS=HTGS)
    playing_stat['ATGS'] = ATGS
    playing_stat['HTGC'] = HTGC
    playing_stat['ATGC'] = ATGC

    return playing_stat

def get_points(result):
    """get respective points win = 3, lose = 0, draw = 1 point"""
    if result == 'W':
        return 3
    elif result == 'D':
        return 1
    else:
        return 0

def get_cuml_points(matchres):
    """get cumulated points for every week
       :param matchres a dictionary of teams and there respetive results every week
    """
    matchres_points = matchres.applymap(get_points)
    for i in range(2, 39):
        matchres_points[i] = matchres_points[i] + matchres_points[i - 1]

    matchres_points.insert(column=0, loc=0, value=[0 * i for i in range(20)])
    return matchres_points

def get_matchres(playing_stat):
    """Create a dictionary with team names as keys
       and there match result for every week as values
    """
    teams = {}
    for i in playing_stat.groupby('HomeTeam').mean().T.columns:
        teams[i] = []

    # the value corresponding to keys is a list containing the match result
    for i in range(len(playing_stat)):
        if playing_stat.iloc[i].FTR == 'H':
            teams[playing_stat.iloc[i].HomeTeam].append('W')
            teams[playing_stat.iloc[i].AwayTeam].append('L')
        elif playing_stat.iloc[i].FTR == 'A':
            teams[playing_stat.iloc[i].AwayTeam].append('W')
            teams[playing_stat.iloc[i].HomeTeam].append('L')
        else:
            teams[playing_stat.iloc[i].AwayTeam].append('D')
            teams[playing_stat.iloc[i].HomeTeam].append('D')

    return pd.DataFrame(data=teams, index=[i for i in range(1, 39)]).T

def add_agg_points(playing_stat):
    """inserts 2 columns containing home/away team points"""
    matchres = get_matchres(playing_stat)
    cum_pts = get_cuml_points(matchres)
    HTP = []
    ATP = []
    j = 0
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])

        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HTP'] = HTP
    playing_stat['ATP'] = ATP
    return playing_stat

def get_form(playing_stat, num):
    """ get team form (how the team is doing in the last num games?)"""
    form = get_matchres(playing_stat)
    form_final = form.copy()
    for i in range(num, 39):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i - j]
            j += 1
    return form_final

def add_form(playing_stat, num):
    """add each team form to the data frame"""
    form = get_form(playing_stat, num)
    h = ['M' for i in range(num * 10)]  # since form is not available for n MW (n*10)
    a = ['M' for i in range(num * 10)]

    j = num
    for i in range((num * 10), 380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam

        past = form.loc[ht][j]  # get past n results
        h.append(past[num - 1])  # 0 index is most recent

        past = form.loc[at][j]  # get past n results.
        a.append(past[num - 1])  # 0 index is most recent

        if ((i + 1) % 10) == 0:
            j = j + 1

    playing_stat['HM' + str(num)] = h
    playing_stat['AM' + str(num)] = a

    return playing_stat

def add_form_df(playing_statistics):
    """add a team form for evry 1 to 5 games"""
    playing_statistics = add_form(playing_statistics, 1)
    playing_statistics = add_form(playing_statistics, 2)
    playing_statistics = add_form(playing_statistics, 3)
    playing_statistics = add_form(playing_statistics, 4)
    playing_statistics = add_form(playing_statistics, 5)
    return playing_statistics

def get_last(playing_stat, Standings, year):
    """gets last season standings"""
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(380):
        ht = playing_stat.iloc[i].HomeTeam
        at = playing_stat.iloc[i].AwayTeam
        HomeTeamLP.append(Standings.loc[ht][year])
        AwayTeamLP.append(Standings.loc[at][year])
    playing_stat['HomeTeamLP'] = HomeTeamLP
    playing_stat['AwayTeamLP'] = AwayTeamLP
    return playing_stat

def get_mw(playing_stat):
    """Get MatchWeek"""
    j = 1
    MatchWeek = []
    for i in range(380):
        MatchWeek.append(j)
        if ((i + 1)% 10) == 0:
            j = j + 1
    playing_stat['MW'] = MatchWeek
    return playing_stat

def get_form_points(string):
    """Gets the form points"""
    sum = 0
    for letter in string:
        sum += get_points(letter)
    return sum

# Identify Win/Loss Streaks if any.
def get_3game_wins(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0

def get_5game_wins(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0

def get_3game_lose(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0

def get_5game_lose(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0

def only_homewin(string):
    """ if the home team wins or not"""
    if string == 'H':
        return 'H'
    else:
        return 'NH'


data_frames = []
playing_statistics = []
loc = "../static/predictions/Data/"
# Get Last Year's Position as also an independent variable:
Standings = pd.read_csv(loc + "EPLStandings.csv")
Standings.set_index(['Team'], inplace=True)
Standings = Standings.fillna(18)

for year in range(2005, 2014):
    #url = "http://www.football-data.co.uk/mmz4281/" + str(year)[-2:] + str(year + 1)[-2:] + "/E0.csv"
    url = loc + str(year) + "-" + str(year + 1)[-2:] + ".csv"
    data_frame = pd.read_csv(url)
    #data_frame.Date = parse_date(data_frame.Date[1])# FIXME : doesn't convert date well

    # Gets all the statistics related to gameplay
    columns_req = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
    playing_stat = data_frame[columns_req]
    playing_stat = add_gs_gc(playing_stat)
    playing_stat = add_agg_points(playing_stat)
    playing_stat = add_form_df(playing_stat)
    # Rearranging columns
    cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTGS', 'ATGS', 'HTGC',
            'ATGC', 'HTP', 'ATP', 'HM1','HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5']
    playing_stat = playing_stat[cols]
    playing_stat = get_last(playing_stat, Standings, int(str(year)[-2:]))# TODO: add last year standing
    playing_stat = get_mw(playing_stat)
    playing_statistics.append(playing_stat)
    data_frames.append(playing_stat)

# Final data frame
dataset = pd.concat(data_frames, ignore_index=True)

dataset['HTFormPtsStr'] = dataset['HM1'] + dataset['HM2'] + dataset['HM3'] + dataset['HM4'] + dataset['HM5']
dataset['ATFormPtsStr'] = dataset['AM1'] + dataset['AM2'] + dataset['AM3'] + dataset['AM4'] + dataset['AM5']

dataset['HTFormPts'] = dataset['HTFormPtsStr'].apply(get_form_points)
dataset['ATFormPts'] = dataset['ATFormPtsStr'].apply(get_form_points)

dataset['HTWinStreak3'] = dataset['HTFormPtsStr'].apply(get_3game_wins)
dataset['HTWinStreak5'] = dataset['HTFormPtsStr'].apply(get_5game_wins)
dataset['HTLossStreak3'] = dataset['HTFormPtsStr'].apply(get_3game_lose)
dataset['HTLossStreak5'] = dataset['HTFormPtsStr'].apply(get_5game_lose)

dataset['ATWinStreak3'] = dataset['ATFormPtsStr'].apply(get_3game_wins)
dataset['ATWinStreak5'] = dataset['ATFormPtsStr'].apply(get_5game_wins)
dataset['ATLossStreak3'] = dataset['ATFormPtsStr'].apply(get_3game_lose)
dataset['ATLossStreak5'] = dataset['ATFormPtsStr'].apply(get_5game_lose)

dataset.keys()

# Get Goal Difference
dataset['HTGD'] = dataset['HTGS'] - dataset['HTGC']
dataset['ATGD'] = dataset['ATGS'] - dataset['ATGC']

# Diff in points
dataset['DiffPts'] = dataset['HTP'] - dataset['ATP']
dataset['DiffFormPts'] = dataset['HTFormPts'] - dataset['ATFormPts']

# Diff in last year positions
dataset['DiffLP'] = dataset['HomeTeamLP'] - dataset['AwayTeamLP']

# Scale DiffPts , DiffFormPts, HTGD, ATGD by Matchweek.
cols = ['HTGD','ATGD','DiffPts','DiffFormPts','HTP','ATP']
dataset.MW = dataset.MW.astype(float)

for col in cols:
    dataset[col] = dataset[col] / dataset.MW

dataset['FTR'] = dataset.FTR.apply(only_homewin)

# Testing set (2013-14 season)
testset = dataset[3040:]

dataset.to_csv(loc + "final_dataset.csv")
testset.to_csv(loc + "test.csv")

print(testset)