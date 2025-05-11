import pandas as pd
import numpy as np
import random
from eloRating import *
from getData import *
from soccerMetric import * 

elo = eloRating()
gData = getData()
soccer = soccerMetric()

table = elo.getTableStats_().reset_index()
#wdfNew = gData.getFikstür(week = 34)

df = soccer.uptadeEloRating_().reset_index().set_index("Team")
elo_dict = {team: col['EloRating']for team, col in df.iterrows()}

def getFixtures(week):

    return gData.getFikstür(week = week)

# Maç simülasyonu fonksiyonu
def simulate_match(elo_home, elo_away, home_advantage=65):

    elo_home += home_advantage
    p_home = 1 / (1 + 10 ** ((elo_away - elo_home) / 400))
    p_draw = max(0.1, 0.3 - abs(elo_home - elo_away) / 1600)
    p_draw = min(p_draw, 0.35)
    p_home -= p_draw / 2
    p_away = 1 - p_home - p_draw
    result = random.choices(["home", "draw", "away"], [p_home, p_draw, p_away])[0]
    if result == "home": return 3, 0, random.randint(1, 3)
    elif result == "away": return 0, 3, -random.randint(1, 3)
    else: return 1, 1, 0

# Sezon simülasyonu
def simulate_season(puan_df, fikstur_df, elo_dict):

    #fikstur_df = getFixtures(week=week)
    
    table = {row["Team"]: {"points": row["Point"], "gd": 0} for _, row in puan_df.iterrows()}

    for _, row in fikstur_df.iterrows():

        home, away = row["Home"], row["Away"]
        
        if home in elo_dict and away in elo_dict:
            pts_home, pts_away, goal_diff = simulate_match(elo_dict[home], elo_dict[away])
            table[home]["points"] += pts_home
            table[away]["points"] += pts_away
            table[home]["gd"] += goal_diff
            table[away]["gd"] -= goal_diff

    return sorted(table.items(), key=lambda x: (x[1]["points"], x[1]["gd"]), reverse=True)

# Lig sıralama dağılımı (100 simülasyonla hızlı sonuç)
def run_rank_distribution_simulations(puan_df, fikstur_df, elo_dict, n_sim=100):

    teams = list(puan_df["Team"])
    rank_dist = pd.DataFrame(0, index=teams, columns=list(range(1, len(teams) + 1)))
    
    for _ in range(n_sim):
        season = simulate_season(puan_df, fikstur_df, elo_dict)
        for rank, (team, _) in enumerate(season, start=1):
            rank_dist.loc[team, rank] += 1

    rank_dist = (rank_dist / n_sim).round(2)
    return rank_dist.sort_index()

# Simülasyonu çalıştır
def runSimulation(puan_df, fikstur_df, elo_dict):

    rank_distribution_df = run_rank_distribution_simulations(puan_df = puan_df, fikstur_df = fikstur_df, 
                                                             elo_dict = elo_dict, n_sim=10000)
    sim_df = table.set_index("Team").merge(rank_distribution_df,left_index=True, right_index=True).reindex(np.arange(1,20), axis=1)

    sim_df = sim_df.reset_index().rename(columns = {"index": "Team"}).set_index("Team")

    return sim_df

def calculate_team_outcomes(rank_distribution_df, puan_df, fikstur_df, elo_dict, n_sim=10000):
    
    teams = list(puan_df["Team"])
    xpts_dict = dict.fromkeys(teams, 0)

    # xPTS hesaplamak için n_sim sezonu simüle et
    for _ in range(n_sim):
        # Başlangıç puanlarını ve fikstürü kopyala
        temp_table = {row["Team"]: row["Point"] for _, row in puan_df.iterrows()}
        for _, row in fikstur_df.iterrows():
            home, away = row["Home"], row["Away"]
            if home in elo_dict and away in elo_dict:
                pts_home, pts_away, _ = simulate_match(elo_dict[home], elo_dict[away])
                temp_table[home] += pts_home
                temp_table[away] += pts_away
        # Her takım için topla
        for team in teams:
            xpts_dict[team] += temp_table[team]

    # Ortalama xPTS'yi al
    xpts_dict = {team: round(xpts / n_sim, 2) for team, xpts in xpts_dict.items()}

    # Avrupa (% ilk 4), Şampiyonluk (%1), Düşme hattı (%17-19) olasılıklarını hesapla
    outcome_df = pd.DataFrame(index=teams)
    outcome_df["xPTS"] = pd.Series(xpts_dict)
    outcome_df["Title"] = rank_distribution_df[1] * 100
    outcome_df["Europe"] = rank_distribution_df.loc[:, 1:4].sum(axis=1) * 100
    outcome_df["Relegation"] = rank_distribution_df.loc[:, 17:19].sum(axis=1) * 100

    return outcome_df.sort_values("xPTS", ascending=False)

def runExpectedOutcome(rank_distribution_df, puan_df, fikstur_df, elo_dict):

    outcomes = calculate_team_outcomes(rank_distribution_df = rank_distribution_df, puan_df= puan_df, fikstur_df = fikstur_df, elo_dict = elo_dict, n_sim=10000)
    outcomes = outcomes.rename_axis("Team", axis=0)
    
    return outcomes