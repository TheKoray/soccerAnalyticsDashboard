import pandas as pd
import dash
#import dash_table
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from plotly.colors import sequential
import statsmodels.formula.api as smf
from scipy.stats import poisson,skellam
import statsmodels.api as sm
from collections import defaultdict

from getData import *
from eloRating import * 

gData = getData()
elo = eloRating()

def mainPage():

    tableStats = elo.getTableStats_().drop('EloRating', axis=1)
    tableStats['Change'] = tableStats['Change'].apply(lambda x: round(x,2))

    df_img = pd.read_csv("tff_team_img.csv")
    df_img.drop('Unnamed: 0', axis=1, inplace = True)

    result = tableStats.copy()
    #result = result.set_index('Team')
    for i in result.index:
        for k,v in df_img.iterrows():
            if i == v['Home']:
                result.loc[i,'Img'] = v['Img']

    result = result.reset_index("Team").rename(columns = {'index': 'Team'})
    result = result.reindex(["Img",'Team',"OM" ,"Win", "Draw", "Lost", 'AG', 'YG', 'Point', 'MP','Change', 'Form'], axis=1)

    return result

def ratingPage():

    dfRAting = pd.read_csv("team_rating.csv").drop("Unnamed: 0", axis=1).drop("EloRating", axis=1).set_index("Team")

    df_img = pd.read_csv("tff_team_img.csv")
    df_img.drop('Unnamed: 0', axis=1, inplace = True)

    #result = result.set_index('Team')
    for i in dfRAting.index:
        for k,v in df_img.iterrows():
            if i == v['Home']:
                dfRAting.loc[i,'Img'] = v['Img']

    return dfRAting.reset_index().reindex(['Img', 'Team', 'Rating'], axis=1)

def predictionTable():
    
    #pred_table = pd.read_csv("predictTable.csv").drop("Unnamed: 0", axis=1)
    pred_table = pd.read_csv("predictTable.csv")
    pred_table["xPTS"] = pred_table['xPTS'].apply(lambda x: round(x,2))

    return pred_table

def simulationPage():
    sim_df = pd.read_csv("simDf.csv")
    table = dbc.Table.from_dataframe(sim_df, striped=True, bordered=True, hover=True, index=False, size = 'sm')
    return table

def format_form_column(value):

    #form kolundaki string değerleri tek tek renklendirmek için o kolonu html formatına çeviren fonksiyon

    colors = {'W': 'green', 'L': 'red', 'D': 'gray'}
    
    return ''.join([f'<span style="color: {colors[char]}; font-weight: bold;">{char}</span>' for char in value])


# Dinamik color scale hesaplama fonksiyonu tüm kolonlar için
def generate_color_scale_for_all_columns(df, excluded_columns=[], color_scale=sequential.Reds):
    """
    DataFrame'deki tüm sayısal kolonlara renk skalası uygular.
    :param df: DataFrame
    :param excluded_columns: Renk skalasından hariç tutulacak kolonların listesi
    :param color_scale: Kullanılacak renk skalası (Plotly renk skalaları kullanılabilir)
    :return: style_data_conditional listesi
    """
    style_conditions = []

    for column_name in df.columns:
        if column_name in excluded_columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            continue  # Sadece sayısal sütunlara uygulanır

        # Min ve max değerleri al
        min_val = df[column_name].min()
        max_val = df[column_name].max()

        # Plotly renk skalasından renkler oluştur
        colors = color_scale

        # Her değere uygun rengi eşleştir
        for val in df[column_name].unique():
            norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0  # Normalize değer (0 ile 1 arasında)
            color_index = int(norm_val * (len(colors) - 1))  # Normalize değere göre renk seçimi
            style_conditions.append({
                "if": {"filter_query": f"{{{column_name}}} = {val}", "column_id": column_name},
                "backgroundColor": colors[color_index],
                "color": "white" if color_index > len(colors) / 2 else "black"  # Yazı rengi kontrastına göre
            })

    return style_conditions

# Color scale uygulaması için fonksiyon Tek kolon için
def generate_color_scale(df, column_name, color_map_name="Reds", num_bins=10, ascending=True):
    """
    Dinamik bir renk skalasını min-max değerlerine göre uygular ve gradyanı ascending (artan) veya descending (azalan) yapar.
    :param column_name: Hedef kolonun adı
    :param color_map_name: Matplotlib renk haritası adı (ör. "Reds", "Blues", "Greens")
    :param num_bins: Renk segment sayısı (varsayılan 10)
    :param ascending: Gradyan renginin düşükten yükseğe (True) ya da yüksekte düşük (False) olması
    :return: style_data_conditional listesi
    """
    # Renk haritasını oluştur
    cmap = plt.get_cmap(color_map_name)
    
    # Min ve max değerlerini al
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    
    # Renkleri belirle
    color_bins = [cmap(i / (num_bins - 1)) for i in range(num_bins)]  # 0'dan 1'e kadar eşit aralıklarla renkler
    
    # Gradyanı ters çevirmek için renk listesini ters çevir
    if not ascending:
        color_bins = color_bins[::-1]
    
    color_rgba = [f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})" 
                  for r, g, b, a in color_bins]  # RGBA formatına dönüştür
    
    return [
        {
            'if': {
                'filter_query': f'{{{column_name}}} >= {min_value + (i / num_bins) * (max_value - min_value)} && '
                                f'{{{column_name}}} <= {min_value + ((i + 1) / num_bins) * (max_value - min_value)}',
                'column_id': column_name,
            },
            'backgroundColor': color_rgba[i],
            'color': 'black'
        } for i in range(num_bins)
    ]

def colorScaleStrColumns(df, excluded_columns=[], color_scale=sequential.Reds):
    """
    DataFrame'deki tüm sayısal kolonlara renk skalası uygular.
    :param df: DataFrame
    :param excluded_columns: Renk skalasından hariç tutulacak kolonların listesi
    :param color_scale: Kullanılacak renk skalası (Plotly renk skalaları kullanılabilir)
    :return: style_data_conditional listesi
    """
    style_conditions = []

    for column_name in df.columns:
        column_name = str(column_name)
        
        if column_name in excluded_columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            continue  # Sadece sayısal sütunlara uygulanır

        # Min ve max değerleri al
        min_val = df[column_name].min()
        max_val = df[column_name].max()

        # Plotly renk skalasından renkler oluştur
        colors = color_scale

        # Her değere uygun rengi eşleştir
        for val in df[column_name].unique():
            norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0  # Normalize değer (0 ile 1 arasında)
            color_index = int(norm_val * (len(colors) - 1))  # Normalize değere göre renk seçimi
            style_conditions.append({
                "if": {"filter_query": f"{{{column_name}}} = {val}", "column_id": column_name},
                "backgroundColor": colors[color_index],
                "color": "white" if color_index > len(colors) / 2 else "black"  # Yazı rengi kontrastına göre
            })

    return style_conditions

def colorScaleStrColumnsV2(df, excluded_columns=[], color_scale=sequential.Reds):
    """
    DataFrame'deki tüm sayısal kolonlara renk skalası uygular.
    :param df: DataFrame
    :param excluded_columns: Renk skalasından hariç tutulacak kolonların listesi
    :param color_scale: Kullanılacak renk skalası (Plotly renk skalaları kullanılabilir)
    :return: style_data_conditional listesi
    """
    style_conditions = []

    # Dash, column_id'nin string olmasını ister, bunu garanti edelim
    df.columns = df.columns.astype(str)

    for column_name in df.columns:
        if column_name in excluded_columns or not pd.api.types.is_numeric_dtype(df[column_name]):
            continue  # Sadece sayısal sütunlara uygulanır

        # Min ve max değerleri al
        min_val = df[column_name].min()
        max_val = df[column_name].max()

        # Eğer tüm değerler aynıysa, renk atamaya gerek yok
        if min_val == max_val:
            continue

        # Plotly renk skalasından renkler oluştur
        colors = color_scale

        # Her değere uygun rengi eşleştir
        for val in df[column_name].dropna().unique():  # NaN değerleri çıkardık
            if not isinstance(val, (int, float)):  
                continue  # Sayısal olmayanları atla

            norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0
            color_index = int(norm_val * (len(colors) - 1))

            style_conditions.append({
                "if": {
                    "filter_query": f"{{{column_name}}} = {val}" if isinstance(val, (int, float)) else f"{{{column_name}}} = '{val}'",
                    "column_id": str(column_name)  # column_id her zaman string olmalı!
                },
                "backgroundColor": colors[color_index],
                "color": "white" if color_index > len(colors) / 2 else "black"
            })

    return style_conditions
def getEurope():
    #result : simülasyon DataFrame
    #result = result.reset_index().drop("Img", axis=1)

    res = pd.read_csv("simDfImg.csv").drop("Img", axis=1).set_index("Team").copy()
    
    res = res.assign(TITLE = res["1"])\
    .assign(UCL = res.groupby("Team")[["1","2"]].sum().apply(lambda x: sum(x), axis=1))\
    .assign(UEL = res.groupby("Team")[["3"]].sum().apply(lambda x: sum(x), axis=1))\
    .assign(UECL = res.groupby("Team")[["4"]].sum().apply(lambda x: sum(x), axis=1))\
    .assign(REL = res.groupby("Team")[["16","17","18","19"]].sum().apply(lambda x: sum(x), axis=1))

    return res[['TITLE', 'UCL', 'UEL', 'UECL', 'REL']]

def modelPoisson():
    play_df = gData.getNewData(played = True)
    
    goal_model_data = pd.concat([play_df[['Home','Away','HomeScore']].assign(home=1).rename(
            columns={'Home':'team', 'Away':'opponent','HomeScore':'goals'}),
           play_df[['Away','Home','AwayScore']].assign(home=0).rename(
            columns={'Away':'team', 'Home':'opponent','AwayScore':'goals'})])

    poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data,
                            family=sm.families.Poisson()).fit()
    return poisson_model


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam,
                                                           'opponent': awayTeam, 'home': 1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam,
                                                           'opponent': homeTeam, 'home': 0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals + 1)] for team_avg in
                 [home_goals_avg, away_goals_avg]]
    
    return (np.outer(np.array(team_pred[0]), np.array(team_pred[1])))

def predictOutcome(model,home_team, away_team):

    max_goals=5
    score_matrix = pd.DataFrame(simulate_match(model, home_team, away_team,max_goals))\
                   .applymap(lambda x: round(x, 2))

    h=np.sum(np.tril(score_matrix, -1))
    d=np.sum(np.diag(score_matrix))
    a=np.sum(np.triu(score_matrix, 1))

    homewin=np.sum(np.tril(score_matrix, -1))
    draw=np.sum(np.diag(score_matrix))
    awaywin=np.sum(np.triu(score_matrix, 1))
    
    return round(homewin,2), round(draw,2), round(awaywin,2)

def getWeekMatch(week):
    
    predWeek = pd.read_csv("fikstür.csv")

    return predWeek.loc[predWeek["Hafta"].isin([week])]

def getWeeklyPredict(foot_model, df : pd.DataFrame):

    pred = defaultdict(list)
    for home, away in zip(df['Home'], df['Away']):

        h, d, a = predictOutcome(model = foot_model,home_team = home,  away_team = away)

        pred['Home'].append(home)
        pred['HomeProb'].append(round(h* 100,2))
        pred['DrawProb'].append(round(d * 100, 2))
        pred['AwayProb'].append(round(a * 100, 2))
        pred['Away'].append(away)

    return pd.DataFrame(pred)

def getScoreMatrix(homeTeam, awayTeam):

    poisson_model = modelPoisson()
    score_matrix = pd.DataFrame(simulate_match(foot_model = poisson_model, homeTeam = homeTeam, awayTeam =  awayTeam,max_goals = 5))\
                    .applymap(lambda x: round(x*100, 2))

    return score_matrix

