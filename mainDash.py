import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash.dependencies import State
import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from helperDash import mainPage, ratingPage, simulationPage, generate_color_scale_for_all_columns, generate_color_scale,getWeekMatch, getWeeklyPredict
from helperDash import modelPoisson, getScoreMatrix, colorScaleStrColumnsV2, format_form_column, predictionTable
from plotly.colors import sequential
from eloRating import * 

# Dash uygulamasını başlat
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Örnek DataFrame (gerçek verinizi kullanın)

# Sidebar 1 (Dropdown ile filtreleme)
SIDEBAR_STYLE = {
    "position": "fixed",  
    "top": 0,           
    "left": 0,
    "bottom": 0,      
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}   
#f8f9fa
# Main content style
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "18rem",
    "padding": "2rem 1rem",
}

# Sidebar 1: Dropdown ve Navigation Links
sidebar = html.Div(
    [
        html.H2("SUPER LEAGUE", className="display-4"),
        html.Hr(),
        html.P("TSL ANALYTICS", className="lead"),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Team Rating", href="/page-1", active="exact"),
                dbc.NavLink("Prediction", href="/page-2", active="exact"),
                dbc.NavLink("Fixtures", href="/page-3", active="exact"),
                dbc.NavLink("Simulation", href="/page-4", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)
fikstür = pd.read_csv("fikstür.csv").drop("Unnamed: 0", axis=1) # fikstür page için dışarda tanımladık çünkü callback de local hatası vermemesi gerekiyor
dfRAting = pd.read_csv("team_rating.csv").drop("Unnamed: 0", axis=1).drop("EloRating", axis=1)

# Layout
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Page Content for each route
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):

    if pathname == "/":

        df = mainPage()
        #team ve form kolonlarını html formatına dönüştürdüğümüz için presentation : markdown yaptık o iki kolon için.
        columns = [{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'Img' or col == "Form" else {'name': col, 'id': col} for col in df.columns]
        # Resim kolonunu Markdown formatında işlemek için HTML ekleme
        df['Img'] = df['Img'].apply(lambda url: f"![img]({url})") #image'leri html formatına getiriyoruz


        style_data_conditional = generate_color_scale(df =df, column_name = "Change", color_map_name="Reds", num_bins=10, ascending=False) #renk scalasını ayarladık. Change için gredient coloru ascending yaptık
        style_data_conditional_2 = generate_color_scale(df = df, column_name = "Point", color_map_name="Greens", num_bins=10)
        style_data_conditional.append({
                                        "if": {"column_id": "Team"}, #team kolonunu fontu bold yaptık
                                        "fontWeight": "bold",
                                        "backgroundColor": "white"  # Arka planı hafif gri yapabilirsiniz
                                    })
        df["Form"] = df["Form"].apply(lambda x: format_form_column(value = x))
        return html.Div([
                        dash_table.DataTable(
                            id="datatable",
                            columns= columns,
                            data=df.to_dict('records'),
                            style_data_conditional = style_data_conditional + style_data_conditional_2,
                            style_table={
                                'width': '100%',  # Tablo genişliğini yüzde 100 yap
                                'overflowX': 'hidden'  # Scroll'u kaldır
                            },
                            style_cell={
                                        'textAlign': 'center',
                                        'padding': '0.5px', # Hücre içindeki boşluğu küçült
                                        'maxWidth': '100px', #150px
                                        'whiteSpace': 'normal',
                                        'lineHeight': '1.2', # Hücre içindeki Satır yüksekliğini ayarla.
                                    },
                            style_cell_conditional=[
                                {'if': {'column_id': c}, 'width': f'{100/len(df.columns)}%'} for c in df.columns  # Her sütun için eşit genişlik
                            ],
                            style_header={'backgroundColor': 'white', #columnsların stylerini ayarlar
                                          'color': 'black', 'fontWeight': "bold"},
                            css=[{ #tabloda ki image'lerin görüntülerini ayarlar
                                    'selector': 'img', #Resimleri Satır İçinde Dikey Olarak Ortalamak İçin display: flex Kullanımı
                                    'rule': 'height: 25px; display: flex; justify-content: center; align-items: center; margin: auto;'
                                }],
                                markdown_options={"html": True} #görüntü için html = True dedik
                        )
                    ], style={'width': '100%'})

    elif pathname == "/page-1":

        ratingTable = ratingPage()
        columns = [{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'Img'  else {'name': col, 'id': col} for col in ratingTable.columns]
        ratingTable['Img'] = ratingTable['Img'].apply(lambda url: f"![img]({url})")
        ratingColConditional = generate_color_scale(df = ratingTable, column_name = "Rating", color_map_name="Greens", num_bins=10)

        return html.Div([dash_table.DataTable(id = "datatable",
                                              columns = columns,
                                              data = ratingTable.to_dict('records'),
                                              style_data_conditional = ratingColConditional, #rating kolonuna gradient color scale uyguladık
                                              style_table={
                                                        'width': '30%',  # Tablo genişliğini yüzde 100 yap
                                                        'overflowX': 'hidden'  # Scroll'u kaldır
                                                    },
                                                style_cell={
                                                            'textAlign': 'center',
                                                            'padding': '0.5px', # Hücre içindeki boşluğu küçült
                                                            'maxWidth': '150px',
                                                            'whiteSpace': 'normal',
                                                            'lineHeight': '12px', # Hücre içindeki metin yüksekliğini azalt
                                                            },
                                                style_cell_conditional=[
                                                                        {'if': {'column_id': c}, 'width': f'{100/len(ratingTable.columns)}%'} for c in ratingTable.columns  # Her sütun için eşit genişlik
                                                                       ],
                                                style_header={'backgroundColor': 'white', #columnsların stylerini ayarlar
                                                              'color': 'black', 'fontWeight': "bold"},
                                                css=[{ #tabloda ki image'lerin görüntülerini ayarlar
                                                        'selector': 'img', #Resimleri Satır İçinde Dikey Olarak Ortalamak İçin display: flex Kullanımı
                                                        'rule': 'height: 25px; display: flex; justify-content: center; align-items: center; margin: auto;'
                                                    }],
                                            markdown_options={"html": True} #görüntü için html = True dedik
                                            )
                                        ])
    elif pathname == "/page-2":
        return html.Div([
                            html.H1("PREDICTIONS", style={"textAlign": "center",
                                                          'alignItems': 'center',  #  ortalamak için
                                                          'justifyContent': 'flex-start',  # üst kısma hizala
                                                          }),
                            # Dropdownları yan yana koymak için flexbox düzeni
                            html.Div([
                                dcc.Dropdown(
                                    id='dropdown-pred-week',  # Benzersiz ID
                                    options=[{'label': week, 'value': week} for week in fikstür['Hafta'].unique()],
                                    value=None,  # Default değer
                                    placeholder="Select a week",
                                    style={'width': '80%', 'margin': '0px 5px'}),
                                dcc.Dropdown(
                                    id='dropdown-home-team',  # Benzersiz ID
                                    options=[{'label': team, 'value': team} for team in dfRAting['Team'].unique()],
                                    value=None,  # Default değer
                                    placeholder="Select a home team",
                                    style={'width': '80%', 'margin': '0px 5px'}),
                                dcc.Dropdown(
                                    id='dropdown-away-team',  # Benzersiz ID
                                    options=[{'label': team, 'value': team} for team in dfRAting['Team'].unique()],
                                    value=None,  # Default değer
                                    placeholder="Select a away team",
                                    style={'width': '80%', 'margin': '0px 5px'})
                            ], style={
                                'display': 'flex',  # Flexbox düzeni
                                'gap': '5px',  # Aradaki boşluk
                                'justify-content': 'center',  # Ortaya hizala
                                'marginBottom': '10px',  # Alt boşluk. dropdown ile altında oluşan DataTable arasında ki boşluk
                                'width' : '80%'
                                }
                            ),

                            # Butonları yan yana hizalamak için flexbox düzeni
                            html.Div([
                                dbc.Button("Match", id="btn-filter-match", outline=True, color="danger", style={'width': '48%'}),
                                dbc.Button("Goals", id="btn-filter-goals", outline=True, color="danger", style={'width': '48%'}),
                                dbc.Button("Table", id="btn-filter-table", outline=True, color="danger", style={'width': '48%'}),
                            ], style={
                                'display': 'flex',
                                'gap': '1%',
                                'justify-content': 'center',
                                'marginBottom': '10px', # Alt boşluk. dropdown ile altında oluşan DataTable arasında ki boşluk
                                'width' : '80%',
                            }),

                            # Sonuçları göstermek için placeholder
                            html.Div(id='filtered-data-prediction', style={"marginTop": "30px"})
                        ], style={
                                'display': 'flex',
                                'flexDirection': 'column',  # Dikey düzen
                                'alignItems': 'center',  #  ortalamak için
                                'justifyContent': 'flex-start',  # üst kısma hizala
                                'marginLeft': '40px',  # Sidebar için alan bırak
                                'height': '100vh',  # Yükseklik tamamen kaplasın
                                'padding': '20px',  # İçerik kenar boşlukları
                                            }
                        )
    
    elif pathname == "/page-3":

        return html.Div([
        html.H1("Fixtures", style={"textAlign": "center"}),
        html.Div([
            dcc.Dropdown(
                id='dropdown-fikstür',
                options=[{'label': week, 'value': week} for week in fikstür["Hafta"].unique()],
                value=None,
                placeholder="Select a week",
                style={'width': '80%', 'margin': '0px 5px'}
            ),
        ], style={'display': 'flex',     #Flexbox ile düzenlem
                   'textAlign': 'center', 
                  'justify-content': 'flex-start', # Yatayda üste hizala
                 'align-items': 'center',  # Dikeyde ortala
                 'width': '80%', #dropdown genişliği
                 }),
        html.Div(id='filtered-data-fixture', style={"marginTop": "30px"}) #bu kodda ki id ile aşağıda ki callback decarotarundaki ilk parametre aynı olur ki onları bağlantısını bu şekilde kurarız.
    ], style={
            'display': 'flex',
            'flexDirection': 'column',  # Dikey düzen
            'alignItems': 'center',  #  ortalamak için
            'justifyContent': 'flex-start',  # üst kısma hizala
            'marginLeft': '250px',  # Sidebar için alan bırak
            'height': '100vh',  # Yükseklik tamamen kaplasın
            'padding': '20px',  # İçerik kenar boşlukları
            }
    )

    elif pathname == "/page-4":

        sim_df = pd.read_csv("simDfImg.csv")
        columns = [{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'Img' else {'name': col, 'id': col} for col in sim_df.columns]
        # Resim kolonunu Markdown formatında işlemek için HTML ekleme
        sim_df['Img'] = sim_df['Img'].apply(lambda url: f"![img]({url})")


        style_data_conditional = generate_color_scale_for_all_columns(sim_df, excluded_columns=["Team"], color_scale=sequential.Reds) #renk scalasını ayarladık
        style_data_conditional.append({
                                        "if": {"column_id": "Team"}, #team kolonunu fontu bold yaptık
                                        "fontWeight": "bold",
                                        "backgroundColor": "white"  # Arka planı hafif gri yapabilirsiniz
                                    })
        return html.Div([
                        dash_table.DataTable(
                            id="datatable",
                            columns= columns,
                            data=sim_df.to_dict('records'),
                            style_data_conditional = style_data_conditional,
                            style_table={
                                'width': '120%',  # Tablo genişliğini yüzde 100 yap
                                'overflowX': 'hidden'  # Scroll'u kaldır
                            },
                            style_cell={
                                        'textAlign': 'center',
                                        'padding': '0.5px', # Hücre içindeki boşluğu küçült
                                        'maxWidth': '150px',
                                        'whiteSpace': 'normal',
                                        'lineHeight': '12px', # Hücre içindeki metin yüksekliğini azalt
                                    },
                            style_cell_conditional=[
                                {'if': {'column_id': c}, 'width': f'{100/len(sim_df.columns)}%'} for c in sim_df.columns  # Her sütun için eşit genişlik
                            ],
                            style_header={'backgroundColor': 'white', #columnsların stylerini ayarlar
                                          'color': 'black', 'fontWeight': "bold"},
                            css=[{ #tabloda ki image'lerin görüntülerini ayarlar
                                    'selector': 'img',
                                    'rule': 'height: 25px; display: flex; justify-content: center; align-items: center; margin: auto;'
                                }],
                                markdown_options={"html": True} #görüntü için html = True dedik
                        )
                    ], style={'width': '100%'})

    return html.Div([  # 404 page
        html.H1("404: Not Found", className="text-danger"),
        html.Hr(),
        html.P(f"The pathname {pathname} was not recognized."),
    ])

#prediction callback
@app.callback(
    Output('filtered-data-prediction', 'children'),
    [Input('btn-filter-match', 'n_clicks'), #input tupple'ında ki ilk değer prediction page'inde ki ilk button'un id'si ile aynı. Yukarıda tanımlardığımız button dropdownların idlerini kullanacagımız
     Input('btn-filter-goals', 'n_clicks'), #-callback'te inputun ilk değeri olarak yazarız. Böyleleik id'ler üzerinden connect kurarız.
    Input('btn-filter-table', 'n_clicks'),
    Input('dropdown-pred-week', 'value'),
    Input('dropdown-home-team', 'value'), #-callback'te inputun ilk değeri olarak yazarız. Böyleleik id'ler üzerinden connect kurarız.
    Input('dropdown-away-team', 'value')]
)
def update_prediction(btn_match, btn_goals, btn_tables, selected_week, selected_home, selected_away):
    #print(f"Triggered ID: {dash.callback_context.triggered}")  # Hangi buton tetiklendi?
    #print(f"Selected Team: {selected_week}")
    # tüm buton değerlerinin return ettiği dataFrame'leri aynı isimle niteliriz. Çünkü fonksiyonun return ettiği dataTable da tek data tanımlarız. O yüzden butona göre filtreledğimiz dataFramelerin hepsi aynı isimdir
    weekDf = getWeekMatch(week = selected_week)
    poisson_model = modelPoisson() # poisson modeli helperDash modülünden çağırdık.
    filter_data = pd.DataFrame()
    columns = []
    style_data_conditional = [] #local variable hatası almamak için if-elseDe kullanılan tüm variableları burada tanımladık.
    style_table = {}

    triggered = dash.callback_context.triggered
    triggered_id = triggered[0]['prop_id'].split('.')[0]
    if not selected_week: # dropdownlar için if else yazdık. Çğnkğ table butonu sadece buton basıldıgında predict return etmesi gerekiyor. Hiçbit şekilde dropdown dan değer almamıza gerek yoktur. ondan
        #ondan dolayı if not selected_week dedik ki hiçbir dropdown'a değer vermeden direk table butonuna bastıgımızda prediction table return edebilelelim
        if triggered_id == "btn-filter-table": #table butonuna basınca prediction tablosu return edilir.

            filter_data = predictionTable()
            
            style_data_conditional = generate_color_scale_for_all_columns(filter_data, excluded_columns=["Team"], color_scale=sequential.Reds) #renk scalasını ayarladık
            columns = [{'name': col, 'id': col, 'presentation': 'markdown'} if col == 'Img' else {'name': col, 'id': col} for col in filter_data.columns]
            filter_data['Img'] = filter_data['Img'].apply(lambda url: f"![img]({url})") #Resim kolonunu Markdown formatında işlemek için HTML ekleme
        
        elif triggered_id  == 'btn-filter-goals': #goals butonuna basınca gol olasılıkları tablosunu return eder
            
            filter_data = getScoreMatrix(homeTeam = selected_home, awayTeam = selected_away) #score_matrixi seçtik. Gol olasılıklarını veren matris
            filter_data.reset_index(inplace = True) #indexi görebilmek için indexi kolona alıyoruz
            filter_data = filter_data.rename(columns = {'index' : 'H/A'}) #index kolona geldi ismini H/A yaptık. index kolonunda 1den5'e değerler vardır. onların görünmesini istedik

            style_data_conditional = colorScaleStrColumnsV2(filter_data, excluded_columns=[], color_scale=sequential.Reds) #score matris için renk scalasını ayarladık 
            columns = [{"name": str(i), "id": str(i)} for i in filter_data.columns] #dataTable kolonları string olmalı. Bizim score matrisimizin kolonları int oldugu ıcın str() yaptık kolonları.
            style_table = {'width': '160%', 'height' : '90%','overflowX': 'hidden','overflowY': 'hidden'}
    else:
        # Boş tetikleme kontrolü
        #triggered = dash.callback_context.triggered
        #triggered_id = triggered[0]['prop_id'].split('.')[0]
        if not triggered:
            return "No button has been clicked yet."
        
        if triggered_id == 'btn-filter-match':

            filter_data = getWeeklyPredict(foot_model = poisson_model, df = weekDf)
            columns = [{"name": i, "id": i} for i in filter_data.columns]
        """
        elif triggered_id  == 'btn-filter-goals': #goals butonuna basınca gol olasılıklarını verir
            
            filter_data = getScoreMatrix(homeTeam = selected_home, awayTeam = selected_away) #score_matrixi seçtik. Gol olasılıklarını veren matris
            filter_data.reset_index(inplace = True) #indexi görebilmek için indexi kolona alıyoruz
            filter_data = filter_data.rename(columns = {'index' : 'H/A'}) #index kolona geldi ismini H/A yaptık. index kolonunda 1den5'e değerler vardır. onların görünmesini istedik

            style_data_conditional = colorScaleStrColumns(filter_data, excluded_columns= [], color_scale=sequential.Reds) #renk scalasını ayarladık
            columns = [{"name": str(i), "id": str(i)} for i in filter_data.columns]

        elif triggered_id == "btn-filter-table": #table butonuna basınca prediction tablosu return edilir.

            filter_data = predictionTable()
            columns = [{"name": i, "id": i} for i in filter_data.columns]
            
        if weekDf.empty:
            return "No data available for the selected team."""

    return dash_table.DataTable(
        id='table-filtered-prediction',
        columns= columns,
        data=filter_data.to_dict('records'),
        style_data_conditional = style_data_conditional,
        style_table= style_table,
        style_cell={
                'textAlign': 'center',
                'padding': '0.5px', # Hücre içindeki boşluğu küçült
                'maxWidth': '150px',
                'whiteSpace': 'normal',
                'lineHeight': '12px', # Hücre içindeki metin yüksekliğini azalt
                'marginLeft': '20px',  # Sidebar için alan bırak
                },
        style_cell_conditional=[
                            {'if': {'column_id': str(c)}, 'width': f'{100/len(filter_data.columns)}%'} for c in filter_data.columns  # Her sütun için eşit genişlik
                            ],
        css=[{ #tabloda ki image'lerin görüntülerini ayarlar
            'selector': 'img',
            'rule': 'height: 25px; display: flex; justify-content: center; align-items: center; margin: auto;'
            }],
        markdown_options={"html": True} #görüntü için html = True dedik
    )

#fikstür callback
@app.callback(
    Output('filtered-data-fixture', 'children'),
    [Input('dropdown-fikstür', 'value')]
)
def update_fixtures(selected_week):
    print(f"Selected Week: {selected_week}")

    if not selected_week:
        return "Please select a week."

    filtered_df = fikstür[fikstür["Hafta"] == selected_week]  # Haftaya göre filtreleme

    if filtered_df.empty:
        return "No fixtures available for the selected week."

    return dash_table.DataTable(
        id='table-filtered-fixtures',
        columns=[{"name": i, "id": i} for i in filtered_df.columns],
        data=filtered_df.to_dict('records'),
        style_table={'height': '300px', 'overflowY': 'auto'}
    )

if __name__ == "__main__":
    app.run_server(debug=True, port=8888)