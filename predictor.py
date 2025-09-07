#TODO: make it flask?

import os
from prophet import Prophet
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import dash


DATA_FOLDER = "predictor/data/"
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

diseases = [
    {'label': f.replace('.csv', '').replace('_', ' ').title(), 'value': f} 
    for f in csv_files
    ]

app = dash.Dash(__name__)
app.title = "Childhood Immunization Forecasting Dashboard"

def load_data(csv_file):
    path = os.path.join(DATA_FOLDER, csv_file)
    data_frame = pd.read_csv(path)
    vac_col = data_frame.columns[-1]
    #last column always vac rate
    data_frame = data_frame.rename(columns={"Entity": "Country", vac_col: "VaccinationRate"})
    data_frame = data_frame[["Country", "Year", "VaccinationRate"]].dropna()
    data_frame["Year"] = pd.to_datetime(data_frame["Year"], format="%Y")
    return data_frame

def forecasts(data_frame, country, periods=10):
    country_df = data_frame[data_frame['Country'] == country][['Year', 'VaccinationRate']]
    prophet_df = country_df.rename(columns={'Year': 'ds', 'VaccinationRate': 'y'})

    if len(prophet_df) < 5:  #not enough data to train
        return country_df, None

    model = Prophet()
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq='Y')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(0, 100)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(0, 100)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(0, 100)
    return country_df, forecast

@app.callback(
    Output('country-dropdown', 'options'),
    Output('country-dropdown', 'value'),
    Input('disease-dropdown', 'value')
)

def update_counts(selected_disease):
    df = load_data(selected_disease)
    countries = sorted(df['Country'].unique())
    options = [{'label': c, 'value': c} for c in countries]
    value = countries[0] if countries else None
    return options, value

@app.callback(
    Output('forecast-graph', 'figure'),
    Input('disease-dropdown', 'value'),
    Input('country-dropdown', 'value')
)

def update_graph(selected_disease, selected_country):
    df = load_data(selected_disease)
    country_df, forecast = forecasts(df, selected_country)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=country_df['Year'], y=country_df['VaccinationRate'],
        mode='lines+markers', name='Historical',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6, color='royal blue')
    ))

    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat'],
            mode='lines', name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(255,255,0,0.2)', name='Confidence Interval'
        ))

    fig.update_layout(
        title=dict(
            text=f"{selected_country} - {selected_disease.replace('.csv','').replace('_',' ').title()}",
            font=dict(family="Segoe UI, sans-serif", size=24, color="#2c3e50")
        ),
        xaxis_title=dict(
            text="Year",
            font=dict(family="Segoe UI, sans-serif", size=18, color="#2c3e50")
        ),
        yaxis_title=dict(
            text="Vaccination Rate (%)",
            font=dict(family="Segoe UI, sans-serif", size=18, color="#2c3e50")
        ),
        font=dict(family="Segoe UI, sans-serif", size=14, color="#2c3e50"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(title=dict(text='Legend', font=dict(size=14, family='Segoe UI, sans-serif')))
    )
    return fig

app.layout = html.Div(
    style={
        "padding": "50px",        
    },
    children=[
        html.H1(
            "Childhood Immunization Forecasting Dashboard",
            style={'textAlign': 'center', 'font-family': 'Segoe UI, sans-serif'}
        ),

        html.Label("Select Disease:", style={'font-weight': 'bold', 'font-family': 'Segoe UI'}),
        dcc.Dropdown(
            id='disease-dropdown',
            options=diseases,
            value=csv_files[0] if csv_files else None,
            style={
                'width': '60%',
                'margin-bottom': '20px',
                'font-family': 'Segoe UI, sans-serif'
            }
        ),

        html.Label("Select Country:", style={'font-weight': 'bold', 'font-family': 'Segoe UI'}),
        dcc.Dropdown(
            id='country-dropdown',
            style={
                'width': '60%',
                'margin-bottom': '20px',
                'font-family': 'Segoe UI, sans-serif'
            }
        ),

        html.Div(id='selected-country-display', style={'textAlign': 'center', 'font-family': 'Segoe UI'}),
        dcc.Graph(id='forecast-graph')
    ]
)

if __name__ == "__main__":
    app.run(debug=True)
