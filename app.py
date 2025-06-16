import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import io
import base64
import requests
import plotly.graph_objs as go
import plotly.io as pio
import csv
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

model_med = joblib.load("xgb_model_medium.pkl")
model_ext = joblib.load("xgb_model_extracoarse.pkl")
WEATHER_API_KEY = "91630cf2d06c4a2caa731112251604"

# Helper to encode matplotlib fig as base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    drift_plot_html = wind_plot_html = profile_plot_html = None
    best_table = None
    location = ''
    drone_height = 3
    droplet_type = 'medium'
    hour_start, hour_end = 8, 17
    csv_data = None
    
    # Check if we have stored form parameters in the session
    if 'last_form' in session:
        form = session['last_form']
        location = form['location']
        drone_height = form['drone_height']
        droplet_type = form['droplet_type']
        hour_start = form['hour_start']
        hour_end = form['hour_end']
        lat = form.get('lat')
        lon = form.get('lon')
        
        if lat and lon:
            latlon = f"{lat},{lon}"
        else:
            # fallback: geocode the location string
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_resp = requests.get(geo_url).json()
            if not geo_resp.get('results'):
                return render_template('index.html', error='Location not found.')
            lat = geo_resp['results'][0]['latitude']
            lon = geo_resp['results'][0]['longitude']
            latlon = f"{lat},{lon}"
        
        # Weather fetch
        def fetch_hourly_forecast(location_latlon, day_offset):
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": WEATHER_API_KEY,
                "q": location_latlon,
                "days": day_offset + 1,
                "aqi": "no"
            }
            data = requests.get(url, params=params).json()
            return data["forecast"]["forecastday"][day_offset]["hour"], data["forecast"]["forecastday"][day_offset]["date"]
        
        model = model_med if droplet_type == "medium" else model_ext
        all_data = []
        
        for day_offset in range(3):
            forecast, date_str = fetch_hourly_forecast(latlon, day_offset)
            for hour_data in forecast[hour_start:hour_end + 1]:
                time_str = hour_data["time"]
                hour_label = pd.to_datetime(time_str).strftime("%H:%M")
                weather = {
                    "Temperature": hour_data["temp_c"],
                    "Wind.Speed": hour_data["wind_mph"],
                    "Humidity": hour_data["humidity"],
                    "Dew.Point": hour_data.get("dewpoint_c", 0),
                    "wind direction": hour_data.get("wind_degree", 0)
                }
                
                def get_drift_cutoff_location(weather, model, height):
                    for loc in np.arange(0.0, 40.5, 0.5):
                        input_df = pd.DataFrame([{
                            "Height": height,
                            "Temperature": weather["Temperature"],
                            "Dew.Point": weather["Dew.Point"],
                            "Humidity": weather["Humidity"] / 100.0,
                            "Wind.Speed": weather["Wind.Speed"],
                            "Location": loc,
                            "wind direction": weather["wind direction"]
                        }])
                        drift = model.predict(input_df)[0]
                        drift = max(0, min(drift, 100))
                        if drift < 5.0:
                            return loc
                    return 40.5
                
                cutoff_location = get_drift_cutoff_location(weather, model, drone_height)
                all_data.append({
                    "Date": date_str,
                    "Hour": hour_label,
                    "Wind (mph)": weather["Wind.Speed"],
                    "Humidity (%)": weather["Humidity"],
                    "Temperature (°C)": weather["Temperature"],
                    "Dew Point (°C)": weather["Dew.Point"],
                    "Wind Dir (°)": weather["wind direction"],
                    "Drift <5% After (m)": cutoff_location,
                    "Weather Snapshot": weather
                })
        
        df_result = pd.DataFrame(all_data)
        
        # Drift Reach Chart (Plotly)
        fig1 = go.Figure()
        for date in df_result['Date'].unique():
            data = df_result[df_result['Date'] == date]
            fig1.add_trace(go.Scatter(
                x=data['Hour'],
                y=data['Drift <5% After (m)'],
                mode='lines+markers',
                name=date
            ))
        fig1.update_layout(
            title="Drift Reach by Hour",
            xaxis_title="Hour",
            yaxis_title="Drift <5% After (m)",
            legend_title="Date"
        )
        drift_plot_html = pio.to_html(fig1, full_html=False)
        
        # Wind Speed Chart (Plotly)
        fig2 = go.Figure()
        for date in df_result['Date'].unique():
            data = df_result[df_result['Date'] == date]
            fig2.add_trace(go.Scatter(
                x=data['Hour'],
                y=data['Wind (mph)'],
                mode='lines+markers',
                name=date
            ))
        fig2.update_layout(
            title="Wind Speed by Hour",
            xaxis_title="Hour",
            yaxis_title="Wind (mph)",
            legend_title="Date"
        )
        wind_plot_html = pio.to_html(fig2, full_html=False)
        
        # Drift Profile for Best Time (Plotly)
        profile_plot_html = None
        # Get top 3 best times, using wind speed as tiebreaker
        df_result['Wind_Speed'] = df_result['Wind (mph)']  # Create a copy for sorting
        best_rows = df_result.sort_values(
            by=['Drift <5% After (m)', 'Wind_Speed']
        ).head(3)
        best_rows = best_rows.drop(columns=['Wind_Speed', 'Weather Snapshot'])
        
        if not best_rows.empty:
            best_weather = df_result.loc[best_rows.index[0]]["Weather Snapshot"]
            best_date = best_rows.iloc[0]["Date"]
            best_hour = best_rows.iloc[0]["Hour"]
            
            def get_drift_profile(weather, model, height):
                drift_data = []
                for loc in np.arange(0.0, 40.5, 0.5):
                    input_df = pd.DataFrame([{
                        "Height": height,
                        "Temperature": weather["Temperature"],
                        "Dew.Point": weather["Dew.Point"],
                        "Humidity": weather["Humidity"] / 100.0,
                        "Wind.Speed": weather["Wind.Speed"],
                        "Location": loc,
                        "wind direction": weather["wind direction"]
                    }])
                    drift = model.predict(input_df)[0]
                    drift = max(0, min(drift, 100))
                    drift_data.append({"Location": loc, "Drift (%)": drift})
                return pd.DataFrame(drift_data)
            
            drift_profile = get_drift_profile(best_weather, model, drone_height)
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=drift_profile["Location"],
                y=drift_profile["Drift (%)"],
                mode='lines',
                name='Drift Profile',
                line=dict(color='black')
            ))
            fig3.add_trace(go.Scatter(
                x=drift_profile["Location"],
                y=drift_profile["Drift (%)"],
                fill='tozeroy',
                mode='none',
                fillcolor='rgba(144,238,144,0.4)',
            ))
            fig3.add_hline(y=5, line_dash="dash", line_color="red")
            fig3.update_layout(
                title="Drift Profile at Best Spraying Hour",
                xaxis_title="Drift Distance (m)",
                yaxis_title="Drift (%)",
                title_x=0.5,
                title_y=0.95,
                margin=dict(t=80),
                annotations=[
                    dict(
                        text=f"Best Time: {best_date} {best_hour}",
                        xref="paper", yref="paper",
                        x=0.5, y=1.08, showarrow=False, font=dict(size=16)
                    )
                ]
            )
            profile_plot_html = pio.to_html(fig3, full_html=False)
        
        # Prepare CSV for download
        csv_data = df_result.drop(columns=["Weather Snapshot", "Wind_Speed"]).to_csv(index=False)
        best_table = best_rows.to_html(classes="table table-striped table-bordered text-center align-middle", index=False) if not best_rows.empty else None
    
    if request.method == 'POST':
        try:
            location = request.form['location']
            drone_height = int(request.form['drone_height'])
            droplet_type = request.form['droplet_type']
            hour_start = int(request.form['hour_start'])
            hour_end = int(request.form['hour_end'])
            lat = request.form.get('lat')
            lon = request.form.get('lon')
            
            if lat and lon:
                latlon = f"{lat},{lon}"
            else:
                # fallback: geocode the location string as before
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
                geo_resp = requests.get(geo_url).json()
                if not geo_resp.get('results'):
                    return render_template('index.html', error='Location not found.')
                lat = geo_resp['results'][0]['latitude']
                lon = geo_resp['results'][0]['longitude']
                latlon = f"{lat},{lon}"
            
            # Store only form parameters in session
            session['last_form'] = {
                'location': location,
                'drone_height': drone_height,
                'droplet_type': droplet_type,
                'hour_start': hour_start,
                'hour_end': hour_end,
                'lat': lat,
                'lon': lon
            }
            
            return redirect(url_for('index'))
            
        except Exception as e:
            return render_template('index.html', error=f'An error occurred: {str(e)}')
    
    return render_template('index.html',
        drift_plot_html=drift_plot_html,
        wind_plot_html=wind_plot_html,
        profile_plot_html=profile_plot_html,
        best_table=best_table,
        location=location,
        drone_height=drone_height,
        droplet_type=droplet_type,
        hour_start=hour_start,
        hour_end=hour_end,
        csv_data=csv_data
    )

@app.route('/location-suggestions')
def location_suggestions():
    query = request.args.get('q', '')
    if not query or len(query) < 3:
        return jsonify([])
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=5&language=en&format=json"
    resp = requests.get(url).json()
    results = resp.get('results', [])
    suggestions = [
        {
            'label': f"{loc['name']}, {loc.get('admin1', '')}, {loc['country']}",
            'lat': loc['latitude'],
            'lon': loc['longitude']
        }
        for loc in results
    ]
    return jsonify(suggestions)

@app.route('/download-csv')
def download_csv():
    csv_data = request.args.get('csv_data')
    if not csv_data:
        return "No data to download", 400
    # Convert the CSV string back to bytes
    buf = io.BytesIO(csv_data.encode('utf-8'))
    buf.seek(0)
    return send_file(buf, mimetype='text/csv', as_attachment=True, download_name='drift_reach_recommendations.csv')

@app.route('/detailed-results')
def detailed_results():
    try:
        # Get form parameters from session
        form = session.get('last_form')
        if not form:
            return render_template('detailed_results.html', error='No previous results found. Please run a prediction first.')
        
        # Re-run the calculation using the stored form parameters
        location = form['location']
        drone_height = form['drone_height']
        droplet_type = form['droplet_type']
        hour_start = form['hour_start']
        hour_end = form['hour_end']
        lat = form.get('lat')
        lon = form.get('lon')
        
        if not lat or not lon:
            # If lat/lon not in session, try to geocode again
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_resp = requests.get(geo_url).json()
            if not geo_resp.get('results'):
                return render_template('detailed_results.html', error='Location not found.')
            lat = geo_resp['results'][0]['latitude']
            lon = geo_resp['results'][0]['longitude']
        
        latlon = f"{lat},{lon}"
        
        # Weather fetch
        def fetch_hourly_forecast(location_latlon, day_offset):
            url = "http://api.weatherapi.com/v1/forecast.json"
            params = {
                "key": WEATHER_API_KEY,
                "q": location_latlon,
                "days": day_offset + 1,
                "aqi": "no"
            }
            data = requests.get(url, params=params).json()
            return data["forecast"]["forecastday"][day_offset]["hour"], data["forecast"]["forecastday"][day_offset]["date"]
        
        model = model_med if droplet_type == "medium" else model_ext
        all_data = []
        
        for day_offset in range(3):
            forecast, date_str = fetch_hourly_forecast(latlon, day_offset)
            for hour_data in forecast[hour_start:hour_end + 1]:
                time_str = hour_data["time"]
                hour_label = pd.to_datetime(time_str).strftime("%H:%M")
                weather = {
                    "Temperature": hour_data["temp_c"],
                    "Wind.Speed": hour_data["wind_mph"],
                    "Humidity": hour_data["humidity"],
                    "Dew.Point": hour_data.get("dewpoint_c", 0),
                    "wind direction": hour_data.get("wind_degree", 0)
                }
                
                def get_drift_cutoff_location(weather, model, height):
                    for loc in np.arange(0.0, 40.5, 0.5):
                        input_df = pd.DataFrame([{
                            "Height": height,
                            "Temperature": weather["Temperature"],
                            "Dew.Point": weather["Dew.Point"],
                            "Humidity": weather["Humidity"] / 100.0,
                            "Wind.Speed": weather["Wind.Speed"],
                            "Location": loc,
                            "wind direction": weather["wind direction"]
                        }])
                        drift = model.predict(input_df)[0]
                        drift = max(0, min(drift, 100))
                        if drift < 5.0:
                            return loc
                    return 40.5
                
                cutoff_location = get_drift_cutoff_location(weather, model, drone_height)
                all_data.append({
                    "Date": date_str,
                    "Hour": hour_label,
                    "Wind (mph)": weather["Wind.Speed"],
                    "Humidity (%)": weather["Humidity"],
                    "Temperature (°C)": weather["Temperature"],
                    "Dew Point (°C)": weather["Dew.Point"],
                    "Wind Dir (°)": weather["wind direction"],
                    "Drift <5% After (m)": cutoff_location
                })
        
        df_result = pd.DataFrame(all_data)
        table_html = df_result.to_html(classes="table table-striped table-bordered text-center align-middle", index=False)
        return render_template('detailed_results.html', table_html=table_html)
        
    except Exception as e:
        return render_template('detailed_results.html', error=f'An error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
