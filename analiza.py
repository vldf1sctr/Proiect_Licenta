import requests
import urllib3
import warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

resp = requests.get(
    'https://proiect_licenta.apimella.com/get_all',
    headers={'x_api_key': API_KEY},
    verify=False
)
df = pd.DataFrame(resp.json())
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data').reset_index(drop=True)

df['dummy'] = df['dummy'].fillna(0).astype(int)
df['ircc']  = df['ircc'].ffill()
df['robor'] = df['robor'].ffill()

exog = df[['ircc', 'robor', 'dummy']].values

os.makedirs('output', exist_ok=True)
log_file = open('output/rezultate.txt', 'w')

def log(text):
    print(text)
    log_file.write(str(text) + '\n')

def test_adf(serie, nume):
    log(f"\n{'='*50}")
    log(f"Test ADF — {nume}")
    log(f"{'='*50}")
    rezultat = adfuller(serie, autolag='AIC')
    log(f"  Statistica ADF : {rezultat[0]:.4f}")
    log(f"  p-value        : {rezultat[1]:.4f}")
    log(f"  Concluzie      : {'STATIONARA' if rezultat[1] < 0.05 else 'NESTATIONARA (necesita diferentiere)'}")
    for cheie, val in rezultat[4].items():
        log(f"  Valoare critica {cheie}: {val:.4f}")

def grafic_acf_pacf(serie, nume):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(serie,  ax=axes[0], lags=min(10, len(serie)//2 - 1), title=f'ACF — {nume}')
    plot_pacf(serie, ax=axes[1], lags=min(10, len(serie)//2 - 1), title=f'PACF — {nume}')
    plt.tight_layout()
    plt.savefig(f'output/acf_pacf_{nume}.png', dpi=150)
    plt.close()
    log(f"Grafic salvat: output/acf_pacf_{nume}.png")

def metrici_evaluare(serie, nume, exog, n_test=3):
    log(f"\n{'='*50}")
    log(f"Metrici evaluare — {nume} (test pe ultimele {n_test} observatii)")
    log(f"{'='*50}")

    train_serie = serie[:-n_test]
    test_serie  = serie[-n_test:]
    train_exog  = exog[:-n_test]
    test_exog   = exog[-n_test:]

    model_eval = auto_arima(
        train_serie,
        X=train_exog,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        with_intercept=False
    )

    pred = model_eval.predict(n_periods=n_test, X=test_exog)

    mae  = np.mean(np.abs(pred - test_serie))
    rmse = np.sqrt(np.mean((pred - test_serie) ** 2))
    mape = np.mean(np.abs((pred - test_serie) / test_serie)) * 100

    log(f"  MAE  : {mae:.4f}")
    log(f"  RMSE : {rmse:.4f}")
    log(f"  MAPE : {mape:.2f}%")

def analizeaza(serie, nume, exog, steps=5):
    log(f"\n{'='*50}")
    log(f"Analiza: {nume}")
    log(f"{'='*50}")

    model = auto_arima(
        serie,
        X=exog,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        with_intercept=False
    )

    log(model.summary())

    X_future = np.tile(exog[-1], (steps, 1))
    forecast, conf_int = model.predict(n_periods=steps, X=X_future, return_conf_int=True)

    future_dates = pd.bdate_range(
        start=df['data'].iloc[-1] + pd.Timedelta(days=1),
        periods=steps
    )

    rezultate = pd.DataFrame({
        'Data':   future_dates.strftime('%d.%m.%Y'),
        nume:     forecast.round(4),
        'CI inf': conf_int[:, 0].round(4),
        'CI sup': conf_int[:, 1].round(4),
    })

    log(f"\nPrognoze {nume} — {steps} zile lucratoare:")
    log(rezultate.to_string(index=False))

    plt.figure(figsize=(10, 4))
    plt.plot(df['data'], serie, label='Valori reale', color='steelblue')
    plt.plot(future_dates, forecast, label='Prognoză', color='tomato', linestyle='--', marker='o')
    plt.fill_between(future_dates, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color='tomato', label='IC 95%')
    plt.axvline(df['data'].iloc[-1], color='gray', linestyle=':', label='Azi')
    plt.title(f'Prognoză {nume}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'output/prognoze_{nume}.png', dpi=150)
    plt.close()
    log(f"Grafic salvat: output/prognoze_{nume}.png")

for serie, nume in [(df['eur_ron'], 'EUR_RON'), (df['usd_ron'], 'USD_RON')]:
    test_adf(serie.values, nume)
    grafic_acf_pacf(serie.values, nume)
    metrici_evaluare(serie.values, nume, exog)
    analizeaza(serie.values, nume, exog)

log_file.close()
print("\nFisier salvat: output/rezultate.txt")
