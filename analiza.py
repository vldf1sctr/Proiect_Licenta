import requests
import urllib3
import warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json

from pmdarima import auto_arima
from arch import arch_model

from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch

from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score


resp = requests.get('https://proiect_licenta_fastapi.apimella.com/get_all', verify=False)
df = pd.DataFrame(resp.json())
# df = pd.read_json("/content/data.json")

df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d')
df.set_index('data', inplace=True)
df = df.asfreq('B')
df.drop(columns=['id'], inplace=True)
df.ffill(inplace=True)
df.dropna(inplace=True)


def seasonalDecompose(series: pd.Series) -> None:
  Path("tsa_res/SeasonalDecompose").mkdir(parents=True, exist_ok=True)

  fig = seasonal_decompose(series).plot()
  fig.tight_layout()
  fig.set_size_inches(8,6)
  fig.savefig(f"tsa_res/SeasonalDecompose/{series.name}.png", dpi=300)
  plt.close()


def unitRootTests(series: pd.Series, regression: str = "c") -> None:
  Path("tsa_res/UnitRootTests").mkdir(parents=True, exist_ok=True)

  adf_result = adfuller(series, regression=regression)
  kpss_result = kpss(series, regression='ct' if regression == 'ct' else 'c')
  zivot_andrews_result = zivot_andrews(series, regression=regression)

  with open(f"tsa_res/UnitRootTests/{series.name}.txt", "w") as f:
    f.write(f"Tip regresie: {regression}\n\n") 
    f.write(f"ADF Test:\n")
    f.write(f"Test Statistic: {adf_result[0]}\n")
    f.write(f"p-value: {adf_result[1]}\n")
    f.write(f"Valori critice:\n")
    for key, value in adf_result[4].items():
        f.write(f"  {key}: {value}\n")

    f.write(f"\nKPSS Test (regresie: {'ct' if regression == 'ct' else 'c'}):\n")
    f.write(f"Test Statistic: {kpss_result[0]}\n")
    f.write(f"p-value: {kpss_result[1]}\n")
    f.write(f"Valori critice:\n")
    for key, value in kpss_result[3].items():
        f.write(f"  {key}: {value}\n")

    f.write(f"\nZivot-Andrews Test:\n")
    f.write(f"Test Statistic: {zivot_andrews_result[0]}\n")
    f.write(f"p-value: {zivot_andrews_result[1]}\n")
    f.write(f"Valori critice:\n")
    for key, value in zivot_andrews_result[2].items():
        f.write(f"  {key}: {value}\n")


def AcfPacf(series: pd.Series, prefix: str = "") -> None:
  Path("tsa_res/ACF&PACF").mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(1, 2, figsize=(10, 4))
  plot_acf(series, ax=ax[0], title=f"{prefix}{series.name}: Autocorrelation")
  plot_pacf(series, ax=ax[1], title=f"{prefix}{series.name}: Partial Autocorrelation")

  plt.tight_layout()
  fig.savefig(f"tsa_res/ACF&PACF/{prefix}{series.name}.png", dpi=300)
  plt.close()


def logReturns(dataframe1: pd.DataFrame, dataframe2: pd.DataFrame, diff_times: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  Path("tsa_res/LogReturns").mkdir(parents=True, exist_ok=True)

  log_series = np.log(dataframe1)
  log_series_values = np.diff(log_series, n=diff_times, axis=0)
  log_diff = pd.DataFrame(log_series_values, index=log_series.index[diff_times:], columns=log_series.columns)
  exog = dataframe2.copy()
  exog["ircc"] = exog["ircc"].diff()
  exog["robor"] = exog["robor"].diff()
  exog = exog.dropna()
  exog_aligned = exog.loc[log_diff.index]

  fig, ax = plt.subplots(1, 2, figsize=(16, 6))
  sns.lineplot(data=log_diff, x=log_diff.index, y=log_diff.columns[0], color="blue", ax=ax[0])
  ax[0].set_title(f"Diferențe logaritmice pentru {log_diff.columns[0]}")

  sns.lineplot(data=log_diff, x=log_diff.index, y=log_diff.columns[1], color="orange", ax=ax[1])
  ax[1].set_title(f"Diferențe logaritmice pentru {log_diff.columns[1]}")

  plt.tight_layout()
  plt.savefig(f"tsa_res/LogReturns/LogReturns.png", dpi=300)
  plt.close()

  return log_diff, exog, exog_aligned
  

def analyzeModel(series: pd.Series, exog: pd.DataFrame = None):
   path = Path("tsa_res/ModelsAnalysis")
   path.mkdir(parents=True, exist_ok=True)
   modelAnalysis = auto_arima(y=series, X=exog, trace=True, seasonal=False)

   if exog is not None:
      with open(path / f"exog_{series.name}_model_summary.txt", "w") as f:
        f.write(modelAnalysis.summary().as_text())
   else:
      with open(path / f"{series.name}_model_summary.txt", "w") as f:
        f.write(modelAnalysis.summary().as_text())

   return modelAnalysis


def evalModel(series: pd.Series, exog: pd.DataFrame = None):
   path = Path("tsa_res/ModelsForecast")
   path.mkdir(parents=True, exist_ok=True)

   split = int(0.8 * len(series))
   y_train = series.values[:split]
   y_test = series.values[split:]

   if exog is not None:
      exog_train = exog.values[:split]
      exog_test = exog.values[split:]

      modelEval = auto_arima(y=y_train, X=exog_train, trace=True, seasonal=False)
      forecast = modelEval.predict(n_periods=len(y_test), X=exog_test)
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.lineplot(x=series.index[split:], y=y_test, color="#2b9721", label="Valori reale")
      sns.lineplot(x=series.index[split:], y=forecast, color="#aa1a1a", label="Prognoze")
      plt.title(f"Valori reale vs. prognoze pentru {series.name} cu variabile exogene")
      plt.legend()
      fig.savefig(path / f"exog_{series.name}_forecast.png", dpi=300)
      plt.close()
      with open(path / f"exog_{series.name}_forecast.txt", "w") as f:
         f.write(f"Prognoze pentru {series.name} cu variabile exogene: {forecast}\n")
         f.write(f"MAE: {mean_absolute_error(y_test, forecast)}\n")
         f.write(f"MSE: {mean_squared_error(y_test, forecast)}\n")
         f.write(f"RMSE: {root_mean_squared_error(y_test, forecast)}\n")
         f.write(f"MAPE: {mean_absolute_percentage_error(y_test, forecast)}\n")
         f.write(f"R^2: {r2_score(y_test, forecast)}\n")
   else:
      modelEval = auto_arima(y=y_train, trace=True, seasonal=False)
      forecast = modelEval.predict(n_periods=len(y_test))
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.lineplot(x=series.index[split:], y=y_test, color="#2b9721", label="Valori reale")
      sns.lineplot(x=series.index[split:], y=forecast, color="#aa1a1a", label="Prognoze")
      plt.title(f"Valori reale vs. prognoze pentru {series.name}")
      plt.legend()
      fig.savefig(path / f"{series.name}_forecast.png", dpi=300)
      plt.close()
      with open(path / f"{series.name}_forecast.txt", "w") as f:
         f.write(f"Prognoze pentru {series.name}: {forecast}\n")
         f.write(f"MAE: {mean_absolute_error(y_test, forecast)}\n")
         f.write(f"MSE: {mean_squared_error(y_test, forecast)}\n")
         f.write(f"RMSE: {root_mean_squared_error(y_test, forecast)}\n")
         f.write(f"MAPE: {mean_absolute_percentage_error(y_test, forecast)}\n")
         f.write(f"R^2: {r2_score(y_test, forecast)}\n")

   return modelEval, forecast, y_test
   
    
def plotResid(model, series: pd.Series, is_exog: bool = False):
   path = Path("tsa_res/Residuals")
   path.mkdir(parents=True, exist_ok=True)

   residuals = model.resid()
   resid_index = series.index[len(series) - len(residuals):]

   plt.figure(figsize=(10, 6))
   sns.lineplot(x=resid_index, y=residuals, color="#5c47e1")
   plt.title(f"Reziduurile modelului {series.name} {'cu variabile exogene' if is_exog else ''}")
   plt.savefig(path / f"{'exog_' if is_exog else ''}{series.name}_residuals.png", dpi=300)
   plt.close()

   residuals_sq = residuals ** 2

   fig, ax = plt.subplots(1, 2, figsize=(10, 4))
   plot_acf(residuals_sq, ax=ax[0], title=f"{series.name} {'cu variabile exogene' if is_exog else ''}: ACF Reziduuri^2")
   plot_pacf(residuals_sq, ax=ax[1], title=f"{series.name} {'cu variabile exogene' if is_exog else ''}: PACF Reziduuri^2")
   plt.tight_layout()
   fig.savefig(path / f"{'exog_' if is_exog else ''}acf_pacf_{series.name}_residuals_squared.png", dpi=300)
   plt.close()



def arch_garchTest(model, series: pd.Series, is_exog: bool = False, p: int = 1, q: int = 1):
   path = Path("tsa_res/ARCH&GARCH")
   path.mkdir(parents=True, exist_ok=True)
    
   residuals = model.resid()
   lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(residuals)
   prefix = "exog_" if is_exog else ""
    
   garch_fit = None
   with open(path / f"{prefix}{series.name}_arch_test.txt", "w") as f:
      f.write(f"ARCH Test pentru {series.name} {"cu variabile exogene" if is_exog else ""}:\n")
      f.write(f"LM Statistic: {lm_stat}\n")
      f.write(f"LM p-value: {lm_pvalue}\n")
      f.write(f"F Statistic: {f_stat}\n")
      f.write(f"F p-value: {f_pvalue}\n")
      f.write(f"\nConcluzie: {'Efecte ARCH prezente => GARCH necesar' if lm_pvalue < 0.05 else 'Nu exista efecte ARCH => GARCH nu este necesar'}\n")

      if lm_pvalue < 0.05:
          garch = arch_model(residuals, vol='Garch', p=p, q=q, rescale=True)
          garch_fit = garch.fit(disp='off')
          f.write(f"\nGARCH({p},{q}) Model:\n")
          f.write(str(garch_fit.summary()))

   return garch_fit


def garchForecast(garch_fit, forecast: np.ndarray, y_test: np.ndarray, series: pd.Series, is_exog: bool = False):
   path = Path("tsa_res/GARCHForecast")
   path.mkdir(parents=True, exist_ok=True)

   prefix = "exog_" if is_exog else ""
   model_name = "ARIMAX+GARCH" if is_exog else "ARIMA+GARCH"
   test_index = series.index[len(series) - len(y_test):]
   
   scale = getattr(garch_fit, "scale", 1.0)
   variance = garch_fit.forecast(horizon=len(forecast)).variance.values[-1]
   volatility = np.sqrt(variance) / scale

   fig, ax = plt.subplots(figsize=(12, 6))
   sns.lineplot(x=test_index, y=y_test, color="#2b9721", label="Valori reale")
   sns.lineplot(x=test_index, y=forecast, color="#aa1a1a", label="Prognoze")
   ax.fill_between(test_index,
                   forecast - 1.96 * volatility,
                   forecast + 1.96 * volatility,
                   alpha=0.3, color="#aa1a1a", label="Interval de încredere 95%")
   plt.title(f"Prognoze {model_name} pentru {series.name}")
   plt.legend()
   plt.tight_layout()
   fig.savefig(path / f"{prefix}{series.name}_garch_forecast.png", dpi=300)
   plt.close()

   with open(path / f"{prefix}{series.name}_garch_forecast.txt", "w") as f:
      f.write(f"Prognoze {model_name} pentru {series.name}:\n")
      f.write(f"MAE: {mean_absolute_error(y_test, forecast)}\n")
      f.write(f"MSE: {mean_squared_error(y_test, forecast)}\n")
      f.write(f"RMSE: {root_mean_squared_error(y_test, forecast)}\n")
      f.write(f"MAPE: {mean_absolute_percentage_error(y_test, forecast)}\n")
      f.write(f"R^2: {r2_score(y_test, forecast)}\n")
   


   


seasonalDecompose(df["eur_ron"])
seasonalDecompose(df["usd_ron"])
seasonalDecompose(df["ircc"])
seasonalDecompose(df["robor"])

unitRootTests(df["eur_ron"], regression="ct")
unitRootTests(df["usd_ron"])
unitRootTests(df["ircc"])
unitRootTests(df["robor"])

AcfPacf(df["eur_ron"])
AcfPacf(df["usd_ron"])

log_diff, exog, exog_aligned = logReturns(dataframe1=df[["eur_ron", "usd_ron"]], dataframe2=df[["ircc", "robor", "dummy"]])

AcfPacf(log_diff["eur_ron"], prefix="log_diff_")
AcfPacf(log_diff["usd_ron"], prefix="log_diff_")

analyzeModel0 = analyzeModel(log_diff["eur_ron"], exog=exog_aligned)
analyzeModel1 = analyzeModel(log_diff["usd_ron"], exog=exog_aligned)
analyzeModel2 = analyzeModel(log_diff["eur_ron"])
analyzeModel3 = analyzeModel(log_diff["usd_ron"])

evalModel0, forecast0, y_test0 = evalModel(log_diff["eur_ron"], exog=exog_aligned)
evalModel1, forecast1, y_test1 = evalModel(log_diff["usd_ron"], exog=exog_aligned)
evalModel2, forecast2, y_test2 = evalModel(log_diff["eur_ron"])
evalModel3, forecast3, y_test3 = evalModel(log_diff["usd_ron"])

plotResid(evalModel0, log_diff["eur_ron"], is_exog=True)
plotResid(evalModel1, log_diff["usd_ron"], is_exog=True)
plotResid(evalModel2, log_diff["eur_ron"], is_exog=False)
plotResid(evalModel3, log_diff["usd_ron"], is_exog=False)

garch_fit0 = arch_garchTest(evalModel0, log_diff["eur_ron"], is_exog=True)
if garch_fit0 is not None:
    garchForecast(garch_fit0, forecast0, y_test0, log_diff["eur_ron"], is_exog=True)
    
garch_fit1 = arch_garchTest(evalModel1, log_diff["usd_ron"], is_exog=True)
if garch_fit1 is not None:
    garchForecast(garch_fit1, forecast1, y_test1, log_diff["usd_ron"], is_exog=True)

garch_fit2 = arch_garchTest(evalModel2, log_diff["eur_ron"])
if garch_fit2 is not None:
    garchForecast(garch_fit2, forecast2, y_test2, log_diff["eur_ron"])

garch_fit3 = arch_garchTest(evalModel3, log_diff["usd_ron"])
if garch_fit3 is not None:
    garchForecast(garch_fit3, forecast3, y_test3, log_diff["usd_ron"])