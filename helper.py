"""Helper functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.gofplots import qqplot
#from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm

def percentile_interval(stat, alpha):
    """Return left and right boundaries for the given significance level.

    Parameters
    ----------
    stat : numpy.array
        Data
    alpha : float
        Significance level

    Returns
    -------
    np.array
        Left and right boundaries
    """
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)]) # Two-sided percentiles
    return boundaries


def describe(series, kind):
    """Briefly describe a variable.

    Parameters
    ----------
    series : pandas.Series
        Variable data
    kind : str
        Numerical ('num') or Categorical ('cat')

    Raises
    ------
    ValueError
        Raise if kind is wrong
    """
    if kind == 'cat':
        print(f'{series.nunique():,.0f} unique')
        nulls = series.isna().sum()
        if nulls:
            print(f'\033[31m{nulls:,.0f} nulls\033[0m')
        else:
            print(f'{nulls:,.0f} nulls')
        print(f"{'':->20}")
        print(series.value_counts().to_string(max_rows=10, name=True))
    elif kind == 'num':
        nulls = series.isna().sum()
        if nulls:
            print(f'\033[31m{nulls:,.0f} nulls\033[0m')
        else:
            print(f'{nulls:,.0f} nulls')
        print(f"Min = {series.min():,.2f}")
        print(f"Max = {series.max():,.2f}")
        print(f"Med = {series.median():,.2f}")
    else:
        raise ValueError("kind must be 'categorical' or 'numerical'")


def plot_hist(array_x, array_y, bins, conf_lvl, xlabel, title):
    """Plot histogram and scatter plot, if `array2` is not None.

    Parameters
    ----------
    array_x : numpy.array
        x-axis values
    array_y : numpy.array
        y-axis values
    bins : int
        Number of bins for histogram
    conf_lvl : float
        Confidence level % for interval
    xlabel : str
        x-axis label for all plots
    title : str
        Common title for all plots
    """
    alpha = 1.0 - conf_lvl / 100.0
    boundaries = percentile_interval(array_x, alpha)
    if array_y is not None:
        plt.figure(figsize=(12, 4))
        first = 1
        num_plots = 3
        plt.subplot(1, num_plots, first)
        plt.scatter(array_x, array_y, color='C1', marker='x', alpha=0.5)
        k, b = np.polyfit(array_x, array_y, deg=1)
        fitted_lr = k * array_x + b
        plt.plot(array_x, fitted_lr, color='C0', ls='--', label='Лин. рег.')
        plt.xlabel(f'{xlabel}', size=10, family='monospace')
        plt.ylabel('Таргет', size=10, family='monospace')
        plt.ylim([array_y.min(), array_y.max()])
        plt.legend()
        plt.grid(lw=0.5)
    else:
        plt.figure(figsize=(8, 4))
        first = 0
        num_plots = 2
    
    plt.subplot(1, num_plots, first + 1)
    plt.hist(array_x, bins=bins, density=True, color='C2', edgecolor='C2', alpha=0.5, histtype='stepfilled')
    plt.vlines(np.median(array_x), 0, plt.gcf().gca().get_ylim()[1], ls='--', color='C0', label=f"Мед. = {np.median(array_x):.1f}")
    plt.xlabel(f'{xlabel}\n(все значения)', size=10, family='monospace')
    #plt.ylabel('Count', size=10, family='monospace')
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(lw=0.5)

    plt.subplot(1, num_plots, first + 2)
    plt.hist(array_x[np.where((array_x >= boundaries[0]) & (array_x <= boundaries[1]))], bins=bins, density=True, color='C2', edgecolor='C2', alpha=0.5, histtype='stepfilled')
    plt.vlines(np.median(array_x), 0, plt.gcf().gca().get_ylim()[1], ls='--', color='C0', label=f"Мед. = {np.median(array_x):.1f}")
    plt.xlabel(f'{xlabel}\n({conf_lvl}%)', size=10, family='monospace')
    plt.suptitle(title, size=10, family='monospace')
    plt.legend()
    plt.gca().set_axisbelow(True)
    plt.grid(lw=0.5)
    plt.tight_layout();


def plot_trace(df, col, xlabel, ylabel, title):
    """Plot a trace plot for the given column.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    col : str
        Column's name
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    title : str
        Plot's title
    """    
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df[col], color='C0', marker='.', lw=0.75)
    plt.hlines(df[col].mean(), xmin=df.index.min(), xmax=df.index.max(), color='C1', lw=1, ls='--', label=f"Ср. знач. = {df[col].mean():.1f}")
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel(ylabel, size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title(title, size=11, family='monospace')
    plt.legend(prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');


def plot_true_pred_trace(x, y_true, y_pred, xlabel, ylabel, title):
    """Plot two trace plots for true target and in-sample predictions.

    Parameters
    ----------
    x : numpy.array
        x-axis data
    y_true : numpy.array
        True target
    y_pred : numpy.array
        Predicted target
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    title : str
        Plot's title
    """    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_true, color='C0', marker='.', lw=0.75, label='Факт')
    plt.plot(x, y_pred, color='C1', marker='.', lw=0.75, label='Предсказания')
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel(ylabel, size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title(title, size=11, family='monospace')
    plt.legend(prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');


def plot_true_pred_scat(y_true, y_pred, xlabel, ylabel):
    """Plot a scatter plot for true target vs predictions.

    Parameters
    ----------
    y_true : numpy.array
        True target
    y_pred : numpy.array
        Predcited target
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    """
    r = np.corrcoef(y_true, y_pred)[0, 1]
    fitted_line = np.poly1d(np.polyfit(y_true, y_pred, deg=1))
    x = np.linspace(y_true.min(), y_true.max(), y_true.shape[0])
    plt.figure(figsize=(4, 4))
    plt.scatter(y_true, y_pred, color='C0', marker='x', alpha=0.5, label=f'$r = {r:.2f}$')
    plt.plot(x, fitted_line(x), color='C1', ls='--', label='Лин. рег.')
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel(ylabel, size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.title('Предсказания vs Факт', size=11, family='monospace')
    plt.legend(loc='lower right', prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');


def plot_pacf(df, col, nlags, alpha, xlabel, title):
    """Plot partial auto-correlation function.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    col : str
        Column's name
    nlags : int
        Number of lags
    alpha : float
        Significance level
    xlabel : str
        x-axis label
    title : str
        Plot's title
    """
    pacf_corr, pacf_ci = pacf(df[col], nlags=nlags, alpha=alpha)
    print(f'PACF(1) = {pacf_corr[1]:.2f}')

    plt.figure(figsize=(7, 4))
    plt.bar(x=[0], height=pacf_corr[0], width=0.35, align='center', color='white', edgecolor='C0', alpha=0.75, label='PACF(0)')
    plt.bar(x=range(1, nlags + 1), height=pacf_corr[1:], width=0.35, align='center', color='C0', edgecolor='C0', alpha=0.75, label=f'PACF(1-{nlags})')
    plt.fill_between(range(0, nlags + 1), pacf_corr - pacf_ci[:, 0], pacf_corr - pacf_ci[:, 1], color='C1', lw=0, alpha=0.25, label='95% CI')
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel('PACF', size=10, family='monospace')
    plt.xticks(list(range(0, nlags + 1)), size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title(title, size=11, family='monospace')
    plt.legend(loc='upper right', prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');


def plot_bar(df, x, y, xlabel, title, kind='vertical', min_val=None):
    """Plot a bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    x : str
        x-axis column's name
    y : str
        y_axis column's name
    xlabel : str
        x-axis label
    title : str
        Plot's title
    kind : str, optional
        Bar plot orientation: 'vertical' or 'horizontal', by default 'vertical'

    Raises
    ------
    ValueError
        Raise if kind is wrong
    """
    if kind == 'vertical':
        plt.figure(figsize=(6, 4))
        bars = plt.bar(range(df[x].shape[0]), df[y].values, width=0.25, align='center', alpha=0.5)
        plt.xlabel(xlabel, size=10, family='monospace')
        plt.xticks(range(df[x].shape[0]), df[x].values)
        plt.ylim([min_val, plt.gca().get_ylim()[1]])
    elif kind == 'horizontal':
        plt.figure(figsize=(6, 4))
        bars = plt.barh(range(df[x].shape[0]), df[y].values, height=0.25, align='center', alpha=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel(xlabel, size=10, family='monospace')
        plt.yticks(range(df[x].shape[0]), df[x].values)
        plt.xlim([min_val, plt.gca().get_xlim()[1]])
    else:
        raise ValueError("kind must be 'vertical' or 'horizontal'")
    for i, _ in enumerate(bars):
        bars[i].set_color(f'C{i}')
        bars[i].set_edgecolor(f'C{i}')
    plt.gca().set_axisbelow(True)
    plt.grid(lw=0.5)
    plt.title(title, size=10, family='monospace');


def plot_decomposition(df, col, period, xlabel, ylabel, title):
    """Decompose time series into trend and seasonal components, plot and return results.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    col : str
        Column's name to decompose
    period : int
        Seasonal period
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    title : str
        Plot's title

    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult
        Trend-seasonal decomposition results
    """
    decompose_result = seasonal_decompose(df[col], period=period)

    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df[col], color='C0', marker='.', lw=0.75, label='Данные')
    plt.plot(decompose_result.trend, color='C1', marker='.', lw=0.75, label='Тренд')
    plt.plot(decompose_result.seasonal, color='C2', lw=0.75, label='Сезонная')
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel(ylabel, size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title(title, size=11, family='monospace')
    plt.legend(prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');

    return decompose_result
    

def plot_goodness(y_true, y_pred, xlabel1, xlabel2, ylabel2):
    """Plot 3 plots to assess goodness-of-fit of a model:
        residuals distribution, q-q plot, residuals vs predictions.

    Parameters
    ----------
    y_true : numpy.array
        True target
    y_pred : numpy.array
        Predicted target
    xlabel1 : str
        x-axis label for residuals distribution plot
    xlabel2 : str
        x-axis label for residuals vs predictions plot
    ylabel2 : str
        y-axis label for residuals vs predictions plot
    """
    residuals = y_true - y_pred
    x = np.linspace(residuals.min(), residuals.max())
    residuals_theor = norm(residuals.mean(), residuals.std(ddof=1))
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(residuals, bins=30, density=True, histtype='stepfilled', color='C0', edgecolor='C0', alpha=0.5, label='Факт')
    plt.plot(x, residuals_theor.pdf(x), color='C1', label='Норм')
    plt.xlabel(xlabel1, size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title('Распределение остатков', size=11, family='monospace')
    plt.legend(loc='upper left', prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement')
    plt.gca().set_axisbelow(True)
    plt.subplot(1, 3, 2)
    qqplot(residuals, dist=residuals_theor, line='q', ax=plt.gca())
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title('Q-Q plot', size=11, family='monospace')
    plt.grid(lw=0.25, color='xkcd:cement')
    plt.gca().set_axisbelow(True)
    plt.subplot(1, 3, 3)
    r = np.corrcoef(residuals, y_pred)[0, 1]
    fitted_line = np.poly1d(np.polyfit(y_pred, residuals, deg=1))
    x = np.linspace(y_pred.min(), y_pred.max(), y_pred.shape[0])
    plt.scatter(y_pred, residuals, color='C0', marker='x', alpha=0.5, label=f'$r = {r:.2f}$')
    plt.plot(x, fitted_line(x), color='C1', ls='--', label='Лин. рег.')
    plt.xlabel(xlabel2, size=10, family='monospace')
    plt.ylabel(ylabel2, size=10, family='monospace')
    plt.xticks(size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title('Остатки vs Предсказания', size=11, family='monospace')
    plt.legend(loc='lower left', prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement')
    plt.gca().set_axisbelow(True)
    plt.tight_layout();


def plot_forecast(full_df, forecast_df, forecast_mean, forecast_ci, col, conf_lvl, xlabel, ylabel, title):
    """Plot actual and forecasted target.

    Parameters
    ----------
    full_df : pandas.DataFrame
        All data
    forecast_df : pandas.DataFrame
        Actual data which was forcasted
    forecast_mean : pandas.Series
        Mean forecasted target
    forecast_ci : pandas.DataFrame
        CI for forecasted target
    col : str
        Target column's name
    conf_lvl : float
        Confidence level % for interval
    xlabel : str
        x-axis label
    ylabel : str
        y-axis label
    title : str
        Plot's title
    """
    plt.figure(figsize=(8, 4))
    plt.plot(full_df[col], color='C0', marker='.', lw=0.75, label='Все')
    plt.plot(forecast_df[col], color='C1', marker='.', lw=1.0, label='Факт')
    plt.plot(forecast_mean, marker='.', color='C2', lw=1.0, label='Прогноз')
    plt.fill_between(forecast_df.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='C2', lw=0, alpha=0.15, label=f'{conf_lvl:.0%} CI')
    plt.xlabel(xlabel, size=10, family='monospace')
    plt.ylabel(ylabel, size=10, family='monospace')
    plt.xticks(full_df.index, full_df.index.month, size=10, family='monospace')
    plt.yticks(size=10, family='monospace')
    plt.title(title, size=11, family='monospace')
    plt.legend(loc='lower left', prop={'size': 9, 'family': 'monospace'})
    plt.grid(lw=0.25, color='xkcd:cement');


def run_adf_test(df, col, alpha):
    """Run Augmented Dickey-Fuller test for stationarity and print results.

    Parameters
    ----------
    df : pandas.DataFrame
        Data
    col : str
        Column's name
    """
    adf_stats, adf_pval, _, _, _, _ = adfuller(df[col])
    print(f'{adf_stats = :>6.2f}, {adf_pval = :.2f}', end=', ')
    print(f'Стационарен для {alpha = }') if adf_pval < alpha else print(f'Нестационарен для {alpha = }')


# def predict_partial(y, X, feature_name, order, seasonal_order, kind, grid_size, ice_perc, feat_perc):
#     '''
#     Calculate partial predictions for one feature with name `feature_name` in Pandas Dataframe `X`.
#     Estimator can by any object that has `predict` method: model, pipeline.
#     `grid_size` controls the number of values of a feature to generate predictions for.
#     Return Pandas DataFrame with:
#     - feature values used for predictions, 
#     - mean predictions -- this is partial dependence (PD), 
#     - Individual Conditional Expectation (ICE) low and high percentiles, as per `ice_perc`
#     '''
#     if kind not in ('num', 'cat', 'ord'):
#         raise ValueError("kind must be 'num', 'cat' or 'ord'")
#     grid_size = min(X[feature_name].nunique(), grid_size)
#     feature_values = []
#     feature_pred_means = []
#     ice_los = []
#     ice_his = []
#     if kind == 'num':
#         # For numerical features we can use either all values in its space or bounded by some percentiles
#         feature_space = np.linspace(*percentile_interval(X[feature_name], alpha=1.0-feat_perc/100), grid_size)
#     elif kind == 'cat':
#         feature_space = X[feature_name].unique()[::X[feature_name].nunique() // grid_size]
#     else: # kind == 'ord'
#         feature_space = X[feature_name].sort_values().unique()[::X[feature_name].nunique() // grid_size]
#     for feature_value in feature_space:
#         feature_values.append(feature_value)
#         feature_new = pd.Series([feature_value] * X.shape[0], name=feature_name, index=X.index) # Fix one feature value for all observations
#         # First drop the whole feature, then concatenate its new values, i.e. replace the feature vector
#         # Predict with this replaced feature vector
#         exog = pd.concat([X.drop(feature_name, axis=1), feature_new], axis=1)
#         model_sarimax = SARIMAX(endog=y, 
#                             exog=exog, 
#                             order=order,
#                             seasonal_order=seasonal_order)
#         fitted_sarimax = model_sarimax.fit()
#         feature_pred = fitted_sarimax.fittedvalues
#         feature_pred_mean = feature_pred.mean() # Partial Dependence
        
#         feature_pred_means.append(feature_pred_mean)
#         ice_lo, ice_hi = percentile_interval(feature_pred, alpha=1.0-ice_perc/100) # ICE boundaries
#         ice_los.append(ice_lo)
#         ice_his.append(ice_hi)
#     return pd.DataFrame({'Value': feature_values, 'Pred_mean': feature_pred_means, 'ICE_lo': ice_los, 'ICE_hi': ice_his})


# def plot_partial(y, X, feature_name, order, seasonal_order, kind='num', grid_size=10, ice_perc=95, feat_perc=100):
#     plt.figure(figsize=(5, 5))
#     pdp_df = predict_partial(y, X, feature_name, order, seasonal_order, kind, grid_size, ice_perc, feat_perc)
#     plt.plot(pdp_df['Value'], pdp_df['Pred_mean'], color='C0')
#     # Show Individual Conditional Expectation (ICE) boundaries as area, as per `ice_perc`
#     plt.plot(pdp_df['Value'], pdp_df['ICE_hi'], color='C1', lw=0.5, alpha=0.5)
#     plt.plot(pdp_df['Value'], pdp_df['ICE_lo'], color='C1', lw=0.5, alpha=0.5)
#     plt.fill_between(pdp_df['Value'], pdp_df['ICE_hi'], pdp_df['ICE_lo'], color='lightgrey', alpha=0.25, label='95% CI for ICE')
#     plt.xlabel(f'{feature_name}', size=10, family='monospace')
#     plt.ylabel('Пред. цена', size=10, family='monospace')
#     plt.xticks(size=10, family='monospace')
#     plt.yticks(size=10, family='monospace')
#     plt.legend(prop={'size': 9, 'family': 'monospace'});
#     plt.grid(lw=0.25, color='xkcd:cement')
#     plt.gca().set_axisbelow(True)
#     plt.suptitle('PDP', size=10, family='monospace')
#     plt.tight_layout();
#     return pdp_df