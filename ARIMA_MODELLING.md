# Performing ARIMA Analysis

### Background

Auto Regressive Integrated Moving Average (ARIMA) is a **forecasting model** for time-series data.
The ARIMA model models three aspects of a time series:
* Seasonality (How cyclical the pattern is)
* Trend (Where the plot is moving towards over time)
* Noise (How much random variability there is)


These aspects of the model is represented by three parameters.
1. *p* is the number of autoregressive terms (for AR)
2. *d* is the number of nonseasonal differences (for I)
3. *q* is the number of moving-average terms (for MA)

### Steps

#### Finding the lag term (d)

We need to turn our data into a straight line that oscillates around y=0.
This is done by taking the **difference** between one time period and the previous time period. To deal with linear trends, we will take the first-order difference, and second-order difference for quadratic trends.

1. Create a new column called "first-diff"
``` df["first_diff"] = df["Open"].diff() ```

2. Because the first row will contain an NA, we use `.dropna()`.
```df = df.dropna()```

3. To test for stationary nature, we use the ** Augmented Dickey Fuller Test **.
```
from statsmodels.tsa.stattools import adfuller
adfuller(df['first_diff'])
```

The second value of the result gives us the p-value. The null hypothesis is that the plot is not stationary. The alternative hypothesis is that the plot is stationary. 

#### Finding the autoregressive terms (p)

To do this, we use the Partial Auto Correlation Function (PACF).
PACF expresses the correlation between observations made at two points in time, while accounting for any influence from other data points.

For instance, given a regression where we predict **y** from y(t-1), y(t-2) and y(t-3), the PACF value is the correlation of y(t) and y(t-3) that is **not predicted/explained** by y(t-1) and y(t-2).

```
plot_pacf(df['first_diff'], method="ywm")
plt.figure(figsize=(20,10))
plt.show()
```

#### Finding the moving average terms (q)

To do this, we use the Auto Correlation Function (ACF)
The ACF tells how many MA terms are required to remove any autocorrelation in the stationarized series.

For the ACF plot, the y-axis is the correlation coefficient, and the x-axis is the number of lags.

#### Combining Everything

With our p, d and q terms, we may now apply ARIMA Modelling.

*Note:* We perform the test using the "Open" or original data, not the "first_diff". This is because d=1 already performs the differencing.

1. Import the ARIMA model 
2. Declare the model with the order parameter (containing the p,d and q parameters).
3. Call `.fit()` on the model
4. Forecast for the length of the test data.
5. Plot the forecast.

```
from statsmodels.tsa.arima.model import ARIMA
mod = ARIMA(train['Open'], order=(2,1,2))
res = mod.fit()
forecast = res.forecast(steps = len(test))
forecast = pd.DataFrame(forecast)
forecast_array = forecast["predicted_mean"]
forecast_series = pd.Series(forecast_array, index = test.index)

# Plotting
plt.figure(figsize = (16,9))
plt.plot(test['Open'])
plt.plot(forecast_series)
plt.legend(['Actual', 'Forecast'])
```