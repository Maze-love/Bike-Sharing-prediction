# Data Understanding Report

## Dataset Shape
- **Rows:** 730
- **Columns:** 16

## Columns

- `instant` (int64)
- `dteday` (str)
- `season` (int64)
- `yr` (int64)
- `mnth` (int64)
- `holiday` (int64)
- `weekday` (int64)
- `workingday` (int64)
- `weathersit` (int64)
- `temp` (float64)
- `atemp` (float64)
- `hum` (float64)
- `windspeed` (float64)
- `casual` (int64)
- `registered` (int64)
- `cnt` (int64)

## Missing Values

- **instant:** 0 (0.0%)
- **dteday:** 0 (0.0%)
- **season:** 0 (0.0%)
- **yr:** 0 (0.0%)
- **mnth:** 0 (0.0%)
- **holiday:** 0 (0.0%)
- **weekday:** 0 (0.0%)
- **workingday:** 0 (0.0%)
- **weathersit:** 0 (0.0%)
- **temp:** 0 (0.0%)
- **atemp:** 0 (0.0%)
- **hum:** 0 (0.0%)
- **windspeed:** 0 (0.0%)
- **casual:** 0 (0.0%)
- **registered:** 0 (0.0%)
- **cnt:** 0 (0.0%)

## Duplicated Rows
- **Count:** 0

## Descriptive Statistics

```
           instant      dteday      season          yr        mnth     holiday     weekday  workingday  weathersit        temp       atemp         hum   windspeed       casual   registered          cnt
count   730.000000         730  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000  730.000000   730.000000   730.000000   730.000000
unique         NaN         730         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN          NaN          NaN          NaN
top            NaN  01-01-2018         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN          NaN          NaN          NaN
freq           NaN           1         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN         NaN          NaN          NaN          NaN
mean    365.500000         NaN    2.498630    0.500000    6.526027    0.028767    2.997260    0.683562    1.394521   20.319259   23.726322   62.765175   12.763620   849.249315  3658.757534  4508.006849
std     210.877136         NaN    1.110184    0.500343    3.450215    0.167266    2.006161    0.465405    0.544807    7.506729    8.150308   14.237589    5.195841   686.479875  1559.758728  1936.011647
min       1.000000         NaN    1.000000    0.000000    1.000000    0.000000    0.000000    0.000000    1.000000    2.424346    3.953480    0.000000    1.500244     2.000000    20.000000    22.000000
25%     183.250000         NaN    2.000000    0.000000    4.000000    0.000000    1.000000    0.000000    1.000000   13.811885   16.889713   52.000000    9.041650   316.250000  2502.250000  3169.750000
50%     365.500000         NaN    3.000000    0.500000    7.000000    0.000000    3.000000    1.000000    1.000000   20.465826   24.368225   62.625000   12.125325   717.000000  3664.500000  4548.500000
75%     547.750000         NaN    3.000000    1.000000   10.000000    0.000000    5.000000    1.000000    2.000000   26.880615   30.445775   72.989575   15.625589  1096.500000  4783.250000  5966.000000
max     730.000000         NaN    4.000000    1.000000   12.000000    1.000000    6.000000    1.000000    3.000000   35.328347   42.044800   97.250000   34.000021  3410.000000  6946.000000  8714.000000
```
