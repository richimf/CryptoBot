from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib.pyplot as plt

from alpha_vantage.timeseries import TimeSeries
from pprint import pprint
import keras

ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, meta_data = ts.get_intraday(symbol='ETH',interval='1min', outputsize='full')

atributos = data.iloc[:0]
print(atributos)

data['1. open'].plot()
plt.tight_layout()
plt.title('Intraday value for bitcoin (BTC)')
plt.grid()
plt.show()