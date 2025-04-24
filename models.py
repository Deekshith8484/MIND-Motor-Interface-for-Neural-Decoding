import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.filters.hp_filter import hpfilter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from statsmodels.tsa.ar_model import AutoReg