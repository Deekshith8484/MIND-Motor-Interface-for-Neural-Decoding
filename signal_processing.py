# =====================
# SIGNAL PROCESSING
# =====================

def butter_bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def sliding_window(data, window_size, step_size):
    return np.array([
        data[i:i + window_size]
        for i in range(0, len(data) - window_size + 1, step_size)
    ])

# =====================
# FEATURE EXTRACTION
# =====================

def extract_features(window):
    features = []

    # 1. AR model (lags = 4)
    try:
        ar_model = AutoReg(window, lags=4, old_names=False).fit()
        features.extend(ar_model.params[1:])
    except:
        features.extend([0]*4)

    # 2. Hurst exponent
    try:
        lags = range(2, 20)
        tau = [np.std(np.subtract(window[lag:], window[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        features.append(poly[0])
    except:
        features.append(0)

    # 3. ADF test
    try:
        adf_stat, adf_p, *_ = adfuller(window)
        features.extend([adf_stat, adf_p])
    except:
        features.extend([0, 1])

    # 4. HP filter std
    try:
        cycle, trend = hpfilter(window, lamb=1600)
        features.extend([np.std(trend), np.std(cycle)])
    except:
        features.extend([0, 0])

    # 5. Signal features
    rms = np.sqrt(np.mean(window**2))
    mav = np.mean(np.abs(window))
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)
    features.extend([rms, mav, wl, zc, ssc])

    return features