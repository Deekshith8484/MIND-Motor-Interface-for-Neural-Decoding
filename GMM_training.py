import models
import GMM
import signal_processing
# Step 1: Load or simulate EMG signal
emg_signal = np.random.randn(10000)  # replace with real EMG data
fs = 1000

# Step 2: Preprocess
filtered = butter_bandpass_filter(emg_signal, fs=fs)
windows = sliding_window(filtered, window_size=250, step_size=125)

# Step 3: Train GMM
model = EMGFeatureGMM(n_components=5)
model.fit(windows)
model.save('my_emg_gmm')

# Step 4: Load and predict on new data
model.load('my_emg_gmm')
predictions = model.predict(windows)