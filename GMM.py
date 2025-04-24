import models
import signal_processing
# =====================
# GMM CLASSIFIER
# =====================

class EMGFeatureGMM:
    def __init__(self, n_components=5):
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

    def fit(self, windows):
        features = np.array([extract_features(w) for w in windows])
        self.X_scaled = self.scaler.fit_transform(features)
        self.gmm.fit(self.X_scaled)

    def predict(self, windows):
        features = np.array([extract_features(w) for w in windows])
        X_scaled = self.scaler.transform(features)
        return self.gmm.predict(X_scaled)

    def predict_proba(self, windows):
        features = np.array([extract_features(w) for w in windows])
        X_scaled = self.scaler.transform(features)
        return self.gmm.predict_proba(X_scaled)

    def save(self, prefix='emg_gmm'):
        dump(self.gmm, f'{prefix}_model.joblib')
        dump(self.scaler, f'{prefix}_scaler.joblib')

    def load(self, prefix='emg_gmm'):
        self.gmm = load(f'{prefix}_model.joblib')
        self.scaler = load(f'{prefix}_scaler.joblib')
