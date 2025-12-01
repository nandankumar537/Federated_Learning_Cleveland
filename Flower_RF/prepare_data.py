import numpy as np, pandas as pd
df = pd.read_csv("cleveland.csv")
X = df[["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal"]].values
# pick first 50 as public calibrator
np.save("X_public.npy", X[:50])
