import pandas as pd
from domain.vae_model import CVAE

df = pd.read_csv('./data/processed/input_ml.csv', index_col=[1,2], header=[0,1]).drop(('Unnamed: 0_level_0', 'Unnamed: 0_level_1'), axis=1)

vae = CVAE(input_dim=(160, 2))
vae.load_model_weights(weights_path='./models/cvae.weights.h5')
history = vae.fit(df, weights_path='./models/cvae.weights.h5')
