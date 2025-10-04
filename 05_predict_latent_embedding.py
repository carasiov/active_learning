import pandas as pd
from domain.vae_model import CVAE
import sklearn.manifold

df_input = pd.read_csv('./data/processed/input_ml.csv', index_col=[0,1,2], header=[0,1], nrows=20000) #.drop('phap', axis=1)

vae = CVAE(input_dim=(160, 2))
vae.load_model_weights(weights_path='./models/cvae.weights.h5')
df_vae_latent, df_vae_reconstruted = vae.predict(df_input)

# t-SNE embedding of the latent space
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
tsne_output = tsne.fit_transform(df_vae_latent.values)
# Create a new DataFrame for the latent space with a MultiIndex for columns
latent_space_tsne_df = pd.DataFrame(tsne_output, index=df_input.index, columns=pd.MultiIndex.from_product([['latent_space'], ['tsne'], range(tsne_output.shape[1])]))
df_vae_latent.columns = pd.MultiIndex.from_tuples([('latent_space', 'latent', num) for num in df_vae_latent.columns])

# Concatenate the new DataFrame with the original DataFrame
# Add a new level to the columns of df_input
df_input.columns = pd.MultiIndex.from_tuples([(var, 'input', num) for var, num in df_input.columns])
# Add a new level to the columns of df_vae_reconstruted
df_vae_reconstruted.columns = pd.MultiIndex.from_tuples([(var, 'recon', num) for var, num in df_vae_reconstruted.columns])
# Concatenate the new DataFrame with the original DataFrame and the reconstructed DataFrame
df_output = pd.concat([df_input, df_vae_reconstruted, df_vae_latent, latent_space_tsne_df], axis=1)

# Save the updated DataFrame if needed
df_output.to_csv('./data/output/latent_space_tsne_embedding.csv', index=True)