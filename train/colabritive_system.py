import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np



class NNCollaborativeFiltering():
    """It's never clear what's Y label in NNColabritive Filtering,
         But it's either the user liked it or not (1 or 0).
         But it doesn't tell us how much did the user like it which isn't enough for me,
         I will change it to predict the rating, will change the activation fucntion of last layer to relu """
    def __init__(self):
        pass

    def get_compiled_model(n_users, n_items, embedding_dims=10, d_layers=[10]):
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')

        embedding_user_mf = Embedding(
            n_users, embedding_dims, name="Embed_user_mf")(user_input)
        user_latent_mf = Flatten()(embedding_user_mf)

        embedding_user_mp = Embedding(
            n_users, embedding_dims, name="Embed_user_mp")(user_input)
        embedding_user_mp = Flatten()(embedding_user_mp)

        embedding_item_mf = Embedding(
            n_items, embedding_dims, name="Embed_item_mf")(item_input)
        item_latent_mf = Flatten()(embedding_item_mf)

        embedding_item_mp = Embedding(
            n_items, embedding_dims, name="Embed_item_mp")(item_input)
        embedding_item_mp = Flatten()(embedding_item_mp)

        mf_vector = Dot()(user_latent_mf, item_latent_mf)
        mlp_vector = Concatenate(
            axes=-1)([embedding_user_mp, embedding_item_mp])

        for layer in d_layers:
            mlp_vector = Dense(layer, activation='relu')(mlp_vector)

        pred_vector = Concatenate(axes=-1)([mf_vector, mlp_vector])

        output = Dense(1, activation="relu")(pred_vector)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse',metircs=['mae'])
        return model

    def train_model(self, data,n_users, n_items, epochs,embedding_dims=10, d_layers=[10]):
        model = self.get_compiled_model(data,n_users, n_items, embedding_dims=embedding_dims, d_layers=d_layers)
        history = model.fit(data,epochs=epochs)
        return history,model
