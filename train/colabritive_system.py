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
    def __init__(self,n_users,n_items):
        self.n_users = n_users
        self.n_items = n_items
        

    def get_compiled_model(self,x_user_inp_shape,x_item_inp_shape,embedding_dims, d_layers):
        user_input = Input(shape=x_user_inp_shape, dtype='int32', name='user_input')
        item_input = Input(shape=x_item_inp_shape, dtype='int32', name='item_input')

        embedding_user_mf = Embedding(
            self.n_users, embedding_dims, name="Embed_user_mf")(user_input)
        user_latent_mf = Flatten(name="Flatten_Embed_user_mf")(embedding_user_mf)

        embedding_user_mp = Embedding(
            self.n_users, embedding_dims, name="Embed_user_mp")(user_input)
        embedding_user_mp = Flatten(name="Flatten_Embed_user_mp")(embedding_user_mp)

        embedding_item_mf = Embedding(
            self.n_items, embedding_dims, name="Embed_item_mf")(item_input)
        item_latent_mf = Flatten(name="Flatten_Embed_item_mf")(embedding_item_mf)

        embedding_item_mp = Embedding(
            self.n_items, embedding_dims, name="Embed_item_mp")(item_input)
        embedding_item_mp = Flatten(name="Flatten_Embed_item_mp")(embedding_item_mp)

        mf_vector = Dot(axes=1,name="Dot_layer")([user_latent_mf, item_latent_mf])
        mlp_vector = Concatenate(
            axis=-1,name="mlp_concate_layer")([embedding_user_mp, embedding_item_mp])
        
        for i,layer in enumerate(d_layers):
            mlp_vector = Dense(layer, activation='relu',name=f"Dense_layer_{i}")(mlp_vector)

        pred_vector = Concatenate(axis=-1,name="Pred_vec_conc_layer")([mf_vector, mlp_vector])

        output = Dense(1, activation="relu",name="Output_layer")(pred_vector)

        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer='adam', loss='mse',metrics=['mae'])
        return model

    def train_model(self, x_user,x_item,y, epochs,embedding_dims=10, d_layers=[10]):
        x_user_inp_shape = x_user.shape[1:]
        x_item_inp_shape = x_item.shape[1:]

        model = self.get_compiled_model(x_user_inp_shape=x_user_inp_shape,x_item_inp_shape=x_item_inp_shape,
                                     embedding_dims=embedding_dims, d_layers=d_layers)
        history = model.fit((x_user,x_item),y,epochs=epochs)
        return history,model
