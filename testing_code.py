
import pandas as pd
import os
import config
from Utils.data.preprocess import preprocessor_anime_data,preprocess_colabritive
from train.colabritive_system import NNCollaborativeFiltering

USER_ID = config.USER_ID  # 75 Lines

"""------This module for testing development only------This module for testing development only------"""

path_anime = os.path.join("Data", "Raw_Data", "anime.csv")
path_anime_list = os.path.join("Data", "Raw_Data", "animelist.csv")

# I want to test on user_id 0
r_anime = pd.read_csv(path_anime,low_memory=True) # THat's the maximum for that

anime_data = preprocessor_anime_data(r_anime).get_transformed_data()

my_class = preprocess_colabritive(path_anime_list,load_rows=88)

x_user,x_item,y = my_class.get_x_y_data_NNCF(my_class.get_users_for_item(),anime_data)
n_users,n_items = my_class.get_num_user_items()

model = NNCollaborativeFiltering(n_users=n_users, n_items=n_items)
history,model = model.train_model(x_user,x_item,y,epochs=15,embedding_dims=10, d_layers=[10])







