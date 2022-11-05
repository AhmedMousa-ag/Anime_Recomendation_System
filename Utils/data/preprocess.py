from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pickle
from Utils.data import config_preprocess
import pandas as pd
import tensorflow as tf
import os
from scipy.sparse import csr_matrix

FIRST_PREPROCESS = config_preprocess.FIRST_PREPROCESS
PREPROCESS_MAP = config_preprocess.PREPROCESS_MAP
WRITE_TF_RECORD_PATH = config_preprocess.WRITE_TF_RECORD_PATH


def label_encoder(data, name, first_preprocess=FIRST_PREPROCESS):
    """
    args:
            data: the data we want to encode
            name: the name of our feature, will be used to save the endocer and load it later.
            first_preprocess: if it's first time to process data, it will save the encoder, otherwise it will load it.
    """
    if first_preprocess:
        encoder = LabelEncoder()
        encoder.fit(data)
        pickle.dump(encoder, open(f'{name}.pkl', 'wb'))
    else:
        encoder = pickle.load(open(f'{name}.pkl', 'rb'))
    return encoder.transform(data)


def onehot_encoder(data, name, first_preprocess=FIRST_PREPROCESS):
    """
    args:
            data: the data we want to encode
            name: the name of our feature, will be used to save the endocer and load it later.
            first_preprocess: if it's first time to process data, it will save the encoder, otherwise it will load it.
    """
    data = np.array(data).reshape(-1, 1)
    if first_preprocess:
        encoder = OneHotEncoder(sparse=False)
        encoder.fit(data)
        pickle.dump(encoder, open(f'{name}.pkl', 'wb'))

    else:
        encoder = pickle.load(open(f'{name}.pkl', 'rb'))
    return encoder.transform(data)


def conv_dur_min(data):
    output = []
    for row in data:
        splitted = [int(s) for s in row.split(" ") if s.isdigit()]
        hours = splitted[0] if len(splitted) > 1 else 0
        if len(splitted) == 0:
            output.append(0)
            continue
        minuts = splitted[1] if len(splitted) > 1 else splitted[0]
        dur_in_min = (hours*60)+minuts
        output.append(dur_in_min)
    return output


def extract_years_num(data):
    """I will extract the production year"""
    years = []
    for row in data:
        numbers = [int(s) for s in row.split() if s.isdigit()]
        if len(numbers) > 0:
            years.append(numbers[0])  # year is the indx no 1
        else:
            years.append(0)  # will assign 0 to unkown aired date
    return years


class label_onehot_encode():
    def __init__(self, input, name, first_preprocess=FIRST_PREPROCESS):
        """This class made because I belive we should get more informations about Genres, splite them and combine them,
         and not to lose some informations about it
         args:
            input: pandas column you want to transform
            first_preprocess: If we are using this class for the first time, otherwise it will import the dic
            name: name of that column because we will save the dictionary in order to use it later in productions
            old_dic: the previously written dictionary
        """
        self.data = input
        self.data_dec = {}
        self.num_uniq = 0
        self.data = self.split_words_indx()
        if first_preprocess:
            self.name = name
            self.data_dec = {}
            self.fit_data()
            pickle.dump(self.data_dec, open(f'{name}.pkl', 'wb'))
        else:
            self.data_dec = pickle.load(open(f'{name}.pkl', 'rb'))
            self.num_uniq = len(self.data_dec)

    def split_words(self, input):
        return input.replace(" ", "").split(",")

    def split_words_indx(self):
        input = self.data
        output = []
        if not isinstance(input, np.ndarray):
            input = np.array(input)
        for row in input:
            output.append(self.split_words(row))
        return np.array(output)

    def transform_data(self):
        new_data = []
        for row in self.data:
            arr = np.zeros(self.num_uniq)
            for label in row:
                indx = self.data_dec[label]
                arr[indx] = 1
            new_data.append(arr)
        return new_data

    def fit_data(self):
        for row in self.data:
            for label in row:
                if not label in self.data_dec:
                    self.data_dec[label] = self.num_uniq
                    self.num_uniq += 1


class preprocessor_anime_data():
    def __init__(self, data, pre_process_dict=PREPROCESS_MAP):
        """args:
            data: pandas data
            pre_process_dict: dictionary defines what happens to each column."""
        self.data = data
        self.pre_process_dict = pre_process_dict
        self.process()

    def get_transformed_data(self):
        return self.data

    def excute_function(self, column, function_name, name):
        processed_data = None
        if function_name == "label_onehot_encode":
            processed_data = label_onehot_encode(column, name).transform_data()
        elif function_name == "label_encoder":
            processed_data = label_encoder(column, name)
        elif function_name == "extract_years_num":
            processed_data = extract_years_num(column)
        elif function_name == "onehot_encoder":
            processed_data = onehot_encoder(column, name)
        elif function_name == "conv_dur_min":
            processed_data = conv_dur_min(column)
        return processed_data

    def process(self):
        preprocess_map = self.pre_process_dict
        for i, feature in enumerate(preprocess_map["anime"]["Process"]):
            function = preprocess_map["anime"]["Process"][feature]
            self.data[feature] = self.excute_function(
                self.data[feature], function_name=function, name=feature)
        self.data.drop(preprocess_map["anime"]["Drop"], axis=1, inplace=True)
        return self.data


class preprocess_colabritive():
    def __init__(self, user_data_path: str, load_rows=None, write_path=WRITE_TF_RECORD_PATH):
        """This class prepare data for WALS training once called"""
        self.data_path = user_data_path
        self.load_rows = load_rows
        self.write_path = write_path
        self.write_tf_file()
        self.num_items = 0
        self.num_users = 0

    def __create_mapping(self, values):
        value_to_id = {value: idx for idx, value in enumerate(values.unique())}
        return value_to_id

    def __get_mapped(self):
        df = pd.read_csv(filepath_or_buffer=self.data_path,
                         usecols=["user_id", "anime_id", "rating",
                                  "watching_status", "watched_episodes"],
                         nrows=self.load_rows)
        user_mapping = self.__create_mapping(df["user_id"])
        item_mapping = self.__create_mapping(df["anime_id"])
        df["userId"] = df["user_id"].map(user_mapping.get)
        df["itemId"] = df["anime_id"].map(item_mapping.get)
        mapped_df = df[["userId", "itemId", "rating",
                        "watching_status", "watched_episodes"]]
        return mapped_df

    def __write_tf_grouped_by(self, write_name, group_by: str = "itemId"):
        mapped_df = self.__get_mapped()
        grouped_by_items = mapped_df.groupby(group_by)
        write_path = os.path.join(self.write_path, write_name)
        with tf.io.TFRecordWriter(write_path) as ofp:
            for item, grouped in grouped_by_items:
                example = tf.train.Example(features=tf.train.Features(feature={
                    "key": tf.train.Feature(int64_list=tf.train.Int64List(value=[item])),
                    "indices": tf.train.Feature(int64_list=tf.train.Int64List(value=grouped["userId"].values)),
                    "values": tf.train.Feature(float_list=tf.train.FloatList(value=grouped["rating"].values))
                }))
                ofp.write(example.SerializeToString())

    def write_tf_file(self):
        if not os.path.exists(self.write_path):
            os.makedirs(self.write_path)

        self.__write_tf_grouped_by("users_for_item", group_by="itemId")
        self.__write_tf_grouped_by("items_for_user", group_by="userId")

    def get_users_for_item(self):
        mapped_df = self.__get_mapped()
        grouped_by_items = mapped_df.groupby("itemId", as_index=False)
        return grouped_by_items

    def get_items_for_users(self):
        mapped_df = self.__get_mapped()
        grouped_by_items = mapped_df.groupby("userId", as_index=False)
        return grouped_by_items

    def get_x_y_data_NNCF(self, data_user, data_item):
        """It's never clear what's Y label in NNColabritive Filtering,
        But it's either the user liked it or not (1 or 0).
        But it doesn't tell us how much did the user like it which isn't enough for me,
        I will change it to predict the rating.
        If you would like to follow the paper please replace y line with the following
        y = np.array([ 0 if int(rating[-1]) < 5 else 1 for rating in data["rating"]]) # below 5 didn't like it
        """
        
        y = np.array([int(rating[-1]) for rating in data_user["rating"]])
        
        x_user = np.array([np.array(col) for row, col in
                           data_user["userId", "itemId", "watching_status", "watched_episodes"]]).squeeze()

        # We can take the following features from item feature MAL_ID , Genres , Type , Duration , Source
        x_item = data_item[["MAL_ID", "Type", "Duration", "Source"]]
        # must combine the anime_id with user rating for that anime

        """Data is corrupted, there're rating for animes that doesn't exist in anime data,
        Next lines will handle that."""
        n_x_item = []
        row_indx_del = 0 # Will handle index outstide of the loop because when we delete a row, the tensor gets shorter then out of index error.
        for id in x_user[:, 1]: #Ironicly when we load more than 88 rows of user data, the data behave differently
            val = x_item[x_item["MAL_ID"] == id].to_numpy().squeeze()
            if len(val)<1:
                x_user = np.delete(x_user,row_indx_del,axis=0)
                y = np.delete(y,row_indx_del,axis=0)
                row_indx_del-=1
            else:
                n_x_item.append(val)
            row_indx_del+=1
            
        x_item = np.array(n_x_item)
        self.num_items = len(np.unique(x_user[:,1])) #Required for building the NN network
        self.num_users = len(np.unique(x_user[:,0])) #Required for building the NN network
        return x_user, x_item, y

    def get_num_user_items(self):
        return int(self.num_users), int(self.num_items)
