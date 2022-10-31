from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pickle
from Utils.data import config_preprocess

FIRST_PREPROCESS = config_preprocess.FIRST_PREPROCESS
PREPROCESS_MAP = config_preprocess.PREPROCESS_MAP


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
        print(f"loaded: {name}")
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
