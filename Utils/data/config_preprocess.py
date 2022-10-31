FIRST_PREPROCESS = True

# Will one hot encocde Genres after splitting it and Type as well,
# Label encode producers, bucktize Aired per year, Label encode studio,
# one hot encode Source, Convert Duration to minutes, One Hot encode Rating.
PREPROCESS_MAP = {"anime":
                            {"Drop":
                                    ["Name","English name", "Japanese name","Premiered", "Licensors"],
                            "Process": {"Genres": "label_onehot_encode",
                                        "Type": "onehot_encoder",
                                        "Producers": "label_encoder",
                                        "Aired": "extract_years_num",
                                        "Studios": "label_encoder",
                                        "Source": "onehot_encoder",
                                        "Duration": "conv_dur_min",
                                        "Rating": "onehot_encoder"}}
                                           }



