import tensorflow as tf

"""Using Tensorflow for faster computation, this boosts preformance a lot"""


class content_base():
    def __init__(self, user_behavior, anime_features, user_id, num_recom=None):
        self.user_behavior = self.convert_tf_tensor(user_behavior)
        # We just need Genres in content based recommendation system
        # Wrapping it in a list because tensorflow refueses to convert it to tensor without it.
        self.anime_feats = tf.squeeze(self.convert_tf_tensor(
            [anime_features["Genres"].to_numpy()]))
        self.raw_anime_ids = tf.squeeze(
            self.convert_tf_tensor(anime_features["MAL_ID"].to_numpy()))
        self.anime_ids = None
        self.curr_user = user_id
        self.num_recom = num_recom
        self.user_rating = None
        self.users_unseen_rating = None
        self.user_anime_feats = None

    def gen_user_anime(self):
        """returns a matrix for each row represent the rating of the movie id to that specific user"""
        print("Started: ", "gen_user_anime")
        user_rows_indx = tf.squeeze(tf.where(tf.equal(
            self.user_behavior[:, 0], self.curr_user)))  # user_id is the zero index of our data
        user_matrix = tf.gather(self.user_behavior, user_rows_indx, axis=0)
        anime_id_rating = tf.gather(user_matrix, [1, 2], axis=1)
        # Will use it in gen_user_anime_feats function, we don't want to do all of these computations again
        self.anime_ids = tf.squeeze(tf.gather(anime_id_rating, [0], axis=1))
        print("Finished: ", "gen_user_anime")
        return tf.transpose(anime_id_rating)

    def gen_user_anime_feats(self):
        print("Started: ", "gen_user_anime_feats")

        #TODO optimize this code
        if not self.anime_ids:
            self.gen_user_anime()
        # Getting the index of rows that has our anime_id
        if not self.user_anime_feats:
            # TODO optimize this part, causing the program to be very slow
            anime_row_index = [True if indx.numpy(
            ) in self.anime_ids else False for indx in self.raw_anime_ids]
            print("Finished something fuck")
            anime_row_index = tf.squeeze(tf.where(anime_row_index))
            # Getting the rows of our anime
            self.user_anime_feats = tf.transpose(tf.squeeze(
                tf.gather(self.anime_feats, anime_row_index, axis=0)))
        print("Finished: ", "gen_user_anime_feats")
        return self.user_anime_feats  # tf.transpose(anime_feats)

    # We could cache the calcuations but I don't trust my ram to keep all these informations
    def gen_user_vector(self):
        print("Started: ", "gen_user_vector")

        user_anime = self.user_anime_feats
        user_vector = tf.matmul(self.anime_feats, user_anime)
        print("Finished: ", "gen_user_vector")
        return tf.transpose(user_vector / tf.reduce_sum(user_vector, axis=1, keepdims=True))

    def gen_user_rating(self):
        print("Started: ", "gen_user_rating")
        if not self.user_rating:
            self.user_rating = tf.cast(tf.transpose(tf.matmul(
                self.gen_user_vector(), self.anime_feats)),tf.float32)
        print("Finished: ", "gen_user_rating")
        return self.user_rating  # I would love to cache the most important info

    def gen_user_unseen_rating(self):
        user_anime = self.gen_user_anime_feats()
        users_unseen_anime = tf.equal(user_anime, tf.zeros_like(user_anime))
        ignore_matrix = tf.zeros_like(tf.cast(user_anime, tf.float32))
        
        if not self.user_rating:
            self.gen_user_rating()
        print(f"users_unseen_anime: {users_unseen_anime.shape}")
        print(f"user_rating: {self.user_rating.shape}")
        print(f"Ignore matrix: {ignore_matrix.shape}")

        self.users_unseen_rating = tf.where(
            users_unseen_anime,
            self.user_rating, ignore_matrix)
        print("Finished: ", "gen_user_unseen_rating")
        return self.users_unseen_rating

    def get_top_rating(self, num_recommendations=None):
        print("Started: ", "get_top_rating")
        if not num_recommendations:
            num_recom = self.num_recom
        else:
            num_recom = num_recommendations

        if not self.users_unseen_rating:
            self.gen_user_unseen_rating()
        print("Finished: ", "get_top_rating")
        return tf.nn.top_k(self.users_unseen_rating, num_recom)[1]

    def convert_tf_tensor(self, data, dtype=None):
        if not tf.is_tensor(data):
            if dtype:
                return tf.constant(data, dtype=dtype)
            else:
                return tf.constant(data)
        else:
            return data

    def change_user(self, user_id):
        self.curr_user = user_id
        self.user_rating = None
        self.users_unseen_rating = None
