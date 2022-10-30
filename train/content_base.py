import tensorflow as tf


class content_base():
    def __init__(self, user_behavior, anime_features, user_id, num_recom=None):
        self.user_behavior = self.convert_tf_tensor(user_behavior)
        self.anime_features = self.convert_tf_tensor(anime_features)
        self.curr_user = user_id
        self.num_recom = num_recom
        self.anime_ids = None
        self.user_rating = None

    def gen_user_anime(self):
        """returns a matrix for each row represent the rating of the movie id to that specific user"""
        user_rows_indx = tf.squeeze(tf.where(tf.equal(
            self.user_behavior[:, 0], self.curr_user)))  # user_id is the zero index of our data
        user_matrix = tf.gather(self.user_behavior, user_rows_indx, axis=0)
        anime_id_rating = tf.gather(user_matrix, [1, 2], axis=1)
        # Will use it in gen_anime_feats function, we don't want to do all of these computations again
        self.anime_ids = tf.squeeze(tf.gather(anime_id_rating, [0], axis=1))
        return tf.transpose(anime_id_rating)

    def gen_anime_feats(self):
        #TODO optimize this code
        if not self.anime_ids:
            self.gen_user_anime()

        anime_id_col = tf.squeeze(tf.gather(self.anime_features, [0], axis=1))
        # Getting the index of rows that has our anime_id
        anime_row_index = tf.squeeze(tf.where(
            [True if indx in self.anime_ids else False for indx in anime_id_col]))
        # Getting the rows of our anime
        anime_feats = tf.squeeze(
            tf.gather(self.anime_features, anime_row_index, axis=0))
        # Genre is column indx 2
        anime_feats = tf.squeeze(tf.gather(anime_feats, [2], axis=1))
        return anime_feats  # tf.transpose(anime_feats)

    # We could cache the calcuations but I don't trust my ram to keep all these informations
    def gen_user_vector(self):
        user_anime = self.gen_user_anime()
        anime_feats = self.gen_anime_feats
        user_vector = tf.matmul(anime_feats, user_anime)
        return user_vector / tf.reduce_sum(user_vector, axis=1, keepdims=True)

    def gen_user_rating(self):
        if not self.user_rating:
            self.user_rating = tf.matmul(
                self.gen_user_vector(), tf.transpose(self.gen_anime_feats()))
        return self.user_rating  # I would love to cache the most important info

    def gen_user_unseen_rating(self):
        user_anime = self.gen_user_anime()
        users_unseen_anime = tf.equal(user_anime, tf.zeros_like(user_anime))
        ignore_matrix = tf.zeros_like(tf.cast(user_anime, tf.float32))
        self.users_unseen_rating = tf.where(
            users_unseen_anime,
            self.user_rating, ignore_matrix)
        return self.users_unseen_rating

    def get_top_rating(self, num_recommendations=None):
        if not num_recommendations:
            num_recom = self.num_recom
        else:
            num_recom = num_recommendations

        if not self.users_unseen_rating:
            self.gen_user_unseen_rating()

        return tf.nn.top_k(self.users_unseen_rating, num_recom)[1]

    def convert_tf_tensor(self, data):
        if not tf.is_tensor(data):
            return tf.constant(data)
        else:
            return data

    def change_user(self, user_id):
        self.curr_user = user_id
        self.user_rating = None
        self.users_unseen_rating = None
