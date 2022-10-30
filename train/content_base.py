import tensorflow as tf


class content_base():
    def __init__(self, user_behavior, anime_features, user_id):
        self.user_behavior = self.convert_tf_tensor(user_behavior)
        self.anime_features = self.convert_tf_tensor(anime_features)
        self.curr_user = user_id
        self.anime_ids = None

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
        return anime_feats #tf.transpose(anime_feats)

    def gen_user_vector(self):
        user_anime = self.gen_user_anime()
        anime_feats = self.gen_anime_feats
        user_vector = tf.matmul(anime_feats, user_anime)
        return user_vector / tf.reduce_sum(user_vector, axis=1, keepdims=True)
    #TODO a function to generate movies rating

    #TODO a function to get the top predicted rating

    def convert_tf_tensor(self, data):
        if not tf.is_tensor(data):
            return tf.constant(data)
        else:
            return data

    def change_user(self, user_id):
        self.curr_user = user_id


user = [[1, 3, 5, 6, 4, 7], [2, 3, 5, 6, 7, 8],
        [1, 1, 3, 6, 7, 8], [1, 6, 10, 6, 7, 8]]
movie = [[3, 6, 0, 7, 8], [10, 6, 3, 7, 8], [1, 6, 3, 7, 8], [6, 6, 7, 7, 8]]
#[[3, 6, [0,1], 7, 8], [10, 6, [2,3], 7, 8], [1, 6, [3,6], 7, 8], [6, 6, [7,4], 7, 8]]
con = content_base(user, movie, 1)
print(f"Our value: {con.gen_anime_feats()}")
