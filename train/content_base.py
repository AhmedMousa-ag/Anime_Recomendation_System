import tensorflow as tf
class content_base():
    def __init__(self, user_behavior, anime_features, user_id):
        self.user_behavior = self.convert_tf_tensor(user_behavior)
        self.anime_features = self.convert_tf_tensor(anime_features)
        self.curr_user = user_id

#TODO a function to get user rating to each movie "users movies"
    def gen_user_movie(self):
        # returns a matrix for each row represent the rating of the movie id
        # all can be find in animelist file
        #user_id,anime_id,rating,watching_status,watched_episodes
        #0,67,9,1,1
        #0,6702,7,1,4
        user_rows_indx = tf.squeeze(tf.where(tf.equal(self.user_behavior[:,0],self.curr_user))) #user_id is the zero index of our data
        user_matrix = tf.gather(self.user_behavior,user_rows_indx,axis=0)
        # now you need to take all anime id and align them, then put ratings underneath each of them
        anime_id_rating = tf.gather(user_matrix,[1,2],axis=1)

        return tf.transpose(anime_id_rating)
    #TODO a function to get movies feats according to genre

    #TODO a fucntion to generate user vector

    #TODO a function to generate movies rating

    #TODO a function to get the top predicted rating
    def convert_tf_tensor(self,data):
        if not tf.is_tensor(data):
            return tf.constant(data)
        else:
            return data
    def change_user(self, user_id):
        self.curr_user = user_id

user = [[1,3,5,6,4,7],[2,3,5,6,7,8],[1,1,3,6,7,8],[1,6,10,6,7,8]]
movie = [24,6,5,7,8,]
con = content_base(user,movie,1)
print(f"Our value: {con.gen_user_movie()}")


