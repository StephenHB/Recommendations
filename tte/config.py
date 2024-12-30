class BaseConfig:
    def __init__(self):
        
        self.embedding_dimension = 16
        self.max_history_length = 100
        self.product_id_col = 'movie_id'
        self.batch_size = 512
        self.validation_freq = 5
        self.num_epochs = 300

        self.str_vectorizer_config = self.get_str_vectorizer_config()
        self.list_vectorizer_config = self.get_list_vectorizer_config()
        self.str_lookup_config = self.get_str_lookup_config()
        self.normalizer_config = self.get_normalizer_config()


        def get_str_vectorizer_config(self):
            return {
                'account': [],
                'product': ['movie_title']
            }

        def get_list_vectorizer_config(self):
            return {
                'account': [],
                'product': []
            }

        def get_str_lookup_config(self):
            return {
                'account': [],
                'product': []
            }

        def get_normalizer_config(self):
            return {
                'account': ['user_rating'],
                'product': []
            }
        
        self.feature_names = [
                            "user_occupation_text",
                            "movie_id",
                            "movie_genres",
                            "movie_title",
                            "user_id",
                            "user_occupation_label",
                            "raw_user_age",
                            "user_zip_code",
                            "user_gender",
                        ]
       