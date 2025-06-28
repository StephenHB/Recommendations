from typing import Dict, List, Any


class BaseConfig:
    """Configuration class for Two Tower Embedding (TTE) model"""
    
    def __init__(self):
        # Model parameters
        self.embedding_dimension = 16
        self.max_history_length = 100
        self.product_id_col = 'movie_id'
        self.batch_size = 512
        self.validation_freq = 5
        self.num_epochs = 300
        self.layer_output = 16  # Added missing parameter
        
        # Dense layer configuration
        self.layers = [64, 32]  # Dense layer sizes
        self.cross_layer = False  # Whether to use cross layer
        self.projection_dim = 64  # Projection dimension for cross layer
        self.norm_layer = True  # Whether to use L2 normalization
        self.dropout_layer = True  # Whether to use dropout
        self.dropout_rate = 0.1  # Dropout rate
        
        # Feature configuration
        self.str_vectorizer_config = self.get_str_vectorizer_config()
        self.list_vectorizer_config = self.get_list_vectorizer_config()
        self.str_lookup_config = self.get_str_lookup_config()
        self.int_lookup_config = self.get_int_lookup_config()
        self.normalizer_config = self.get_normalizer_config()
        
        # Feature names
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

    def get_str_vectorizer_config(self) -> Dict[str, List[str]]:
        """Get string vectorizer configuration"""
        return {
            'account': [],
            'product': ['movie_title']
        }

    def get_list_vectorizer_config(self) -> Dict[str, List[str]]:
        """Get list vectorizer configuration"""
        return {
            'account': [],
            'product': []
        }

    def get_str_lookup_config(self) -> Dict[str, List[str]]:
        """Get string lookup configuration"""
        return {
            'account': [],
            'product': []
        }
    
    def get_int_lookup_config(self) -> Dict[str, List[str]]:
        """Get integer lookup configuration"""
        return {
            'account': [],
            'product': []
        }

    def get_normalizer_config(self) -> Dict[str, List[str]]:
        """Get normalizer configuration"""
        return {
            'account': ['user_rating'],
            'product': []
        }
       