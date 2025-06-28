# Two Tower Embedding (TTE) for Assortment Recommendations

This folder contains a cleaned implementation of a Two Tower Embedding model for assortment recommendations using TensorFlow Recommenders. The model learns separate embeddings for users (accounts) and items (products) to enable efficient recommendation generation.

## Overview

The Two Tower Embedding model is a popular approach for recommendation systems that:
- Learns separate embedding spaces for users and items
- Uses contrastive learning to bring similar user-item pairs closer together
- Enables efficient retrieval through approximate nearest neighbor search
- Incorporates logQ correlation correction for unbiased training

## Key Features

- **Two-Tower Architecture**: Separate towers for user and item embeddings
- **Flexible Feature Processing**: Support for various feature types (categorical, numerical, text)
- **LogQ Correlation**: Corrects for sampling bias in recommendation training
- **TensorFlow Recommenders**: Built on TFRS for production-ready recommendations
- **Configurable Architecture**: Easy to customize embedding dimensions, layers, and features

## File Structure

```
tte/
├── config/
│   ├── __init__.py
│   └── base.py              # Configuration class
├── model/
│   ├── __init__.py
│   ├── adapt_layer.py       # Adaptive layer creation
│   ├── data_preprocess.py   # Data preprocessing utilities
│   ├── embedding_model.py   # Core embedding model
│   ├── l2_norm_layer.py     # L2 normalization layer
│   ├── log_q_correlation.py # LogQ correlation implementation
│   ├── single_tower_model.py # Single tower (user/item) model
│   └── two_tower_model.py   # Main two-tower model
├── utils/
│   ├── __init__.py
│   └── utils.py             # Utility functions
├── notebooks/
│   └── tte_v0.py           # Example notebook
├── execute.py              # Main execution script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The main dependencies are:
   - Python >= 3.9, < 3.12
   - TensorFlow >= 2.15.0
   - TensorFlow Recommenders >= 0.7.0
   - TensorFlow Datasets >= 4.9.0
   - NumPy >= 1.21.0
   - Pandas >= 1.5.0

## Usage

### Basic Usage

```python
from tte.config.base import BaseConfig
from tte.model.two_tower_model import TwoTowerModel
import tensorflow_recommenders as tfrs

# 1. Configure the model
config = BaseConfig()
config.embedding_dimension = 32
config.batch_size = 256
config.num_epochs = 100

# 2. Create the model
task = tfrs.tasks.Retrieval(remove_accidental_hits=True)
model = TwoTowerModel(
    config=config,
    vocab_dict=vocab_dict,
    adapted_layers=adapted_layers,
    label_probs=product_prob_lookup,
    lookup=product_lookup,
    task=task
)

# 3. Compile and train
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(training_data, epochs=config.num_epochs)
```

### Running the Full Pipeline

```bash
python execute.py
```

## Model Architecture

### Two-Tower Structure

The model consists of two separate towers:

1. **Account Tower**: Processes user features to generate user embeddings
2. **Product Tower**: Processes item features to generate item embeddings

Each tower can have:
- Embedding layers for categorical features
- Text vectorization for text features
- Normalization layers for numerical features
- Dense layers for feature interaction
- L2 normalization for cosine similarity

### Feature Processing

The model supports multiple feature types:

- **Categorical Features**: String/Integer lookup + embedding
- **Text Features**: Text vectorization + embedding + pooling
- **Numerical Features**: Normalization layers
- **List Features**: Integer lookup + embedding + pooling

### Training Process

1. **Data Preprocessing**: Feature extraction and vocabulary building
2. **Adaptive Layers**: Layer adaptation for feature processing
3. **LogQ Correlation**: Probability lookup table creation
4. **Model Training**: Contrastive learning with sampled negatives
5. **Evaluation**: Retrieval metrics computation

## Configuration

The `BaseConfig` class supports:

### Model Parameters
- `embedding_dimension`: Size of final embeddings
- `batch_size`: Training batch size
- `num_epochs`: Number of training epochs
- `validation_freq`: Validation frequency

### Architecture Parameters
- `layers`: Dense layer sizes `[64, 32]`
- `cross_layer`: Whether to use cross layer
- `norm_layer`: Whether to use L2 normalization
- `dropout_layer`: Whether to use dropout
- `dropout_rate`: Dropout rate

### Feature Configuration
- `str_vectorizer_config`: Text features per tower
- `list_vectorizer_config`: List features per tower
- `str_lookup_config`: String categorical features
- `int_lookup_config`: Integer categorical features
- `normalizer_config`: Numerical features

## Key Components

### TwoTowerModel
Main model class that inherits from `tfrs.Model`:
- Implements contrastive learning
- Handles sampled negatives
- Computes retrieval metrics

### EmbeddingModel
Core embedding generation:
- Creates embedding layers for different feature types
- Handles feature concatenation
- Supports various input formats

### LogQCorrelation
Bias correction for recommendation training:
- Computes item popularity probabilities
- Creates lookup tables for correction
- Handles product ID mapping

### SingleTowerModel
Individual tower implementation:
- Processes features for one tower type
- Supports dense layers and normalization
- Configurable architecture

## Example Output

When training completes successfully:

```
=== Two Tower Embedding (TTE) Model Training ===
1. Initializing configuration...
2. Preparing data preprocessing...
   - Ratings dataset size: 100000
   - Movies dataset size: 1000
3. Creating adapted layers...
4. Preparing training data...
5. Setting up logQ correlation...
6. Creating Two Tower Model...
7. Compiling model...
8. Initializing model inputs...
9. Preparing training batches...
10. Starting model training...
    - Epochs: 300
    - Batch size: 512

11. Model Summary:
Model: "two_tower_model"
...
=== Training Completed Successfully! ===
```

## Notes

- The model uses TensorFlow Recommenders for efficient training
- LogQ correlation helps correct for sampling bias
- The architecture is highly configurable for different use cases
- Supports both user and item cold-start scenarios
- Designed for production deployment with TFRS serving

## References

- Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for youtube recommendations.
- Yi, X., Yang, J., Hong, L., Cheng, D., Heldt, L., Kumthekar, A., ... & Chi, E. H. (2019). Sampling-bias-corrected neural modeling for large corpus item recommendations. 