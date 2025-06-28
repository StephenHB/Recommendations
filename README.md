# Recommendations Project

A comprehensive collection of recommendation system implementations using both classical econometric methods and modern deep learning approaches.

## Overview

This project contains two main recommendation approaches:

1. **BLP (Berry, Levinsohn, and Pakes)**: Classical econometric approach using instrumental variables and two-stage least squares
2. **TTE (Two Tower Embedding)**: Modern deep learning approach using TensorFlow Recommenders

## Project Structure

```
Recommendations/
├── blp/                    # BLP econometric model
│   ├── config/            # Configuration classes
│   ├── model/             # Core estimation logic
│   ├── test/              # Test suite
│   ├── example_usage.py   # Usage examples
│   └── README.md          # BLP documentation
├── tte/                    # Two Tower Embedding model
│   ├── config/            # Configuration classes
│   ├── model/             # Core model components
│   ├── utils/             # Utility functions
│   ├── execute.py         # Main execution script
│   └── README.md          # TTE documentation
├── .gitignore             # Git ignore rules
├── LICENSE                # Project license
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python >= 3.9, < 3.12
- TensorFlow >= 2.15.0
- NumPy >= 1.21.0
- Pandas >= 1.5.0

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Recommendations
```

2. Install dependencies for both projects:
```bash
# Install BLP dependencies
pip install -r blp/requirements.txt

# Install TTE dependencies
pip install -r tte/requirements.txt
```

## BLP Model

The BLP (Berry, Levinsohn, and Pakes) model is a classical econometric approach for demand estimation that addresses endogeneity using instrumental variables.

### Key Features
- Two-stage least squares (2SLS) estimation
- Instrumental variables support
- Tikhonov regularization
- TensorFlow-based computation

### Quick Usage
```python
from blp.config.base import Base
from blp.model.estimation import TFEstimator
import pandas as pd

# Configure and run
config = Base()
config.target_name = "quantity"
config.exog_ind_names = ["income"]
config.exog_dep_names = ["price"]
config.instrument_variable_names = ["cost", "wage"]

estimator = TFEstimator()
estimator.set_config(config).set_data(data)
results = estimator.fit()
```

### Testing
```bash
cd blp/test
python run_tests.py
```

## TTE Model

The Two Tower Embedding (TTE) model is a modern deep learning approach for recommendation systems using TensorFlow Recommenders.

### Key Features
- Separate user and item embedding towers
- LogQ correlation for bias correction
- Flexible feature processing
- Production-ready with TFRS

### Quick Usage
```python
from tte.config.base import BaseConfig
from tte.model.two_tower_model import TwoTowerModel
import tensorflow_recommenders as tfrs

# Configure and run
config = BaseConfig()
task = tfrs.tasks.Retrieval(remove_accidental_hits=True)
model = TwoTowerModel(config=config, ...)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
model.fit(training_data, epochs=config.num_epochs)
```

### Training
```bash
cd tte
python execute.py
```

## Model Comparison

| Aspect | BLP | TTE |
|--------|-----|-----|
| **Approach** | Classical econometrics | Deep learning |
| **Method** | 2SLS with IV | Two-tower embeddings |
| **Use Case** | Demand estimation | General recommendations |
| **Data Requirements** | Structured, IV needed | Flexible feature types |
| **Scalability** | Medium | High |
| **Interpretability** | High | Medium |
| **Production Ready** | Yes | Yes |

## Development

### Code Quality

Both projects include:
- Type hints throughout
- Comprehensive error handling
- Unit tests
- Documentation
- Linting configuration

### Running Tests

```bash
# BLP tests
cd blp/test
python run_tests.py

# TTE tests (if available)
cd tte
python -m pytest
```

### Code Style

The project follows PEP 8 guidelines with:
- Black code formatting
- Pylint for static analysis
- Type checking with mypy

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

### BLP Model
- Berry, S., Levinsohn, J., & Pakes, A. (1995). Automobile prices in market equilibrium. Econometrica, 63(4), 841-890.
- Nevo, A. (2001). Measuring market power in the ready-to-eat cereal industry. Econometrica, 69(2), 307-342.

### TTE Model
- Covington, P., Adams, J., & Sargin, E. (2016). Deep neural networks for youtube recommendations.
- Yi, X., Yang, J., Hong, L., Cheng, D., Heldt, L., Kumthekar, A., ... & Chi, E. H. (2019). Sampling-bias-corrected neural modeling for large corpus item recommendations.

## Support

For questions and support:
- Check the individual project READMEs in `blp/README.md` and `tte/README.md`
- Review the example usage files
- Run the test suites for verification

# Recommendations
## 1. What is this repository for?
This repository is designed to develop diverse solutions for assortment recommendations, incorporating several widely recognized models from both industry and academia. The objective is to create an automated system that seamlessly handles data preparation, model training and validation, and model selection to deliver optimal recommendations at scale. 

- The implemented models include discrete choice models (e.g., BLP), two-tower embeddings, Factorization Machines, and Collaborative Filtering.
- Validation method could include multi armed bandit (MAB), etc.
   
## 2. Experiment with different recommendation systems

### Two Tower Embedding (tte)
The **Two-Tower Embedding Model** is a neural network architecture commonly used for tasks such as recommendation systems, information retrieval, and matching problems. The model consists of two separate towers (or sub-networks) that learn embeddings for two different types of inputs (e.g., user and item, query and document).

- **Structure**: 
  - The first tower processes one input, typically using an embedding layer to transform categorical features into a continuous vector representation.
  - The second tower processes the other input in the same way.
  - These two towers are trained independently but are typically joined at the end for a similarity measure, such as cosine similarity, dot product, or other distance metrics, to match the inputs.

- **Use Cases**: 
  - **Recommendation Systems**: Matching users to items by learning a shared embedding space.
  - **Search**: Matching queries to documents based on learned representations.
  
- **Training**:
  - The model is typically trained using contrastive loss, where the goal is to bring the representations of matching inputs closer together while pushing non-matching inputs further apart.

- **Advantages**:
  - **Scalability**: The separate towers allow for efficient training on large datasets by enabling the reuse of embeddings.
  - **Flexibility**: Each tower can be optimized independently, allowing for different types of architectures for different inputs.


### Berry–Levinsohn–Pakes (BLP) Model

The **Berry–Levinsohn–Pakes (BLP) model** is a widely used econometric model for estimating demand systems in differentiated product markets, especially in the context of the industrial organization field. It is particularly useful for analyzing the effects of market characteristics, such as product prices and features, on consumer choices and firm behavior.

#### Key Features of the BLP Model:

- **Demand System**: The BLP model specifies a demand system where consumers are assumed to choose among a set of differentiated products (e.g., automobiles, mobile phones). Each product is characterized by observable features (e.g., price, brand, quality) and possibly unobservable characteristics (e.g., consumer preferences, firm-level heterogeneity).

- **Random Coefficients**: One of the key innovations in the BLP model is the use of random coefficients, which allows for heterogeneity in consumer preferences. Consumers are assumed to have different sensitivities to product characteristics, like price, which varies across individuals.

- **Endogeneity of Prices**: The BLP model accounts for the endogeneity of prices — that is, the possibility that prices are correlated with unobserved product characteristics that affect demand. This is addressed by using instruments (variables that are correlated with prices but not with the unobserved factors affecting demand) to identify the parameters of the model.

- **Instrumental Variables**: To deal with the endogeneity problem, the BLP model typically uses instruments for prices. These instruments might include variables such as cost shifters, regional market characteristics, or product characteristics that influence prices but are not directly related to demand.

- **Estimation**: The BLP model is typically estimated using a method called **GMM (Generalized Method of Moments)**, which minimizes the difference between the model's predictions and observed data. This estimation can be computationally intensive, particularly because it requires solving a high-dimensional integral over consumer preferences.

#### Steps in the BLP Model:
1. **Specification of the Utility Function**:
   Consumers choose the product that maximizes their utility, which depends on product characteristics (e.g., price, features) and their preferences.
   The utility of consumer \( i \) for product \( j \) can be written as:
   
   $U_{ij} = \beta_j X_j + \alpha_i Z_j + \epsilon_{ij}$
   
   where:
   - $U_{ij}$ is the utility of consumer $i$ from product $j$,
   - $X_j$ is a vector of observable characteristics for product $j$,
   - $Z_j$ is a vector of characteristics influencing individual preferences,
   - $\beta_j$ and $\alpha_i$ are parameters to be estimated,
   - $\epsilon_{ij}$ represents unobserved factors (such as taste variation).

2. **Market Share Equation**:
   The market share $s_j$ for product $j$ is derived from the probability of consumer $i$ choosing product $j$:
   
   $$s_j = \frac{e^{X_j \beta}}{\sum_{k} e^{X_k \beta}}$$

   where:
   - $X_j$ represents the characteristics of product $j$,
   - $\beta$ is a vector of parameters related to the characteristics.

3. **Identification of Parameters**:
   The model identifies the parameters using **instrumental variables** to handle the endogeneity of prices, often through the assumption that instruments are correlated with prices but not with unobserved demand shocks.

4. **Estimation**:
   The parameters of the demand system are typically estimated using **Generalized Method of Moments (GMM)**:
   
   $$\hat{\theta} = \arg \min_{\theta} \left( g(\theta)' W g(\theta) \right)$$
   
 


