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
   - $X_j$is a vector of observable characteristics for product $j$,
   - $Z_j$is a vector of characteristics influencing individual preferences,
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
   
 


