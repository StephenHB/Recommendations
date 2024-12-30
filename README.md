# Recommendations
## Experiment with different recommendation systems
1. Two Tower Embedding (tte)
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

- **Example**: In a movie recommendation system, one tower could process user information (e.g., demographics, preferences), while the other processes movie features (e.g., genre, cast). The model learns to match users with movies they are likely to enjoy.
2. BLP