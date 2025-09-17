# EduGraph Models

This project provides tools for generating models related to classification and embedding generation with
the [EduGraph competency ontology](https://github.com/christian-bick/edugraph-ontology). 

In combination, these two types of models allow for high accuracy similarity search of completely unlabeled 
learning material across all media formats supported by a multimodal LLM like documents, images, audio and video.

To generate classification models it currently relies on fine-tuning foundational multimodal models using
[generated & labeled learning material](https://github.com/christian-bick/imagine-content) for supervised learning.

To generate embedding models, it solely relies on the structure of the ontology itself using state-of-the-art 
knowledge graph embedding strategies that map the knowledge graph structure into high dimensional vector spaces.

## Ontology

This project is centered around the EduGraph ontolog which is automatically retrieved from the 
[ontology repository](https://github.com/christian-bick/edugraph-ontology) during model generation.

## Features

*   **Classification Model Data Generation:** Generates training data for fine-tuning Gemini to classify learning materials according to a defined ontology.
*   **Embeddings Model Training:** Trains an RGCN (Relational Graph Convolutional Network) model to create embeddings for the ontology entities.

## Getting Started

### Prerequisites

*   Python >= 3.13
*   [UV](https://astral.sh/blog/uv) for build & dependency management
*   Updated CUDA drivers

### Installation

1. Install the dependencies using uv:
    ```bash
    uv sync
    ```

3.  Set up the environment variables by copying the `.env.example` file to `.env` and filling in the required values:
    ```bash
    cp .env.example .env
    ```

## Usage

### Classification Model

To generate the training data for the classification model, follow these steps:

1.  **Generate the ontology prompt:** This script creates a prompt file that includes the ontology's taxonomy.
    ```bash
    uv run src.edugraph.models.classification.generate_ontology_prompt
    ```
    This will generate the `out/entity_classification_instruction.txt` file.

2.  **Generate the training data:** This script fetches metadata from a Google Cloud Storage (GCS) bucket, merges it, and then generates a `JSONL` file for training.
    ```bash
    uv run src.edugraph.models.classification.generate_training_data <your-gcs-bucket-name>
    ```
    Replace `<your-gcs-bucket-name>` with the name of your GCS bucket. The training data will be saved in `out/training_data.jsonl`.

### Embeddings Model

To train the embeddings model, run the following command:
```bash
uv run src.edugraph.models.embeddings.entity_embeddings_train
```
This script will:
1.  Download the ontology from the specified URL.
2.  Build a PyTorch Geometric graph from the ontology.
3.  Train an RGCN model.
4.  Export the trained model to ONNX format (`out/embed_entities_biased.onnx` and `out/embed_entities_neutral.onnx`) and save the inference data (`out/embed_entities_text.pt`).

## Contributing

Contributions are welcome! Particularly for adding support to more foundational multimodal classification models.

Ideally always open an issue first to make sure your contribution aligns with the project's scope.

Please make sure to add tests with your contribution and to only submit PRs with green tests.

```bash
uv run pytest
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
