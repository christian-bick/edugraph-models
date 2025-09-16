# Imagine Data

This project provides tools for generating models related to learning material classification and embeddings. It leverages an ontology to create training data for fine-tuning Large Language Models (LLMs) and to generate embeddings for classifications based on a knowledge graph.

## Features

*   **Classification Model Data Generation:** Generates training data for fine-tuning LLMs to classify learning materials according to a defined ontology.
*   **Embeddings Model Training:** Trains an RGCN (Relational Graph Convolutional Network) model to create embeddings for the ontology entities.

## Getting Started

### Prerequisites

*   Python >= 3.13
*   [Poetry](https://python-poetry.org/) for dependency management.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/imagine-data.git
    cd imagine-data
    ```

2.  Install the dependencies using Poetry:
    ```bash
    poetry install
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
    python -m src.imagine.models.classification.generate_ontology_prompt
    ```
    This will generate the `out/entity_classification_instruction.txt` file.

2.  **Generate the training data:** This script fetches metadata from a Google Cloud Storage (GCS) bucket, merges it, and then generates a `JSONL` file for training.
    ```bash
    python -m src.imagine.models.classification.generate_training_data <your-gcs-bucket-name>
    ```
    Replace `<your-gcs-bucket-name>` with the name of your GCS bucket. The training data will be saved in `out/training_data.jsonl`.

### Embeddings Model

To train the embeddings model, run the following command:
```bash
python -m src.imagine.models.embeddings.entity_embeddings_train
```
This script will:
1.  Download the ontology from the specified URL.
2.  Build a PyTorch Geometric graph from the ontology.
3.  Train an RGCN model.
4.  Export the trained model to ONNX format (`out/embed_entities_biased.onnx` and `out/embed_entities_neutral.onnx`) and save the inference data (`out/embed_entities_text.pt`).

## Ontology

The project uses the EduGraph ontology. The ontology is loaded from the following URL: `https://github.com/christian-bick/edugraph-ontology/releases/download/v0.4.0/core-ontology.rdf`.
