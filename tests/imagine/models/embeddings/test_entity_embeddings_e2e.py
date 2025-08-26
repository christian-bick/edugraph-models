import pytest
from assertpy import assert_that
import os
import numpy as np
from scipy.spatial.distance import cosine

# Adjust the path to import from the source directory
# This assumes the test is run from the project root directory
from imagine.models.embeddings.entity_embeddings_infer import embed_entities

# --- Test Configuration ---

# Prerequisite: The training script must be run first to generate these files in the output directory.
OUT_DIR = "out"
MODEL_PATH_BIASED = os.path.join(OUT_DIR, "embed_entities_biased.onnx")
MODEL_PATH_NEUTRAL = os.path.join(OUT_DIR, "embed_entities_neutral.onnx")
DATA_PATH = os.path.join(OUT_DIR, "embed_entities.pt")

# --- Test Constants ---

# We use cosine distance (1 - similarity), so a smaller value means more similar.
DISTANCE_THRESHOLD_LOW = 0.10  # Corresponds to similarity > 0.90
DISTANCE_THRESHOLD_HIGH = 0.40 # Corresponds to similarity < 0.60

# --- Test Cases ---

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"Skipping E2E tests: Model/data files not found in '{OUT_DIR}'. Run training script first.")
def test_1_subset_superset_proximity():
    """Tests that a subset's embedding is very close to its superset's embedding."""
    print("\nRunning Test 1: Subset/Superset Proximity")
    set_a = ["http://edugraph.io/edu#IntegerAddition", "http://edugraph.io/edu#IntegerSubtraction"]
    set_b = ["http://edugraph.io/edu#IntegerAddition", "http://edugraph.io/edu#IntegerSubtraction", "http://edugraph.io/edu#ProcedureExecution"]

    vector_a = embed_entities(set_a, MODEL_PATH_NEUTRAL, DATA_PATH)
    vector_b = embed_entities(set_b, MODEL_PATH_NEUTRAL, DATA_PATH)

    assert_that(vector_a).is_not_none()
    assert_that(vector_b).is_not_none()

    distance = cosine(vector_a, vector_b)
    print(f"  Cosine distance between subset and superset: {distance:.4f}")
    assert_that(distance).is_less_than(DISTANCE_THRESHOLD_LOW)

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"Skipping E2E tests: Model/data files not found in '{OUT_DIR}'. Run training script first.")
def test_2_disjoint_concept_distance():
    """Tests that embeddings for two unrelated concepts are far apart."""
    print("\nRunning Test 2: Disjoint Concept Distance")
    set_a = ["http://edugraph.io/edu#IntegerAddition", "http://edugraph.io/edu#ProcedureExecution"]
    set_b = ["http://edugraph.io/edu#LinearShapeDrawing", "http://edugraph.io/edu#WritingFluency"]

    vector_math = embed_entities(set_a, MODEL_PATH_NEUTRAL, DATA_PATH)
    vector_language = embed_entities(set_b, MODEL_PATH_NEUTRAL, DATA_PATH)

    assert_that(vector_math).is_not_none()
    assert_that(vector_language).is_not_none()

    distance = cosine(vector_math, vector_language)
    print(f"  Cosine distance between disjoint concepts: {distance:.4f}")
    assert_that(distance).is_greater_than(DISTANCE_THRESHOLD_HIGH)

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"Skipping E2E tests: Model/data files not found in '{OUT_DIR}'. Run training script first.")
def test_3_parent_child_hierarchy():
    """Tests that a child concept is closer to its parent than to an unrelated concept."""
    print("\nRunning Test 3: Parent/Child Hierarchy")
    child = ["http://edugraph.io/edu#IntegerAddition"]
    parent = ["http://edugraph.io/edu#Arithmetic"]
    unrelated = ["http://edugraph.io/edu#LinearShapeDrawing"]

    vector_child = embed_entities(child, MODEL_PATH_NEUTRAL, DATA_PATH)
    vector_parent = embed_entities(parent, MODEL_PATH_NEUTRAL, DATA_PATH)
    vector_unrelated = embed_entities(unrelated, MODEL_PATH_NEUTRAL, DATA_PATH)

    assert_that(vector_child).is_not_none()
    assert_that(vector_parent).is_not_none()
    assert_that(vector_unrelated).is_not_none()

    distance_child_parent = cosine(vector_child, vector_parent)
    distance_child_unrelated = cosine(vector_child, vector_unrelated)

    print(f"  Distance Child-Parent: {distance_child_parent:.4f}")
    print(f"  Distance Child-Unrelated: {distance_child_unrelated:.4f}")
    assert_that(distance_child_parent).is_less_than(distance_child_unrelated)

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"Skipping E2E tests: Model/data files not found in '{OUT_DIR}'. Run training script first.")
def test_4_biased_vs_neutral_divergence():
    """Tests that the biased and neutral models produce different embeddings."""
    print("\nRunning Test 4: Biased vs. Neutral Divergence")
    set_mixed = ["http://edugraph.io/edu#Arithmetic", "http://edugraph.io/edu#ProcedureExecution"]

    vector_biased = embed_entities(set_mixed, MODEL_PATH_BIASED, DATA_PATH)
    vector_neutral = embed_entities(set_mixed, MODEL_PATH_NEUTRAL, DATA_PATH)

    assert_that(vector_biased).is_not_none()
    assert_that(vector_neutral).is_not_none()

    # With the re-introduction of weighted pooling, the biased and neutral
    # models should now produce different embeddings for a mixed-type set.
    are_equal = np.array_equal(vector_biased, vector_neutral)
    print(f"  Are biased and neutral vectors identical? {are_equal}")
    assert_that(are_equal).is_false()

@pytest.mark.skipif(not os.path.exists(DATA_PATH), reason=f"Skipping E2E tests: Model/data files not found in '{OUT_DIR}'. Run training script first.")
def test_5_idempotency():
    """Tests that running inference on the same input produces the same output."""
    print("\nRunning Test 5: Idempotency")
    set_a = ["http://edugraph.io/edu#IntegerAddition", "http://edugraph.io/edu#IntegerSubtraction"]

    vector_a1 = embed_entities(set_a, MODEL_PATH_NEUTRAL, DATA_PATH)
    vector_a2 = embed_entities(set_a, MODEL_PATH_NEUTRAL, DATA_PATH)

    assert_that(vector_a1).is_not_none()
    assert_that(vector_a2).is_not_none()

    # Use assertpy's equality check for lists/arrays
    assert_that(vector_a1.tolist()).is_equal_to(vector_a2.tolist())
    print("  First and second runs produced identical embeddings.")