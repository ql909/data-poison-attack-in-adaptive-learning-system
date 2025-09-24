import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # AFM is a logistic regression with special features
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Determine column name mappings (called "defaults" in BKT) so that you don’t have to edit function arguments later
def detect_and_map_columns(dataset_path):

    # "HamptonAlg" dataset
    if "HamptonAlg" in dataset_path:
        dataset_columns = {
            "student": "student",  # student ID
            "skill": "knowledge",  # “knowledge component” (name or ID of the skill)
            "correct": "assessment",  # binary label of correctness (1 = correct, 0 = incorrect)
            # "order": "actionid"  # order of attempts (to compute opportunity counts)
        }

    # "Assistment" dataset
    elif "Assistment" in dataset_path:
        dataset_columns = {
            "student": "studentId",  # student ID
            "skill": "knowledge",  # “knowledge component” (name or ID of the skill)
            "correct": "correct",   # binary label of correctness (1 = correct, 0 = incorrect)
            # "order": None  # will need to compute, "action_num" is perhaps not suitable?
        }

    else:
        raise ValueError("Unknown dataset format. Add mapping for your file in detect_and_map_columns().")

    assert list(dataset_columns.keys()) == ["student", "skill", "correct"]
    return dataset_columns


def load_data(dataset_path, dataset_columns, verbose=True):
    df = pd.read_csv(dataset_path)

    if verbose:
        print("\n========================================")
        print(f"Using the file '{dataset_path}', which has {len(df):,} rows and {len(df.columns):,} columns")
        print(f"Unique students: {df[dataset_columns["student"]].nunique():,}")
        print(f"Unique skills: {df[dataset_columns["skill"]].nunique():,}")

    # Ensure correct types
    df[dataset_columns["student"]] = df[dataset_columns["student"]].astype(str)  # represent as string for one-hot encoding
    df[dataset_columns["skill"]] = df[dataset_columns["skill"]].astype(str)  # represent as string for one-hot encoding

    # Handle missing correctness values (e.g., hint requests)
    n_before = len(df)
    df = df.dropna(subset=[dataset_columns["correct"]])
    n_after = len(df)
    dropped = n_before - n_after
    if dropped > 0:
        print(f"Dropped {dropped:,} rows with missing correctness (likely hint requests).")
    # Convert correctness column to integer (safe now, since no NaN remain)
    df[dataset_columns["correct"]] = df[dataset_columns["correct"]].astype(int)

    # Compute opportunity counts (per student-skill)
    # Counts how many times a student has attempted a particular skill before the current row.
    # This is critical for AFM because it models learning over repeated opportunities.
    # df = df.sort_values([dataset_columns["student"], dataset_columns["order"]])
    df["opportunity"] = df.groupby([dataset_columns["student"], dataset_columns["skill"]]).cumcount()

    return df


def build_features_for_train_data(df, dataset_columns, scale_features=True):
    # Encode students and skills (each student ID into a binary vector, and each skill ID into another binary vector)
    encoder_students = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    encoder_skills = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    students_encoded = encoder_students.fit_transform(df[[dataset_columns["student"]]])
    skills_encoded = encoder_skills.fit_transform(df[[dataset_columns["skill"]]])

    # Model the effect of practice on each skill
    opportunity_values = df[["opportunity"]].to_numpy(dtype=np.float64)  # Numeric opportunity counts
    skill_x_opportunity = skills_encoded.multiply(opportunity_values)  # Interaction term: skill indicator × opportunity count

    # Final feature matrix with 3 components:
    # [student | which skill is being practiced | how many times this skill was practiced by this student before]
    X_train = np.hstack([students_encoded.toarray(), skills_encoded.toarray(), skill_x_opportunity.toarray()])

    # Scale features (helps convergence and performance)
    scaler = None
    if scale_features:
        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X_train)

    # The encoders and scaler are needed later for test data, so that we use the same mappings
    encoders = [encoder_students, encoder_skills]
    return X_train, encoders, scaler


def build_features_for_test_data(df, dataset_columns, encoders, scaler):
    # Same logic as training, but uses .transform() instead of .fit_transform()
    # so it doesn’t learn new encodings for unseen data.
    students_encoded = encoders[0].transform(df[[dataset_columns["student"]]])
    skills_encoded = encoders[1].transform(df[[dataset_columns["skill"]]])

    opportunity_values = df[["opportunity"]].to_numpy(dtype=np.float64)
    skill_x_opportunity = skills_encoded.multiply(opportunity_values)

    # Final feature matrix with 3 components (same as before):
    X_test = np.hstack([students_encoded.toarray(), skills_encoded.toarray(), skill_x_opportunity.toarray()])
    if scaler:
        X_test = scaler.transform(X_test)
    return X_test


def evaluate_model_performance(X, y, model, training_dataset, file_output_flag="w"):
    model_predictions = model.predict(X)
    accuracy = accuracy_score(y, model_predictions)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f}")

    assert file_output_flag == "w" or file_output_flag == "a"
    results_file = 'results/' + training_dataset + '_results.json'
    with open(results_file, mode=file_output_flag) as f:
        json.dump({"accuracy": accuracy, "AUC": auc}, f, indent=4)


def main(all_training_datasets, test_dataset):
    # Initialize settings and the same test set for all the evaluations
    dataset_columns = detect_and_map_columns(test_dataset)
    test_df = load_data(test_dataset + '.csv', dataset_columns)

    # Train various datasets and evaluate them all on the same test set
    for training_dataset_name in all_training_datasets:
        train_df = load_data(training_dataset_name + '.csv', dataset_columns)
        dataset_columns = detect_and_map_columns(training_dataset_name)

        # Extract features (build training design matrix)
        X_train, encoders, scaler = build_features_for_train_data(train_df, dataset_columns)
        y_train = train_df[dataset_columns["correct"]].to_numpy()

        # Train logistic regression (AFM)
        model = LogisticRegression(
            penalty="l2",
            C=1.0,  # Inverse regularization strength (higher means less regularization)
            solver="saga",  # Better for sparse data, can try with lbfgs or other solvers
            max_iter=100  # Max iterations (convergence limit for the logistic regression solver)
        )
        model.fit(X_train, y_train)

        # Training set evaluation
        print("Training set performance:")
        evaluate_model_performance(X_train, y_train, model, training_dataset_name, "w")

        # Test set evaluation
        X_test = build_features_for_test_data(test_df, dataset_columns, encoders, scaler)
        y_test = test_df[dataset_columns["correct"]].to_numpy()

        print("Test set performance:")
        evaluate_model_performance(X_test, y_test, model, training_dataset_name, "a")


if __name__ == "__main__":
    ALL_HAMPTON_DATASETS = [
        "HamptonAlg_train",
        "HamptonAlg_poisoned_5", "HamptonAlg_poisoned_25", "HamptonAlg_poisoned_50",
        "HamptonAlg_sequential_pattern_5", "HamptonAlg_sequential_pattern_25", "HamptonAlg_sequential_pattern_50"
    ]
    #main(ALL_HAMPTON_DATASETS, "HamptonAlg_test")

    ALL_ASSISTMENT_DATASETS = [
        #"Assistment_challenge_train",
        #"Assistment_poisoned_5", "Assistment_poisoned_25", "Assistment_poisoned_50",
        #"Assistment_sequential_pattern_5(1)", Assistment_sequential_pattern_25(1)", "Assistment_sequential_pattern_50(1)",
        "Assistment_hint_abuse_5", "Assistment_hint_abuse_25", "Assistment_hint_abuse_50"
    ]
    main(ALL_ASSISTMENT_DATASETS, "Assistment_challenge_test")
