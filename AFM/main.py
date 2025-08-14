import argparse
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression  # AFM is a logistic regression with special features
from sklearn.metrics import accuracy_score, roc_auc_score

# Determine column name mappings (called "defaults" in BKT) so that you don’t have to edit function arguments later
def detect_and_map_columns(dataset_path):

    # "HamptonAlg" dataset detection
    if "HamptonAlg" in dataset_path:
        dataset_columns = {
            "student": "student",  # student ID
            "skill": "knowledge",  # “knowledge component” (name or ID of the skill)
            "correct": "assessment",  # binary label of correctness (1 = correct, 0 = incorrect)
            # "order": "actionid"  # order of attempts (to compute opportunity counts)
        }

    # "Assistment" dataset detection
    elif "Assistment_challenge" in dataset_path:
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


def load_data(dataset_path, verbose=True):
    df = pd.read_csv(dataset_path)

    if verbose:
        print(f"File size of '{dataset_path}' is: {len(df):,} rows, {len(df.columns):,} columns")
        print(f"Unique students: {df[DATASET_COLUMNS["student"]].nunique():,}")
        print(f"Unique skills: {df[DATASET_COLUMNS["skill"]].nunique():,}\n")

    # Ensure correct types
    df[DATASET_COLUMNS["student"]] = df[DATASET_COLUMNS["student"]].astype(str)  # represent as string for one-hot encoding
    df[DATASET_COLUMNS["skill"]] = df[DATASET_COLUMNS["skill"]].astype(str)  # represent as string for one-hot encoding
    df[DATASET_COLUMNS["correct"]] = df[DATASET_COLUMNS["correct"]].astype(int)  # 1 or 0

    # Compute opportunity counts (per student-skill)
    # Counts how many times a student has attempted a particular skill before the current row.
    # This is critical for AFM because it models learning over repeated opportunities.
    # df = df.sort_values([DATASET_COLUMNS["student"], DATASET_COLUMNS["order"]])
    df["opportunity"] = df.groupby([DATASET_COLUMNS["student"], DATASET_COLUMNS["skill"]]).cumcount()

    return df


def build_features_for_train_data(df, scale_features=True):
    # Encode students and skills (each student ID into a binary vector, and each skill ID into another binary vector)
    encoder_students = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    encoder_skills = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    students_encoded = encoder_students.fit_transform(df[[DATASET_COLUMNS["student"]]])
    skills_encoded = encoder_skills.fit_transform(df[[DATASET_COLUMNS["skill"]]])

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


def build_features_for_test_data(df, encoders, scaler):
    # Same logic as training, but uses .transform() instead of .fit_transform()
    # so it doesn’t learn new encodings for unseen data.
    students_encoded = encoders[0].transform(df[[DATASET_COLUMNS["student"]]])
    skills_encoded = encoders[1].transform(df[[DATASET_COLUMNS["skill"]]])
    opportunity_values = df[["opportunity"]].to_numpy(dtype=np.float64)
    skill_x_opportunity = skills_encoded.multiply(opportunity_values)

    # Final feature matrix with 3 components (same as before):
    X_test = np.hstack([students_encoded.toarray(), skills_encoded.toarray(), skill_x_opportunity.toarray()])
    if scaler:
        X_test = scaler.transform(X_test)
    return X_test


def evaluate_model_performance(X, y, model, results_path=None):
    model_predictions = model.predict(X)
    accuracy = accuracy_score(y, model_predictions)
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"Accuracy: {accuracy:.4f} | AUC: {auc:.4f}\n")
    if results_path:
        with open(results_path, mode="w") as f:
            json.dump({"accuracy": accuracy, "AUC": auc}, f, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and evaluate AFM model.")
    parser.add_argument("--dataset", default="Assistment_challenge",
                        help="Dataset: either 'HamptonAlg' or 'Assistment_challenge'.")
    parser.add_argument("--C", type=float, default=1.0,
                        help="Inverse regularization strength (higher means less regularization).")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Max iterations (convergence limit for the logistic regression solver).")
    arguments = parser.parse_args()
    assert arguments.dataset in ["HamptonAlg", "Assistment_challenge"]
    return arguments


def main(args):
    # Initialize settings and data
    train_df = load_data(args.dataset + '_train.csv')
    test_df = load_data(args.dataset + '_test.csv')

    # Extract features (build training design matrix)
    X_train, encoders, scaler = build_features_for_train_data(train_df)
    y_train = train_df[DATASET_COLUMNS["correct"]].to_numpy()

    # Train logistic regression (AFM)
    model = LogisticRegression(
        penalty="l2",
        C=args.C,
        solver="saga",  # better for sparse data, can try with lbfgs or other solvers
        max_iter=args.max_iter
    )
    model.fit(X_train, y_train)

    # Training set evaluation
    print("Training set performance:")
    evaluate_model_performance(X_train, y_train, model, args.dataset + '_results_train.json')

    # Test set evaluation
    X_test = build_features_for_test_data(test_df, encoders, scaler)
    y_test = test_df[DATASET_COLUMNS["correct"]].to_numpy()

    print("Test set performance:")
    evaluate_model_performance(X_test, y_test, model, args.dataset + '_results_test.json')


if __name__ == "__main__":
    parsed_args = parse_arguments()
    DATASET_COLUMNS = detect_and_map_columns(parsed_args.dataset)  # Global variable
    main(parsed_args)
