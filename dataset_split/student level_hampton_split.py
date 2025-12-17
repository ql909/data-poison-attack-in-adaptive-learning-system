import pandas as pd
from sklearn.model_selection import train_test_split

# File paths
test_path = '/content/HamptonAlg_test.csv'
train_path = '/content/HamptonAlg_train.csv'
merged_output_path = '/content/HamptonAlg_merged.csv'
train_output_path = '/content/HamptonAlg_train_new.csv'
test_output_path = '/content/HamptonAlg_test_new.csv'

# 1. Load datasets
try:
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit()

# 2. Check column name consistency
print("Test dataset columns:", df_test.columns.tolist())
print("Train dataset columns:", df_train.columns.tolist())

if list(df_test.columns) != list(df_train.columns):
    print("Warning: Column names are not identical. Aligning to common columns.")
    common_columns = df_test.columns.intersection(df_train.columns).tolist()
    print("Common columns:", common_columns)
    df_test = df_test[common_columns]
    df_train = df_train[common_columns]
else:
    print("Column names are identical. Proceeding with merge.")

# 3. Check data volume
print(f"\nTest dataset rows: {len(df_test)}")
print(f"Train dataset rows: {len(df_train)}")

# 4. Check if student and knowledge columns exist
id_column = 'student'
skill_column = 'knowledge'
if id_column not in df_test.columns or id_column not in df_train.columns:
    print(f"Error: Column '{id_column}' not found in one or both datasets.")
    exit()
if skill_column not in df_test.columns or skill_column not in df_train.columns:
    print(f"Error: Column '{skill_column}' not found in one or both datasets.")
    exit()

# 5. Check for overlapping students between train and test datasets
test_students = set(df_test[id_column].unique())
train_students = set(df_train[id_column].unique())
intersection = test_students.intersection(train_students)
if len(intersection) > 0:
    print(f"\nWarning: Found {len(intersection)} overlapping students between test and train datasets.")
    print("Merging may result in duplicate records. Proceeding with merge.")
else:
    print(f"\nNo overlapping students found in original datasets.")

# 6. Merge datasets
merged_df = pd.concat([df_train, df_test], ignore_index=True)
print(f"\nMerged dataset rows: {len(merged_df)}")
print(f"Merged dataset unique students: {merged_df['student'].nunique()}")
print(f"Merged dataset unique knowledge: {merged_df['knowledge'].nunique()}")
print("Merged dataset columns:", merged_df.columns.tolist())

# 7. Check for duplicate rows in merged dataset
duplicates = merged_df.duplicated().sum()
if duplicates > 0:
    print(f"\nWarning: Found {duplicates} duplicate rows in the merged dataset.")
    # Option to remove duplicates
    # merged_df = merged_df.drop_duplicates()
    # print(f"After removing duplicates, merged dataset rows: {len(merged_df)}")
else:
    print("\nNo duplicate rows found in the merged dataset.")

# 8. Handle missing values
merged_df = merged_df.dropna(subset=[id_column, skill_column])
print(f"\nAfter removing rows with missing student or knowledge, rows: {len(merged_df)}")

# 9. Save merged dataset
merged_df.to_csv(merged_output_path, index=False)
print(f"\nMerged dataset saved to: {merged_output_path}")

# 10. Split by student in 2:8 ratio, ensuring training set covers all knowledge
unique_students = merged_df['student'].unique()
all_knowledge = set(merged_df['knowledge'].unique())
train_students = []
test_students = []

# Group by knowledge, find students for each knowledge
student_knowledge = merged_df.groupby('student')['knowledge'].unique().apply(set)
rare_knowledge = set()
for knowledge in all_knowledge:
    students_with_knowledge = student_knowledge[student_knowledge.apply(lambda x: knowledge in x)].index
    if len(students_with_knowledge) <= 5:  # Assume knowledge is rare if it appears in 5 or fewer students
        rare_knowledge.add(knowledge)
        train_students.extend(students_with_knowledge)

# Remove duplicates and convert to set
train_students = list(set(train_students))
remaining_students = [s for s in unique_students if s not in train_students]

# Split remaining students in 2:8 ratio
train_size = int(0.8 * len(unique_students)) - len(train_students)
if train_size > 0 and remaining_students:
    train_remaining, test_remaining = train_test_split(
        remaining_students,
        train_size=train_size,
        random_state=42
    )
    train_students.extend(train_remaining)
    test_students = test_remaining
else:
    test_students = remaining_students

# 11. Ensure training set covers all knowledge
train_df = merged_df[merged_df['student'].isin(train_students)]
train_knowledge = set(train_df['knowledge'].unique())
if train_knowledge != all_knowledge:
    print("\nWarning: Training set does not cover all knowledge. Adjusting...")
    missing_knowledge = all_knowledge - train_knowledge
    for knowledge in missing_knowledge:
        students_with_knowledge = merged_df[merged_df['knowledge'] == knowledge]['student'].unique()
        if students_with_knowledge.size > 0:
            train_students.append(students_with_knowledge[0])
            test_students = [s for s in test_students if s not in students_with_knowledge]
    train_df = merged_df[merged_df['student'].isin(train_students)]
    train_knowledge = set(train_df['knowledge'].unique())

# 12. Create test dataset
test_df = merged_df[merged_df['student'].isin(test_students)]

# 13. Verify no overlapping students
train_students_set = set(train_students)
test_students_set = set(test_students)
intersection = train_students_set.intersection(test_students_set)
if len(intersection) > 0:
    print(f"Error: Found {len(intersection)} overlapping students in new split.")
    exit()
else:
    print("\nNo overlapping students between new train and test datasets.")

# 14. Verify knowledge coverage
test_knowledge = set(test_df['knowledge'].unique())
if not test_knowledge.issubset(train_knowledge):
    print("\nError: Test set contains knowledge not in training set.")
    exit()
else:
    print("\nAll knowledge in test set are present in training set.")

# 15. Check data volume and ratio after splitting
total_students = len(unique_students)
train_student_count = len(train_students)
test_student_count = len(test_students)
train_student_percentage = (train_student_count / total_students) * 100
test_student_percentage = (test_student_count / total_students) * 100

print(f"\nNew train dataset: {len(train_df)} rows, {train_student_count} students ({train_student_percentage:.2f}%)")
print(f"New test dataset: {len(test_df)} rows, {test_student_count} students ({test_student_percentage:.2f}%)")
print(f"Training set unique knowledge: {len(train_knowledge)}")
print(f"Test set unique knowledge: {len(test_knowledge)}")

# 16. Verify row ratio
total_rows = len(merged_df)
train_row_percentage = (len(train_df) / total_rows) * 100
test_row_percentage = (len(test_df) / total_rows) * 100
print(f"New train dataset row proportion: {train_row_percentage:.2f}%")
print(f"New test dataset row proportion: {test_row_percentage:.2f}%")

if abs(test_row_percentage - 20) < 5 and abs(train_row_percentage - 80) < 5:
    print("The new split is approximately 2:8 (test:train) by rows.")
else:
    print(f"The new split does not closely match 2:8 by rows. Actual split: {test_row_percentage:.2f}:{train_row_percentage:.2f}")

# 17. Check column consistency
if list(train_df.columns) == list(test_df.columns):
    print("\nNew train and test datasets have identical columns.")
else:
    print("\nError: New train and test datasets have different columns.")
    exit()

# 18. Save split datasets
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)
print(f"\nNew train dataset saved to: {train_output_path}")
print(f"New test dataset saved to: {test_output_path}")

# 19. Final verification
print("\nFinal verification:")
print(f"New train dataset rows: {len(train_df)}, unique students: {train_df['student'].nunique()}, unique knowledge: {train_df['knowledge'].nunique()}")
print(f"New test dataset rows: {len(test_df)}, unique students: {test_df['student'].nunique()}, unique knowledge: {test_df['knowledge'].nunique()}")
print(f"No duplicate students confirmed: {len(train_df[train_df['student'].isin(test_students)]) == 0}")
