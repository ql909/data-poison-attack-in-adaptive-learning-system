# AFM

Requirements: `..._train.csv` and `..._test.csv` datasets from Google Drive.

File explanation:
* `main.py`: Script to run the AFM modeling.
* `..._results_....json`: Output of the script depending on the input file.

## Results for running on `HamptonAlg` dataset (max. iter = 100)

```
File size of 'HamptonAlg_train.csv' is: 199,070 rows, 16 columns
Unique students: 59
Unique skills: 87
File size of 'HamptonAlg_test.csv' is: 41,167 rows, 16 columns
Unique students: 57
Unique skills: 87

Training set performance:
Accuracy: 0.785 | AUC: 0.782
Test set performance:
Accuracy: 0.767 | AUC: 0.755
```

## Results for running on `Assistment` dataset (max. iter = 100)

```
File size of 'Assistment_challenge_train.csv' is: 754,025 rows, 83 columns
Unique students: 1,709
Unique skills: 92
File size of 'Assistment_challenge_test.csv' is: 188,727 rows, 83 columns
Unique students: 1,688
Unique skills: 92

Training set performance:
Accuracy: 0.661 | AUC: 0.655
Test set performance:
Accuracy: 0.658 | AUC: 0.647
```