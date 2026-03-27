# Adult Dataset EDA Report

## Overview
- Rows: 48842
- Features: 14
- Target: income
- Duplicate feature rows: 53
- Task type: binary income classification
- Source: UCI Adult / Census Income dataset

## How To Read This Dataset
- Each row represents one person from the census-derived sample.
- The target is whether annual income is above `50K`.
- `education-num` is an ordinal encoding of `education`, so the two columns carry overlapping information.
- `capital-gain` and `capital-loss` are extremely sparse and should be treated as zero-inflated features.
- Some categorical fields use `?` as a missing-like placeholder in addition to actual `NaN` values.

## Target Distribution
```text
        count  percentage
income                   
<=50K   37155       76.07
>50K    11687       23.93
```

## Data Quality
- The target is imbalanced: most rows belong to the `<=50K` class.
- There are both real `NaN` values and literal `?` markers in categorical columns.
- A small number of duplicate feature rows exist and should be reviewed before training.

```text
                nan_count  question_mark_count  total_missing_like
occupation            966                 1843                2809
workclass             963                 1836                2799
native-country        274                  583                 857
age                     0                    0                   0
capital-gain            0                    0                   0
capital-loss            0                    0                   0
education               0                    0                   0
education-num           0                    0                   0
fnlwgt                  0                    0                   0
hours-per-week          0                    0                   0
marital-status          0                    0                   0
race                    0                    0                   0
relationship            0                    0                   0
sex                     0                    0                   0
```

## Numeric Feature Summary
```text
            age      fnlwgt  education-num  capital-gain  capital-loss  hours-per-week
count  48842.00    48842.00       48842.00      48842.00       48842.0        48842.00
mean      38.64   189664.13          10.08       1079.07          87.5           40.42
std       13.71   105604.03           2.57       7452.02         403.0           12.39
min       17.00    12285.00           1.00          0.00           0.0            1.00
25%       28.00   117550.50           9.00          0.00           0.0           40.00
50%       37.00   178144.50          10.00          0.00           0.0           40.00
75%       48.00   237642.00          12.00          0.00           0.0           45.00
max       90.00  1490400.00          16.00      99999.00        4356.0           99.00
```

## Categorical Cardinality
```text
native-country    43
education         16
occupation        16
workclass         10
marital-status     7
relationship       6
race               5
sex                2
```

## Most Common Categories
### workclass
```text
workclass
Private             33906
Self-emp-not-inc     3862
Local-gov            3136
State-gov            1981
?                    1836
Self-emp-inc         1695
Federal-gov          1432
NaN                   963
```
### education
```text
education
HS-grad         15784
Some-college    10878
Bachelors        8025
Masters          2657
Assoc-voc        2061
11th             1812
Assoc-acdm       1601
10th             1389
```
### occupation
```text
occupation
Prof-specialty       6172
Craft-repair         6112
Exec-managerial      6086
Adm-clerical         5611
Sales                5504
Other-service        4923
Machine-op-inspct    3022
Transport-moving     2355
```
### native-country
```text
native-country
United-States    43832
Mexico             951
?                  583
Philippines        295
NaN                274
Germany            206
Puerto-Rico        184
Canada             182
```

## Income Differences By Target
```text
          age  education-num  capital-gain  capital-loss  hours-per-week
income                                                                  
<=50K   36.87            9.6        147.01         54.15           38.84
>50K    44.28           11.6       4042.24        193.53           45.45
```

## Strongest Numeric Signals
```text
education-num     0.333
age               0.230
hours-per-week    0.228
capital-gain      0.223
capital-loss      0.148
fnlwgt           -0.006
```

- `capital-gain` is zero in 91.74% of rows.
- `capital-loss` is zero in 95.33% of rows.
- Higher income is most associated with education level, age, and hours worked per week among the numeric columns.
- `fnlwgt` appears to have almost no linear relationship with the target.

## High-Income Rate By Key Categories
### sex
```text
        count  high_income_rate
sex                            
Male    32650             30.38
Female  16192             10.93
```
### workclass
```text
                  count  high_income_rate
workclass                                
Private           33906             21.79
Self-emp-not-inc   3862             27.89
Local-gov          3136             29.56
State-gov          1981             26.75
?                  1836             10.40
Self-emp-inc       1695             55.34
Federal-gov        1432             39.18
NaN                 963              7.68
```
### education
```text
              count  high_income_rate
education                            
HS-grad       15784             15.86
Some-college  10878             18.96
Bachelors      8025             41.28
Masters        2657             54.91
Assoc-voc      2061             25.33
11th           1812              5.08
Assoc-acdm     1601             25.80
10th           1389              6.26
```
### marital-status
```text
                       count  high_income_rate
marital-status                                
Married-civ-spouse     22379             44.61
Never-married          16117              4.55
Divorced                6633             10.12
Separated               1530              6.47
Widowed                 1518              8.43
Married-spouse-absent    628              9.24
Married-AF-spouse         37             37.84
```

## Recommended Report Extensions
- Add plots for age, hours-per-week, and education level split by income class.
- Show target rate by category for `education`, `occupation`, `workclass`, and `marital-status` to expose useful nonlinear patterns.
- Separate true missing values from `?` placeholders, because this dataset uses both.
- Add outlier notes for `capital-gain`, `capital-loss`, and `hours-per-week` because these features are heavily skewed.
- Document feature redundancy between `education` and `education-num` so downstream models do not double-count the same signal.
- Include fairness-oriented slices for `sex` and `race` if the report is meant to support modeling decisions.
- Add a preprocessing section describing imputation, encoding, duplicate handling, and class-imbalance strategy.
