# BLG 454E 2023 Term Project, Team 2

## Members
* Bilgenur Çelik - 150200063
* Fatih Baskın - 150210710

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
imblearn
```

## Project Structure

Project's main structure is in the project.ipynb, it can be run directly. The folder data must be present at the same working directory, with the training and testing csv inside. 

Implementation of the PCA, LDA, Factor Analysis, imputers, normalizers are also put seperately in the preprocessing folder. It is not necessary to have it, since the project.ipynb already contain their implementation. 

training_trails directory contains auto testers, which were used to test our code quickly, again, it is not necessary to have to make project.ipynb work. EDA folder contains some outputs of the our old code, where it contains null percentages and field types of columns. 

eda.ipynb is a seperate code which contains olny the EDA operations. It is not necessary to have to make project.ipynb to work.

Inside the data folder, aps_failure_test_set.csv and aps_failure_training_set.csv must be present to make project.ipynb work.
