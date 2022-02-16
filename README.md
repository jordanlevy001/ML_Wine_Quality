# Machine Learning: Wine Quality Scores

## Project Overview
Use machine learning to predict the overall quality score (likeability) of various wines

## Technology/Software Utilized
- Google Colab
- TensorFlow, Keras
- Scikit-learn
- Pandas
- Plotly
- Seaborn
- NumPy
- Matplotlib
- hvPlot

## Data
Data Source: [Kaggle](https://www.kaggle.com/yasserh/wine-quality-dataset)

Columns:
1. Fixed Acidity
2. **Volatile Acidity**
3. **Citric Acid**
4. Residual Sugar
5. Chlorides
6. Free Sulfure Dioxide
7. Total Sulfur Dioxide
8. Density -- this column can also be dropped
9. pH
10. **Sulphates**
11. **Alcohol**
12. **Quality --> OUTPUT: this is the overall quality score**
13. ID -- this column was dropped

<img width="1117" alt="Wine Stats" src="https://user-images.githubusercontent.com/88804543/154178721-63f21163-eb1d-46bb-aaa1-35f497011f22.png">


## Exploratory Data Analysis, Processing, Visualization

Looking at the quality score counts:

<img width="434" alt="Quality Score Counts" src="https://user-images.githubusercontent.com/88804543/154179935-911cc753-3884-4529-9c64-03723b0a8262.png">

- Only (6/1143) 0.5% of wines had a quality score of 3
- Only (16/1143) ~1% of wines had a quality score of 8
- Only (33/1143) ~3% of wines had a quality score of 4


Examining the trends when grouped by quality score:

<img width="1092" alt="Group by QScore" src="https://user-images.githubusercontent.com/88804543/154178843-34e1d2f3-8491-411a-99d1-e3ea16b052e1.png">

The trends observed from this table:
- Volatile Acidity decreases with increasing quality score 
- Citric Acid increases with increasing quality score
- Sulphates increases with increasing quality score
- Alcohol increases with increasing quality score

Taking a closer look at: volatile acidity, citric acid, sulphates

<img width="973" alt="Quality1" src="https://user-images.githubusercontent.com/88804543/154179178-5713213c-0599-4c52-81d8-a5e9218a0160.png">


Taking a closer look at: chlorides, alcohol, pH

<img width="948" alt="Quality2" src="https://user-images.githubusercontent.com/88804543/154179208-a2780459-c9db-4aa1-8a34-a73ccd99a0d6.png">


#### Principal Component Analysis (PCA)
PCA confirms the Volatile Acidity, Citric Acid, Sulphates, Alcohol metrics together account for 99% of the variance

<img width="448" alt="Screen Shot 2022-02-15 at 11 24 54 AM" src="https://user-images.githubusercontent.com/88804543/154134246-ac22e71b-dca0-4788-b7db-29d03f25ad93.png">

Here is the updated DataFrame, with the highest weighted metrics: Volatile Acidity, Citric Acid, Sulphates, Alcohol and quality (which is needed as the output).

<img width="446" alt="Reduced DF" src="https://user-images.githubusercontent.com/88804543/154179638-456ccb74-26dc-4c93-a857-7784cb9d89e4.png">


## Analysis: Linear Regression, Random Forest Classifier

### Linear Regression
Supervised learning can be divided into regression and classification. Regression is used to predict continuous variables. Whereas classification is used to predict discrete outcomes. For this wine analysis, the target variable (what we're trying to predict), is a quality score -- which is a continuous variable. The wine quality score is a numerical value within a given range, making it a continuous variable. In both classification and regression issues, the data is divided into features and targets. Features are the variable used to inform the prediction. The target/output is the predicted outcome.

#### Alcohol vs Quality

<img width="391" alt="Alcohol v Quality" src="https://user-images.githubusercontent.com/88804543/154180858-b220da54-f47a-4172-988b-cdbbeb358358.png">


#### Volatile Acidity vs Quality

<img width="317" alt="Volatile Acidity vs Quality" src="https://user-images.githubusercontent.com/88804543/154180866-c0a56ffb-03d8-4a90-8a30-e414d35101c0.png">

Linear Regression Equation: -1.828x + 6.628


### Random Forest Classifier
Random forest classifiers are a type of ensemble learning model that combines multiple smaller models into a more robust and accurate model. Random forest models use a number of weak learner algorithms (decision trees) and combine their output to make a final classification (or regression) decision. Structurally, random forest models are very similar to neural networks. Random forest models have been popular in machine learning algorithms for many years because of their scalability and robustness. Both output and feature selection of random forest models are easy to interpret. Random forest models can also easily handle outliers and nonlinear data.

The n_estimators will allow us to set the number of trees that will be created by the algorithm. Generally, the higher number makes the predictions stronger and more stable, but can slow down the output because of the higher training time allocated. The best practice is to use between 64 and 128 random forests. This analysis used 128.


### Random Forest Classifer vs Neural Networks
Random forest models will only handle tabular data, so data such as images or natural language data cannot be used in a random forest without heavy modifications to the data. Neural networks can handle all sorts of data types and structures in raw format or with general transformations (such as converting categorical data).

In addition, each model handles input data differently. Random forest models are dependent on each weak learner being trained on a subset of the input data. Once each weak learner is trained, the random forest model predicts the classification based on a consensus of the weak learners. In contrast, deep learning models evaluate input data within a single neuron, as well as across multiple neurons and layers.

As a result, the deep learning model might be able to identify variability in a dataset that a random forest model could miss. However, a random forest model with a sufficient number of estimators and tree depth should be able to perform at a similar capacity to most deep learning models.

## Results

### Random Forest Classifer

The Random Forest Classifier Accuracy Score was 63.3%

<img width="427" alt="RFC Acc" src="https://user-images.githubusercontent.com/88804543/154317771-e676202a-452b-4184-90fa-013a5022e5b6.png">


<img width="405" alt="RFC" src="https://user-images.githubusercontent.com/88804543/154319119-0811811d-f2fd-4ab1-8cd8-086bb4d9a0b7.png">




### Optimizing the Random Forest Classifier

As discussed above:

- Only (6/1143) 0.5% of wines had a quality score of 3
- Only (16/1143) ~1% of wines had a quality score of 8
- Only (33/1143) ~3% of wines had a quality score of 4

The quality scores of 3, 4, 8 were dropped from the model in an attempt to optimize the model. This increased the accuracy of the model by ~2%.

<img width="422" alt="Opt RFC Acc Score" src="https://user-images.githubusercontent.com/88804543/154317756-04be8fcd-5930-4fbd-9e60-0a2faae08022.png">






















