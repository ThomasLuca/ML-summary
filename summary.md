# Machine Learning summary

## 1. The ML landscape

### ML types

| Supervised                             | Unsupervised                   | Reinforcement           |
| -------------------------------------- | ------------------------------ | ----------------------- |
| Labeled data                           | No labels                      | Decision process        |
| Direct feedback                        | No feedback                    | Reward system           |
| Predict outcome/future                 | Find hidden structures in data | Learn series of actions |
| eg: Classification or regression\[^1\] | eg: Anomaly detection          | eg: alphago             |

\[^1\]: Regression = predict a target value from sample's feature

### Model based vs instance based learning

| Model-based                                             | Instance-based                                                     |
| ------------------------------------------------------- | ------------------------------------------------------------------ |
| Evaluate a mathematical function on the unseen instance | Measure similarities between unseen instance and training instance |

### Training a model

1. Choose a parameterized model family (`life satisfaction = θ0 + θ1 ∙ GDP_per_capita`)
1. Find parameter values that maximize a fitness function or minimize a cost function

#### Testing and validation

![train using train and test data](./img/testing_train_test.png)

![train using train, test and validation data](./img/testing_train_test_val_png)

#### Overfitting

Model doesn't generalize enough. It learns your specific training data but underperforms on new data.

Possible cures:

- Use a bigger dataset
- Simplify model
- Reduce the noice in the dataset

#### Underfitting

When the performance is even bad on the training data

Possible cures:

- Increase number of parameters
- Add more features
- Reduce the regularization parameters

### ML workflow

![Machine learning workflow](./img/ml_workflow.png)

### Problems with bad datasets

**Sampling bias**: Dataset can be non-representative if it has an underrepresented classes.

Garbage in == Garbage out: Bad dataset is guaranteed to lead to a bad (trained) model.

## 2. End-to-end ML project

### Workflow

![Project workflow](./img/project_workflow.png)

### Exploratory Data Analysis (EDA)

1. Get an initial feel of the data
1. Visualize and gain insight
1. Prepare the data

#### What to do with missing values

- Remove entry
- **Imputation**: replace by mean, median, 0, ...

#### Categorical attributes

Attributes that can only take a limited number of values (eg: T-shirt sizes)

**one-hot-encoding**: Use on categorical variables to transform them into a format the model can understand.

#### Features scaling

Some extremely big or small features may have an abnormally large impact on the model.
This can be solved by _rescaling_ them using the following techniques:

- **Normalization** (min-max scaling): $x\_{norm} = \frac{x - min(x)}{max(x) - min(x)}$
- **Standardization**: $x\_{stand} = \frac{x - mean(x)}{standard_deviation(x)}$

## 3. Classification

Classification always happens by _supervised learning_ models

### Performance Metrics

#### Accuracy

**Accuracy**: the percentage of predicted labels that corresponds with the ground truth label.

$Accuracy = \frac{TP + TN}{Total}$

#### Confusion Matrix

Columns are predicted labels, rows are true labels

|               | Automobile     | No Automobile  |
| ------------- | -------------- | -------------- |
| Automobile    | True Positive  | False Negative |
| No Automobile | False Positive | True Negative  |

#### Precision and recall

Accuracy of the positive predictions: $precision = \frac{TP}{TP+FP}$

How many of the actual positives are detected?: $recall = \frac{TP}{TP+FN}$

**When do we want high precision?**

- When false positives are costly (eg: medical predictions, fraud detection). You really don't want falsely flag a condition.
- In imbalanced datasets where the majority class is the negative one.

**When do we want high accuracy?**

- False negatives are costly (eg: cancer detection or nsfw filters)
- Information retrieval: recall helps ensure that all relevant documents or information are retrieved

#### F1 score

It combines the precision and recall of a model into a single metric.

$F_{1} = 2 . \frac{precision . recall}{precision + recall}$

### Binary Classification

**Binary**: Only two classes.

- **Decision boundary**: hypersurface that partitions underlying vector space into two sets, one for each class.
- **Score/Class Probability**: $\hat{y}(x^{(i)}) = \begin{cases} +1   if h_{\theta}(x^{(i)}) \geq T \\ -1   if h_{\theta}(x^{(i)}) < T \end{cases}$   (T = threshold as hyperparam)

#### Choosing a threshold

##### Precision vs Recall (Choosing a threshold)

![Precision vs recall](./img/precision_vs_recall.png)

##### ROC curve and Area Under The Curve (AUC)

![Area under curve](./img/AUC.png)

### Multiclass classification

#### One-vs-Rest

Turn multiclass into binary classification (eg: classes [green, blue, red] -> one-vs-rest: [green, rest[blue, red]])

- Classification based on voting
- NumClass * (NumClass – 1)/2 classifiers to train

## 4. Training models

### Linear regression

- **Assumption** (Inductive bias): There is a linear relationship between the input features and the target.
- $Price = \Theta_{0} + Bedrooms * \Theta_{1}$
  - $ \Thata_{0} $: intercept Bias
  - $ \Thata_{1} $: slope weight
- Goal: find optimal parameter that defines line that best fits the data
- The prediction is the weighted sum of the input features with an additional bias
- $\hat{y} = h_{\theta}(x) = \theta \cdot x $
  - $\hat{y}: prediction$
  - x: input features, extended with a “1” value (as bias)
  - $\theta$: model parameters

![Linear regression vector](./img/linreg_vector.png)


####  Linear regression training


