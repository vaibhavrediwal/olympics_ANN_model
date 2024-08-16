# Olympic Athlete Medal Prediction - ANN Model

## Project Overview

This project uses an Artificial Neural Network (ANN) to predict whether an Olympic athlete will win a medal based on various features such as age, sport, and country. The model is trained on historical Olympic data and aims to identify potential medal winners.

## Data Description

The dataset contains information about Olympic athletes, including:

- Age
- Height
- Weight
- Sex
- Team
- National Olympic Committee (NOC)
- Year
- Season (Summer/Winter)
- Sport

The target variable is whether the athlete won a medal (1) or not (0).

## Model Architecture

The ANN model consists of:

- Input layer
- 4 hidden layers with 128, 64, 32, and 16 neurons respectively
- Batch Normalization after each hidden layer
- Dropout (0.3) for regularization
- Output layer with sigmoid activation

## Results

### Model Performance

- Test Accuracy: 69.21%

### Classification Report

|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| No Medal (0) | 0.91      | 0.71   | 0.80     | 46,290  |
| Medal (1)    | 0.26      | 0.60   | 0.36     | 7,934   |
| Accuracy     |           |        | 0.69     | 54,224  |
| Macro Avg    | 0.59      | 0.65   | 0.58     | 54,224  |
| Weighted Avg | 0.82      | 0.69   | 0.73     | 54,224  |

### Feature Importance

1. Season (0.050643)
2. Sport (0.041975)
3. Age (0.041149)
4. NOC (0.039788)
5. Year (0.032650)
6. Weight (0.021006)
7. Sex (0.017289)
8. Team (0.010881)
9. Height (0.003660)

## Key Insights

1. The model shows improved performance in identifying medal winners compared to baseline models, with a recall of 0.60 for medal winners.
2. There's a trade-off between precision and recall for medal prediction, resulting in a lower overall accuracy but better minority class detection.
3. Season, Sport, and Age are the most important features in predicting medal outcomes.
4. The model's balanced performance suggests it's effective in handling the class imbalance present in Olympic medal data.

## Future Improvements

1. Feature Engineering: Create new features based on top important ones, such as sport-specific age categories.
2. Hyperparameter Tuning: Optimize model hyperparameters using techniques like grid search or random search.
3. Ensemble Methods: Combine the ANN with other models (e.g., Random Forests, Gradient Boosting) for potentially better performance.
4. Threshold Adjustment: Experiment with different decision thresholds to optimize the balance between precision and recall for medal winners.

## Conclusion

This ANN model provides a nuanced approach to predicting Olympic medal winners, balancing the trade-off between identifying potential medalists and overall accuracy. The insights gained from feature importance can be valuable for understanding key factors in Olympic success.

