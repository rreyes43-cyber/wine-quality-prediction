# SAD Mini-Analysis and Reflection Answers

## SAD mini-analysis

**Problem:** A wine producer needs a consistent preliminary way to screen wine samples using recorded physicochemical measurements.

**Users:** Quality-control staff, production supervisor, or data analyst.

**Inputs:** Fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and alcohol.

**Process:** Validate the required input columns, preprocess the measurements, apply the trained Random Forest model, and classify each sample.

**Outputs:** GOOD QUALITY or NEEDS IMPROVEMENT prediction labels, plus model evaluation results.

**Key limitation:** The model is trained on a particular red-wine dataset and may not generalize to other wine types, producers, regions, or testing conditions.

## Reflection questions

1. **What is the prediction target, and why is the original quality column excluded from the input features?**  
   The target is the binary quality category: GOOD QUALITY when the original score is 6 or higher, and NEEDS IMPROVEMENT when it is below 6. The original `quality` column is excluded because it is used to create the target. Including it as an input would reveal the answer to the model and cause target leakage.

2. **What does recall for the GOOD QUALITY class tell you? What could low recall mean for the business?**  
   Recall for GOOD QUALITY tells us what proportion of all truly good-quality samples the model correctly identifies. Low recall means some good samples are incorrectly classified as needing improvement, which could lead to unnecessary rechecks or delays.

3. **Why might a model trained on this red-wine dataset perform differently on another producer's wine samples?**  
   Another producer may use different grapes, production methods, regions, or testing conditions. Those differences can change the measurement patterns, so the training data may not represent the new samples.

4. **Why should the prediction be treated as a preliminary screening result rather than a final quality decision?**  
   The model is a classroom prototype trained on a limited public dataset. It does not replace laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control judgment.

5. **What additional data or testing would help improve the reliability of the system?**  
   Collect more representative samples from different producers, regions, vintages, and production conditions; obtain verified laboratory and sensory assessments; test the model on independent samples; and monitor its precision, recall, and F1-score over time.

## Evaluation results
Run `python train_model.py`. Copy the actual accuracy, precision, recall, F1-score, confusion matrix, and classification report from the terminal or `evaluation_metrics.txt` into the submission. Do not invent or estimate these values.
