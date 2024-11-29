**Ocular Disease Detection with Deep Learning**

This project uses deep learning to automate the diagnosis of multiple ocular diseases from fundus images. Leveraging Convolutional Neural Networks (CNNs) and transfer learning, it aims to provide an accurate diagnostic tool to assist healthcare professionals in early detection and management of eye diseases, reducing the risk of preventable blindness.

Table of Contents
Project Overview
Dataset Overview
Addressing Class Imbalance
Model Architecture
Data Preparation and Preprocessing
Model Training and Evaluation
Explainability and Model Interpretability
Limitations and Future Work
Conclusion
Project Overview
According to the World Health Organization (WHO), 2.2 billion people are visually impaired, with at least 1 billion cases being preventable. Early detection is critical for reducing the severity of eye conditions. This project employs transfer learning with VGG-19 to classify ocular diseases based on fundus images, ensuring clinical usefulness and reliability.

**Dataset Overview**

The project utilizes the ODIR-5K dataset, containing:

7,000 fundus images
Eight classes of ocular conditions:
Normal (N)
Diabetes (D)
Glaucoma (G)
Cataract (C)
Age-related Macular Degeneration (A)
Hypertension (H)
Pathological Myopia (M)
Other Diseases/Abnormalities (O

**Addressing Class Imbalance**

To ensure robust performance across all classes, this project handles class imbalance using:

**Data Augmentation:** Generates synthetic samples for underrepresented classes.
**Class Weights:** Adjusts the loss function to account for class imbalances.
**Sampling Strategies:** Employs oversampling techniques for minority classes.

**Model Architecture**

The classification system uses VGG-19 with customizations:

Fully Connected Layers: Added dense layers specific to ocular disease patterns.
Batch Normalization: Stabilizes training and accelerates convergence.
Dropout Regularization: Prevents overfitting by randomly deactivating neurons.

**Data Preparation and Preprocessing**

Steps Applied:
Data Loading: Organizing images with associated labels.
Data Splitting:
Training Set: 80% of the data.
Testing Set: 20% of the data.
One-Hot Encoding: Labels represented as 8-element vectors for multi-class classification.

**Model Training and Evaluation**

Optimizations:
Adam Optimizer: Adapts the learning rate dynamically for efficient training.
Categorical Cross-Entropy Loss: Measures prediction accuracy against one-hot encoded labels.
Mixed Precision Training: Speeds up computation by using lower-precision values on compatible GPUs.

**Regularization Techniques:**

Dropout: Prevents co-adaptations in neurons during training.
L2 Regularization: Penalizes large weights, encouraging simpler configurations.

**Evaluation Metrics:**

Accuracy: Overall correctness of predictions.
Precision: Measures false positives to avoid unnecessary treatments.
Recall (Sensitivity): Ensures diseases are not overlooked.
F1 Score: Balances precision and recall for comprehensive evaluation.

![image](https://github.com/user-attachments/assets/77cd8b00-effe-4117-a7bd-96e3c973225a)    ![image](https://github.com/user-attachments/assets/6cdcb791-4de2-4f91-9d1b-81aa7027c9c5)

The model demonstrated strong training performance, with a consistent decrease in loss and steady improvement in accuracy over the epochs. Validation accuracy showed an encouraging upward trend in the early stages, reflecting the model's ability to generalize well. These results highlight the effectiveness of the model architecture and training strategy, with opportunities to further optimize for validation performance.

![image](https://github.com/user-attachments/assets/ea3d946a-ca39-431a-9c17-3636f57aced2)

The confusion matrix illustrates the model's classification performance across eight ocular disease categories, highlighting the percentage of correct predictions (diagonal) and areas of misclassification (off-diagonal) to identify strengths and improvement areas.


![image](https://github.com/user-attachments/assets/b2e88df2-258f-4bd4-95ec-42790d798cdf)

The ROC curve shows the model's ability to distinguish between classes, with the Area Under the Curve (AUC) indicating high classification performance across most classes, demonstrating the model's robustness in predicting ocular diseases.


**Explainability and Model Interpretability**
In medical applications, interpretability is crucial. This project uses Grad-CAM to highlight regions in each image influencing the model's predictions, providing transparency and clinical reliability.

**Limitations and Future Work**
Limitations:
Dataset Size: A larger dataset could improve performance.
Class Imbalance: Rare disease classes remain challenging despite handling techniques.
Future Work:
Ensemble Models: Combine strengths of multiple architectures to improve accuracy.
Additional Datasets: Incorporate more diverse datasets for better generalization.
Extended Interpretability: Use methods like SHAP or LIME for deeper insights.

**Conclusion**
This project demonstrates the potential of deep learning to aid in early detection of ocular diseases. By leveraging VGG-19 and employing techniques to improve accuracy and interpretability, this diagnostic tool provides a scalable solution for preventive ophthalmology.

**Resources**
Dataset: ODIR-5K
