# 📰 Fake News Detection using Machine Learning

This project is designed to identify whether a given news article is **Fake** or **True** using a machine learning-based classification system. Built on a structured dataset and trained with multiple models, this solution offers a reliable baseline for fake news detection.

---

## 📂 Dataset

The dataset used comprises two CSV files:

* `Fake.csv`: Contains fake news articles
* `True.csv`: Contains authentic news articles

These datasets are **sourced from a publicly available Kaggle dataset** and include columns such as `title`, `text`, `subject`, and `date`.

---

## 🧠 Models Used

The project implements four different supervised learning models:

| Model                        | Description                          |
| ---------------------------- | ------------------------------------ |
| Logistic Regression          | Baseline linear classifier           |
| Decision Tree Classifier     | Rule-based classifier                |
| Random Forest Classifier     | Ensemble of decision trees (Bagging) |
| Gradient Boosting Classifier | Ensemble using boosting strategy     |

Each model is trained using **TF-IDF vectorization** to convert news text into numerical features.

---

## 🔍 Workflow

1. **Data Cleaning**: Combined both datasets, removed unnecessary columns (`title`, `subject`, `date`), and added a `class` column (1 = Fake, 0 = True).
2. **Preprocessing**: Text lowercasing, punctuation removal, regex cleanup, and digit filtering.
3. **Splitting**: 75% training and 25% testing split using `train_test_split`.
4. **Model Training**: Trained all 4 models on TF-IDF features.
5. **Evaluation**: Evaluated using `accuracy_score` and `classification_report`.

---

## 📊 Model Performance

| Model                        | Accuracy (%) |
| ---------------------------- | ------------ |
| Logistic Regression          | \~98.4%      |
| Decision Tree Classifier     | \~99.5%      |
| Random Forest Classifier     | \~98.6%      |
| Gradient Boosting Classifier | \~99.5%      |

All models achieved high accuracy, with ensemble methods slightly outperforming the rest.

---

## 🧪 Manual Testing Feature

You can manually test news headlines/articles by inputting text and seeing predictions from all four models:

```python
Enter the news text for manual testing: "The government has banned all social media apps."
```

---

## 📌 Limitations

* The dataset is static and was collected during a specific time window. It may not reflect recent or evolving fake news patterns.
* The models are not updated with real-time news data, so their understanding of recent language or trends may be limited.
* Some domain-specific or satirical content might be misclassified.

---

## 🚀 Future Scope

* **Real-time News Stream Integration**: Enhance the model with live news feeds using APIs for dynamic adaptability.
* **Deep Learning Approaches**: Incorporate LSTM, BERT, or transformer-based models for better contextual understanding.
* **Multilingual Fake News Detection**: Extend the system to handle fake news in regional and global languages.
* **Explainability Features**: Add model explainability (e.g., SHAP, LIME) to interpret predictions.

---

## 📄 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). You are free to use, modify, and distribute this code with proper attribution.

---

## 🙌 Acknowledgements

* Dataset: [Kaggle Fake and True News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
