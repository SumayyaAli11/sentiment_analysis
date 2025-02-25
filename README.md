# Sentiment Analysis Project

This project focuses on performing sentiment analysis on customer reviews using Python. The project leverages two primary techniques:
1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)** - A rule-based sentiment analysis tool.
2. **Machine Learning Approach** - Using Support Vector Machine (SVM) and Naive Bayes classifiers for sentiment classification.

The project is implemented in a Jupyter Notebook (`SentimentAnalysisWithPreprocessing_nltk.ipynb`) and includes text preprocessing, exploratory data analysis, and model evaluation.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Techniques Used](#techniques-used)
3. [Text Preprocessing Steps](#text-preprocessing-steps)
4. [Sentiment Analysis with VADER](#sentiment-analysis-with-vader)
5. [Machine Learning Approach](#machine-learning-approach)
6. [Results and Evaluation](#results-and-evaluation)
7. [Conclusion](#conclusion)
8. [Future Work](#future-work)

---

## Project Overview
The goal of this project is to analyze customer reviews and classify them into **Positive**, **Negative**, or **Neutral** sentiments. The dataset used contains 10,000 reviews, which are preprocessed and analyzed using two techniques:
- **VADER**: A lexicon and rule-based sentiment analysis tool.
- **Machine Learning**: Using SVM and Naive Bayes classifiers.

---

## Techniques Used
1. **VADER Sentiment Analysis**:
   - A pre-trained model that uses a combination of lexical features and rules to determine sentiment.
   - Outputs sentiment scores: `Positive`, `Negative`, `Neutral`, and `Compound`.

2. **Machine Learning Approach**:
   - Text preprocessing: Lowercasing, removing HTML tags, punctuation, stopwords, and stemming.
   - Text vectorization using `CountVectorizer`.
   - Classification using:
     - Support Vector Machine (SVM)
     - Gaussian Naive Bayes

---

## Text Preprocessing Steps
The following preprocessing steps are applied to the review text:
1. **Lowercasing**: Convert all text to lowercase.
2. **Removing HTML Tags**: Clean any HTML tags present in the text.
3. **Removing Punctuation**: Remove special characters and punctuation.
4. **Removing Stopwords**: Eliminate common stopwords (e.g., "the", "is", "and").
5. **Stemming**: Reduce words to their root form (e.g., "running" → "run").

---

## Sentiment Analysis with VADER
- VADER is used to generate sentiment scores for each review.
- The `compound` score is used to classify reviews into:
  - **Positive**: `compound >= 0.05`
  - **Neutral**: `-0.05 < compound < 0.05`
  - **Negative**: `compound <= -0.05`

---

## Machine Learning Approach
1. **Text Vectorization**:
   - The preprocessed text is converted into numerical features using `CountVectorizer`.
2. **Model Training**:
   - The dataset is split into training and testing sets.
   - Two models are trained:
     - Support Vector Machine (SVM)
     - Gaussian Naive Bayes
3. **Model Evaluation**:
   - The models are evaluated using accuracy and classification reports.

---

## Results and Evaluation
- **SVM** achieved an accuracy of **87.15%**.
- **Naive Bayes** achieved an accuracy of **36.9%**.
- SVM outperformed Naive Bayes in sentiment classification.

### Example Predictions
1. **Using SVM**:
   - Input: `"Good food"`
   - Output: `['Positive']`
2. **Using Naive Bayes**:
   - Input: `"Good food"`
   - Output: `['Neutral']`

---

## Conclusion
- **SVM** is more effective for sentiment analysis on this dataset compared to Naive Bayes.
- Sentiment analysis can be further improved using advanced models like **OpenAI GPT-3.5** or **BERT**.

---

## Future Work
1. Experiment with other machine learning algorithms like **Random Forest** and **Multinomial Naive Bayes**.
2. Use advanced deep learning models like **BERT** or **GPT-3.5** for more accurate sentiment analysis.
3. Expand the dataset to include more diverse reviews for better model generalization.

---

## Requirements
- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `scikit-learn`
  - `tqdm`

---

## How to Run
1. Clone the repository.
2. Open the Jupyter Notebook `SentimentAnalysisWithPreprocessing_nltk.ipynb`.
3. Run the notebook cells to perform sentiment analysis.

---

## Dataset
The dataset used in this project is stored in `McDonald_review.csv`. It contains the following columns:
- `reviewer_id`
- `store_name`
- `category`
- `store_address`
- `latitude`
- `longitude`
- `rating_count`
- `review_time`
- `review`
- `rating`

---

## License
This project is open-source and available under the MIT License.

---

## Contact
For any questions or suggestions, feel free to reach out:
- **Sumayya Ali**
- **Email**: sumayyaali.work@gmail.com
- **GitHub**: [https://github.com/SumayyaAli11]
