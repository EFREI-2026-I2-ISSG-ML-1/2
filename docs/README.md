# EFREI_2026_I2_ML_1 2

# ST2MLE : Machine Learning for IT Engineers

## LAB 2 : Unsupervised learning and text classification

---

## Part 1: 
- Exploratory data analysis (EDA)
- Data pre-processing
- Clustering (k-means)
- Dimensionality reduction (PCA)

---

## Part 2: 
- Text pre-processing
- Text vectorization
- Text classification

---

## Part 1: EDA, Clustering and Dimensionality reduction  

### Context: Pima Indians Diabetes Dataset

The Pima Indians Diabetes Dataset contains medical diagnostic measurements of female Pima Indian patients aged 21 and older.

Collected by the **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)**, USA, the goal is to explore risk factors associated with the development of Type 2 diabetes.

Although the original task is a binary classification (diabetes or not), we will treat the dataset as **unlabeled** and explore it using **unsupervised techniques**:

- Exploratory Data Analysis (EDA)
- K-Means Clustering
- Principal Component Analysis (PCA)

---

### Objectives

By the end of Part 1, you will be able to:

- Perform EDA on numeric datasets
- Check missing data, outliers, scaling
- Apply K-Means clustering and interpret clusters
- Apply PCA for dimensionality reduction and visualization
- Compare clustering results using original features vs. reduced features

---

### Dataset Information

Download: https://www.openml.org/search?type=data&status=active&id=43582  
Or use: `openml.datasets.get_dataset(id)`

**Features:**

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic predisposition to diabetes |
| Age | Age in years |

Note: 8 numerical features, 768 instances. **Ignore the Outcome column.**

---

### Exercise 1: Exploratory Data Analysis (EDA)

1. Load the dataset using `pandas`
2. Display basic statistics (`describe()`)
3. Visualize distributions (histograms, boxplots)
4. Check for missing data or unrealistic zeros (e.g., Insulin)
5. Identify outliers (boxplots)
6. Standardize using `StandardScaler` (watch for data leakage)

---

### Exercise 2: K-Means Clustering

1. Apply K-Means with `k=2` to standardized data
2. Visualize clusters (scatter plot using two features)
3. Interpret the results

---

### Exercise 3: PCA and Visualization

1. Reduce to 2 components with PCA
2. Display explained variance ratio
3. Scatter plot using first 2 PCs, color by K-Means labels
4. Analyze PC1 and PC2 feature contributions
5. Inspect PCA loadings

---

### Exercise 4: K-Means on PCA-transformed data

1. Apply K-Means on PCA-reduced data
2. Compare with Exercise 2
3. Discuss pros and cons of original space vs PCA space

---

### Discussion

- When does PCA help clustering?
- How did scaling impact results?
- Are clusters meaningful? How could they be improved?

---

## Part 2: Text Classification using BoW, TF-IDF, Word2Vec and BERT

### Context: News classification (20 Newsgroups)

~20,000 documents in 20 newsgroups, including:

- `comp.graphics`
- `comp.os.ms-windows.misc`
- `comp.sys.ibm.pc.hardware`
- `comp.sys.mac.hardware`
- `comp.windows.x`

This makes it ideal for IT-related classification tasks.

---

### Dataset Information

- **Features**: Text from newsgroup messages
- **Target**: Newsgroup category (label)

---

### Objectives

- Preprocess text data
- Vectorize using: BoW, TF-IDF, Word2Vec, BERT
- Apply Naive Bayes & Logistic Regression
- Compare performance

---

### Exercise 1: Text Preprocessing

1. Load dataset using `fetch_20newsgroups()`
2. Filter for the 5 computing categories
3. Preprocess:
   - Lowercase
   - Remove punctuation (regex)
   - Lemmatize (spaCy)
4. Split into train/test (80/20) – check for leakage

---

### Exercise 2: Bag of Words (BoW)

1. Vectorize using `CountVectorizer` (remove stopwords)
2. Visualize vocabulary size
3. Train model (MultinomialNB or Logistic Regression)
4. Evaluate accuracy

---

### Exercise 3: TF-IDF

1. Vectorize with `TfidfVectorizer` (remove stopwords)
2. Compare average TF-IDF values for top 10 terms
3. Train same model as Exercise 2
4. Evaluate accuracy

**Question**: How does TF-IDF improve BoW?

---

### Exercise 4: Word2Vec

1. Vectorize with Gensim Word2Vec
2. Represent document = average of embeddings
3. Train (GaussianNB or Logistic Regression)
4. Evaluate accuracy

**Question**: Compare to BoW and TF-IDF. Key differences?

---

### Exercise 5: Doc2Vec

1. Vectorize using Gensim Doc2Vec
2. Train (GaussianNB or Logistic Regression)
3. Evaluate accuracy

**Question**: Comparison to previous models?

---

### Exercise 6: BERT Embeddings

1. Use Hugging Face Transformers
   - Pretrained BERT (e.g., `bert-base-uncased`)
   - Mean pooling of token embeddings
2. Train (Naive Bayes or Logistic Regression)
3. Evaluate accuracy

**Question**: How does BERT compare?

---

### Summary: Comparison of Models and Vectorizations

1. Compare accuracy for:
   - BoW
   - TF-IDF
   - Word2Vec
   - Doc2Vec
   - BERT
2. Bar chart of accuracy
3. Questions:
   - Which performs best?
   - Model complexity vs accuracy trade-offs?
   - Is BERT significantly better?

---

## Deliverables

Submit on Moodle:

- Jupyter notebook or `.py` file
- Screenshots of experiments
- Final comparison table + analysis
- Group of 2–3 students allowed (submit one report)


## Links

- https://moodle.myefrei.fr/mod/assign/view.php?id=108555
