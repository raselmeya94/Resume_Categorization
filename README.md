
# Resume Categorization  

This project focuses on automating the categorization of resumes using Natural Language Processing (NLP) and machine learning techniques. The goal is to analyze resume data and classify resumes into their respective job categories efficiently.  

---

## Project Structure  

- `resume_categorization_notebook.ipynb`: A Jupyter Notebook that contains the implementation for data preprocessing, feature extraction, model training, and evaluation.  
- `script.py`: A Python script to automate the process of categorizing resumes based on the trained model and vectorizer.  
- `test_data/`: Directory containing the test resumes in PDF format.  
- `requirements.txt`: List of all Python packages required for this project.  

---

## Description  

This project employs various machine learning models and a deep learning approach to identify job categories from resumes. Using datasets, resumes are processed and classified using models such as:  
- Random Forest Classifier  
- Logistic Regression  
- K-Nearest Neighbor  
- Support Vector Machine (SVM)  
- Deep Learning: Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM)  

The model training is performed on preprocessed data, and the results are used to create a functional script that categorizes resumes automatically.  

---

## Requirements  

Ensure the following are installed before proceeding:  
- Python 3.10.12 or higher  
- Jupyter Notebook  
- Necessary Python libraries (specified in `requirements.txt`)  

---

## Installation  

1. Clone the repository to your local machine:  
   ```bash  
   git clone https://github.com/raselmeya94/Resume_Categorization.git  
   ```  

2. Navigate to the project directory:  
   ```bash  
   cd Resume_Categorization  
   ```  

3. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

---

## Workflow  

1. **Data Preprocessing**:  
   - Split the dataset into `resume_data` (training set) and `resume_test_data` (testing set).  
   - Clean the text data by removing unnecessary symbols, spaces, and irrelevant content.  

2. **Feature Extraction**:  
   - Use NLP techniques such as TF-IDF vectorization to extract meaningful features from the resumes.  

3. **Model Training**:  
   - Train multiple models to identify the best-performing one for classifying resumes into categories.  
   - Save the trained classifier (`best_clf.pkl`) and the vectorizer (`tfidf.pkl`) as pickle files.  

4. **Automated Categorization**:  
   - Use `script.py` to load the test data (PDF resumes in `test_data/`), vectorize the content, and predict categories.  
   - Organize resumes into corresponding folders based on their predicted category.  

---

## Running the Project  

1. Train the models and generate pickle files:  
   Open the Jupyter Notebook:  
   ```bash  
   jupyter notebook resume_categorization_notebook.ipynb  
   ```  
   Follow the instructions in the notebook to train the models and generate the necessary `best_clf.pkl` and `tfidf.pkl` files.  

2. Place the test resumes in the `test_data/` folder.  

3. Run the categorization script:  
   ```bash  
   python script.py  
   ```  
   Provide the path to the `test_data/` folder when prompted.  

---

## Example Outputs  

- Trained Model: `best_clf.pkl`  
- Vectorizer: `tfidf.pkl`  
- Categorized resumes in the following folder structure:  
  ```
  categorized_resumes/  
    ├── ENGINEERING  
    ├── FINANCE  
    ├── HEALTHCARE  
    ├── TEACHER  
    ├── ...  
  ```  

---

## Contributing  

We welcome contributions to enhance the functionality of this project. To contribute:  
1. Fork the repository.  
2. Create a feature branch.  
3. Submit a pull request describing your changes.  

---

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.  

---

For more details and hands-on usage, refer to the main [notebook](https://github.com/raselmeya94/Resume_Categorization/blob/main/resume_categorization_notebook.ipynb).  

---  

This version provides a clear and concise structure for your GitHub repository's `README.md`, tailored to an audience that may want to contribute or understand the implementation.
