import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score


from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def LDA_QDA_analysis(data_frame1, data_frame2):

    data_frame1['Group'] = 0
    data_frame2['Group'] = 1 
    
    data_frame1 = data_frame1.reset_index(drop=True)
    data_frame2 = data_frame2.reset_index(drop=True)
    
    combined_data = pd.concat([data_frame1, data_frame2], ignore_index=True)    
    combined_data.index = combined_data.index.astype(str)
    
     # Ensure the columns are numeric
    combined_data['symbiont_branch_dnds_avg'] = pd.to_numeric(combined_data['symbiont_branch_dnds_avg'], errors='coerce')
    combined_data['non_symbiont_branch_dnds_avg'] = pd.to_numeric(combined_data['non_symbiont_branch_dnds_avg'], errors='coerce')
    
    combined_data = combined_data.dropna(subset=['symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg'])
    
    print("Number of samples per group:")
    print(combined_data['Group'].value_counts())
    
    
    X = combined_data[['symbiont_branch_dnds_avg', 'non_symbiont_branch_dnds_avg']]
    y = combined_data['Group']  

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    
    # class imbalance correction
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print("Number of samples per group after SMOTE:")
    print(pd.Series(y_train_res).value_counts())
    
    # Apply LDA
    lda = LDA()
    lda.fit(X_train_res, y_train_res)
    y_pred_lda = lda.predict(X_test)
    print("LDA Classification Report:")
    print(classification_report(y_test, y_pred_lda))

    # Apply QDA
    qda = QDA()
    qda.fit(X_train_res, y_train_res)
    y_pred_qda = qda.predict(X_test)
    print("QDA Classification Report:")
    print(classification_report(y_test, y_pred_qda))

    # Apply LDA with cross-validation without SMOTE
    lda = LDA()
    lda_scores = cross_val_score(lda, X_train, y_train, cv=5, scoring='accuracy')
    print("LDA Cross-Validation Accuracy Scores without SMOTE:", lda_scores)
    print("LDA Mean Cross-Validation Accuracy without SMOTE:", lda_scores.mean())

    # Apply LDA with cross-validation with SMOTE
    lda.fit(X_train_res, y_train_res)
    lda_scores_res = cross_val_score(lda, X_train_res, y_train_res, cv=5, scoring='accuracy')
    print("LDA Cross-Validation Accuracy Scores with SMOTE:", lda_scores_res)
    print("LDA Mean Cross-Validation Accuracy with SMOTE:", lda_scores_res.mean())