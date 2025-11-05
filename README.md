
🤖 Resume Screening & Job Recommendation System
https://huggingface.co/spaces/arunnil/resume-app

An intelligent web application built with Flask and scikit-learn to automatically parse, categorize, and recommend jobs from resume files.

🚀 Live Demo / Screenshot

<img width="1465" height="883" alt="image" src="https://github.com/user-attachments/assets/2afc9538-862e-4ef0-98b1-33bd219e21cb" />

This application provides a clean dashboard to analyze resumes. After uploading a PDF or TXT file, the app displays the candidate's parsed info, the resume's predicted category, a recommended job title, and a confidence chart.

✨ Features

Resume Parsing: Automatically extracts key information (Name, Email, Phone, Skills, Education) from PDF and TXT resumes using PyPDF2 and regex.

ML-Powered Categorization: Uses a Random Forest Classifier to categorize the resume into one of 24 profiles (e.g., "Data Science," "HR," "Engineering").

Job Recommendation: Employs a second Random Forest model to recommend a specific job title based on the resume's content.

Confidence Dashboard: Displays the top 6 category predictions in a horizontal bar chart to show the model's confidence.

Web Interface: A responsive, dark-mode UI built with Flask and Chart.js.

🛠️ Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-learn (RandomForestClassifier, TfidfVectorizer)

Data Handling: Pandas, NumPy

File Parsing: PyPDF2, re (Regex)

Frontend: HTML, CSS, JavaScript, Chart.js

Prototyping: Jupyter Notebook (Copy_of_Resume_Screener.ipynb)

How It Works

The application uses a two-model machine learning pipeline:

File Upload: A user uploads a .pdf or .txt resume.

Parsing & Cleaning:

The pdf_to_text function (using PyPDF2) extracts the raw text.

A series of regex functions (extract_email_from_resume, extract_skills_from_resume, etc.) parse the raw text to find candidate details.

The cleanResume function removes URLs, special characters, and other noise.

Model 1: Categorization

The cleaned text is passed to the tfidf_vectorizer_categorization.

The resulting TF-IDF vector is fed into the rf_classifier_categorization (a Random Forest model) to get the primary category (e.g., "Data Science").

The predict_proba method is used to get confidence scores for the dashboard chart.

Model 2: Job Recommendation

The text is passed to a separate tfidf_vectorizer_job_recommendation.

This vector is fed into the rf_classifier_job_recommendation to predict a specific job title (e.g., "Data Scientist").

Render Output: All extracted info and predictions are sent to the resume.html template and displayed to the user.
