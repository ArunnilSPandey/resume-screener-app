from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle

app = Flask(__name__)

# Load models===========================================================================================================
try:
    rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
    tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
    rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
    tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please make sure all .pkl files are in the 'models' directory.")
    rf_classifier_categorization = None  # Handle gracefully
except Exception as e:
    print(f"An unexpected error occurred loading models: {e}")
    rf_classifier_categorization = None


# Clean resume==========================================================================================================
def cleanResume(txt):
    # Use raw strings (r'...') for all regex patterns
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# --- PREDICTION LOGIC MUST MATCH TRAINING LOGIC ---
# NOTE: Your 'predict_category' function was defined but not used in the /predict route.
# The route was predicting directly. I am keeping *your* logic from the route.
# If your model was *actually* trained on skills only, this logic needs to change.
# Based on the code in /predict, I am assuming the model was trained on clean_text.
# def predict_category(resume_text): ... (This function is not being called)


# Prediction and Category Name
def job_recommendation(resume_text):
    # Assuming it's trained on cleaned full text:
    resume_text_cleaned = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text_cleaned])
    recommended_job = rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    return recommended_job


def pdf_to_text(file):
    reader = PdfReader(file)
    text = ''
    # Use the more standard way to iterate
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


# resume parsing
# ... (all your extract functions: extract_contact_number_from_resume, extract_email_from_resume, extract_skills_from_resume, extract_education_from_resume, extract_name_from_resume) ...
# (Keeping your functions as they are)

def extract_contact_number_from_resume(text):
    contact_number = None
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number


def extract_email_from_resume(text):
    email = None
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()
    return email


def extract_skills_from_resume(text):
    skills_list = [
        'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)'
    ]  # <--- END OF FULL SKILLS LIST

    skills = []
    if not isinstance(text, str):
        return []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills


def extract_education_from_resume(text):
    education_list = []
    education_block_match = re.search(r"(?i)\bEDUCATION\b(.*?)\b(EXPERIENCE|PROJECTS|TECHNICAL SKILLS)\b", text,
                                      re.DOTALL)

    if not education_block_match:
        return []

    edu_text = education_block_match.group(1)
    edu_text = re.sub(r'\s+', ' ', edu_text).strip()

    date_pattern = r"(?i)((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s\d{4}(?:\s*–\s*Present)?|\bMarch\s\d{4}\b|\bMay\s\d{4}\b)"

    parts = re.split(date_pattern, edu_text)

    if len(parts) <= 1:
        return [edu_text]

    education_list = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i] and parts[i + 1]:
            entry = parts[i].strip() + " " + parts[i + 1].strip()
            if len(entry) > 20:
                education_list.append(entry)

    if not education_list:
        return [edu_text]

    return education_list


def extract_name_from_resume(text):
    name = None
    pattern = r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b|(\b[A-Z]+(?:\s[A-Z]+){1,2})\b"
    match = re.search(pattern, text)
    if match:
        name = match.group(1) if match.group(1) else match.group(2)
    return name


# routes===============================================

@app.route('/')
def resume():
    # Pass 'None' for category_data on initial load
    return render_template("resume1.html", category_data=None)


# --- THIS IS THE FULLY CORRECTED /PREDICT ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    # Initialize chart data to None
    category_data_for_chart = None

    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        # Check if a file was actually selected
        if filename == '':
            return render_template("resume1.html",
                                   message="No file selected. Please choose a file to upload.",
                                   category_data=category_data_for_chart)

        text = ''  # Initialize text variable

        # --- THIS IS THE MISSING FILE-READING LOGIC ---
        if filename.endswith('.pdf'):
            try:
                text = pdf_to_text(file)
            except Exception as e:
                print(f"Error reading PDF: {e}")
                return render_template('resume1.html', message=f"Error processing PDF file: {e}",
                                       category_data=category_data_for_chart)
        elif filename.endswith('.txt'):
            try:
                # Ensure correct decoding
                text = file.read().decode('utf-8')
            except Exception as e:
                print(f"Error reading TXT: {e}")
                return render_template('resume1.html', message=f"Error processing TXT file: {e}",
                                       category_data=category_data_for_chart)
        else:
            return render_template('resume1.html',
                                   message="Invalid file format. Please upload a PDF or TXT file.",
                                   category_data=category_data_for_chart)
        # --- END OF FILE-READING LOGIC ---

        # Graceful check if models failed to load
        if not rf_classifier_categorization:
            return render_template('resume1.html',
                                   message="Server Error: Model files are not loaded. Please check server logs.",
                                   category_data=category_data_for_chart)

        # Now the 'text' variable is defined, and we can proceed.

        # --- 1. Clean and vectorize text (This is where your error was) ---
        clean_text = cleanResume(text)
        text_features_cat = tfidf_vectorizer_categorization.transform([clean_text])

        # --- 2. Get Top Category Prediction ---
        predicted_category = rf_classifier_categorization.predict(text_features_cat)[0]

        # --- 3. NEW: Get Probabilities for the Chart ---
        category_probabilities = rf_classifier_categorization.predict_proba(text_features_cat)[0]
        all_categories = rf_classifier_categorization.classes_

        # Zip categories with their scores, sort descending
        category_scores = sorted(zip(all_categories, category_probabilities), key=lambda x: x[1], reverse=True)

        # Get top 6 for the chart
        top_categories = category_scores[:6]

        # Format data for Chart.js
        category_data_for_chart = {
            'labels': [item[0] for item in top_categories],
            'scores': [item[1] * 100 for item in top_categories]  # Convert to percentage
        }

        # --- 4. Get Other Info (as before) ---
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text)
        email = extract_email_from_resume(text)
        extracted_skills = extract_skills_from_resume(text)
        extracted_education = extract_education_from_resume(text)
        name = extract_name_from_resume(text)

        # --- 5. Render Template with NEW Chart Data ---
        return render_template(
            'resume1.html',
            predicted_category=predicted_category,
            recommended_job=recommended_job,
            phone=phone,
            name=name,
            email=email,
            extracted_skills=extracted_skills,
            extracted_education=extracted_education,
            category_data=category_data_for_chart  # <-- Pass the new data here
        )
    else:
        # Handle no file uploaded
        return render_template("resume1.html", message="No resume file uploaded.", category_data=category_data_for_chart)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=7860)
