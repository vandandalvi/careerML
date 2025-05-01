from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import joblib
import numpy as np
import random

app = Flask(__name__)
app.secret_key = "career_assessment_secret_key"

# Load the trained model
try:
    model = joblib.load('model/career_model.pkl')
    print("Model loaded successfully!")
    model_available = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_available = False

# Domain information
DOMAINS = {
    'aiml': {
        'name': 'Artificial Intelligence & Machine Learning',
        'description': 'Focuses on creating systems that can learn and make intelligent decisions.',
        'roles': [
            {'title': 'Machine Learning Engineer', 'description': 'Design and implement ML models for production systems'},
            {'title': 'AI Research Scientist', 'description': 'Conduct cutting-edge research in artificial intelligence'},
            {'title': 'Computer Vision Engineer', 'description': 'Develop algorithms to interpret visual information'},
            {'title': 'NLP Specialist', 'description': 'Work on language processing and understanding systems'}
        ]
    },
    'devops': {
        'name': 'DevOps',
        'description': 'Combines software development and IT operations to shorten the development lifecycle.',
        'roles': [
            {'title': 'DevOps Engineer', 'description': 'Bridge development and operations with automation'},
            {'title': 'Site Reliability Engineer', 'description': 'Ensure system reliability and performance'},
            {'title': 'Cloud Architect', 'description': 'Design and manage cloud infrastructure solutions'},
            {'title': 'Release Manager', 'description': 'Coordinate software releases and deployments'}
        ]
    },
    'web': {
        'name': 'Web Development',
        'description': 'Involves building and maintaining websites and web applications.',
        'roles': [
            {'title': 'Frontend Developer', 'description': 'Build user interfaces and client-side logic'},
            {'title': 'Backend Developer', 'description': 'Develop server-side applications and APIs'},
            {'title': 'Full Stack Developer', 'description': 'Handle both frontend and backend development'},
            {'title': 'UI/UX Designer', 'description': 'Create user-centered designs and experiences'}
        ]
    },
    'cyber': {
        'name': 'Cybersecurity',
        'description': 'Focuses on protecting systems and networks from digital attacks.',
        'roles': [
            {'title': 'Security Analyst', 'description': 'Monitor and protect systems from threats'},
            {'title': 'Penetration Tester', 'description': 'Ethically hack systems to find vulnerabilities'},
            {'title': 'Security Architect', 'description': 'Design secure systems and networks'},
            {'title': 'Incident Responder', 'description': 'Investigate and mitigate security breaches'}
        ]
    },
    'data': {
        'name': 'Data Science',
        'description': 'Involves extracting knowledge and insights from structured and unstructured data.',
        'roles': [
            {'title': 'Data Scientist', 'description': 'Extract insights from complex datasets'},
            {'title': 'Data Analyst', 'description': 'Analyze data to inform business decisions'},
            {'title': 'Data Engineer', 'description': 'Build and maintain data pipelines'},
            {'title': 'Business Intelligence Analyst', 'description': 'Create reports and dashboards for stakeholders'}
        ]
    }
}

# Scenario-based questions for each domain
SCENARIO_QUESTIONS = {
    "aiml": [
        {
            "question": "You've been tasked with developing a machine learning model to predict customer churn for a telecom company. You notice the dataset is imbalanced with only 5% of customers classified as churned. What approach would you take?",
            "options": [
                "Undersample the majority class",
                "Oversample the minority class using techniques like SMOTE",
                "Use weighted loss functions",
                "All of the above could be valid approaches"
            ],
            "correct": 3
        },
         {
        "question": "Which of the following activation functions is most commonly used in deep learning for hidden layers due to its non-linearity and ability to avoid vanishing gradients?",
        "options": [
            "Sigmoid",
            "Tanh",
            "ReLU",
            "Softmax"
        ],
        "correct": 2
    },
    {
        "question": "What is the primary purpose of cross-validation in machine learning?",
        "options": [
            "To increase model complexity",
            "To reduce the bias of the model",
            "To assess how well a model generalizes to unseen data",
            "To optimize the loss function"
        ],
        "correct": 2
    },
    {
        "question": "Which technique is commonly used to reduce overfitting in deep learning models?",
        "options": [
            "Using deeper networks",
            "Applying dropout regularization",
            "Increasing the batch size",
            "Reducing the number of hidden layers"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following is a key advantage of using decision trees in machine learning?",
        "options": [
            "They are highly prone to overfitting",
            "They require large amounts of labeled data",
            "They are easy to interpret and visualize",
            "They perform well on high-dimensional data"
        ],
        "correct": 2
    },
    {
        "question": "In unsupervised learning, which algorithm is commonly used for clustering?",
        "options": [
            "Linear Regression",
            "K-Means",
            "Support Vector Machines",
            "Naive Bayes"
        ],
        "correct": 1
    },
    {
        "question": "Which type of neural network is best suited for analyzing sequential data such as time series or natural language processing?",
        "options": [
            "Convolutional Neural Networks (CNNs)",
            "Recurrent Neural Networks (RNNs)",
            "Feedforward Neural Networks",
            "Generative Adversarial Networks (GANs)"
        ],
        "correct": 1
    },
    {
        "question": "What is the primary goal of principal component analysis (PCA) in machine learning?",
        "options": [
            "To reduce the dimensionality of the dataset while preserving variance",
            "To cluster data points into distinct groups",
            "To maximize the accuracy of classification models",
            "To convert categorical variables into numerical ones"
        ],
        "correct": 0
    },
    {
        "question": "Which evaluation metric is most appropriate for a classification problem with highly imbalanced classes?",
        "options": [
            "Accuracy",
            "Precision and Recall",
            "Mean Squared Error",
            "R-Squared"
        ],
        "correct": 1
    },
    {
        "question": "Which machine learning technique involves training multiple models and combining their predictions to improve performance?",
        "options": [
            "Regularization",
            "Gradient Descent",
            "Ensemble Learning",
            "Feature Selection"
        ],
        "correct": 2
    }
        # Additional questions would be added here
    ],
    "devops": [
        {
            "question": "Your team has deployed a new feature to production, but users are reporting errors. The application logs show increased memory usage. What would be your first steps?",
            "options": [
                "Roll back the deployment immediately",
                "Add more server resources to handle the increased memory usage",
                "Analyze logs to identify the specific component causing the issue",
                "Restart the application servers"
            ],
            "correct": 2
        },
        {
        "question": "Which of the following is a key benefit of using Infrastructure as Code (IaC) in DevOps?",
        "options": [
            "Manual configuration of infrastructure",
            "Faster and more reliable provisioning",
            "Increased dependency on cloud vendors",
            "Improved physical server management"
        ],
        "correct": 1
    },
    {
        "question": "What is the primary purpose of a CI/CD pipeline in DevOps?",
        "options": [
            "To automate the process of software development and deployment",
            "To replace manual testing with automated scripts",
            "To eliminate the need for version control systems",
            "To restrict developer access to production environments"
        ],
        "correct": 0
    },
    {
        "question": "Which tool is commonly used for container orchestration in production environments?",
        "options": [
            "Docker",
            "Kubernetes",
            "Jenkins",
            "Terraform"
        ],
        "correct": 1
    },
    {
        "question": "What is the purpose of using Blue-Green Deployment in DevOps?",
        "options": [
            "To maintain two separate environments and switch traffic between them for seamless updates",
            "To replace all infrastructure components during each deployment",
            "To reduce storage costs by limiting backups",
            "To automatically revert changes after a failed deployment"
        ],
        "correct": 0
    },
    {
        "question": "Which of the following DevOps practices helps detect security vulnerabilities early in the development cycle?",
        "options": [
            "Chaos Engineering",
            "Shift-left Security",
            "Load Balancing",
            "Auto Scaling"
        ],
        "correct": 1
    },
    {
        "question": "Which logging and monitoring tool is commonly used to collect and analyze logs in a DevOps environment?",
        "options": [
            "Grafana",
            "ELK Stack (Elasticsearch, Logstash, Kibana)",
            "Jenkins",
            "Ansible"
        ],
        "correct": 1
    },
    {
        "question": "What is the main advantage of using containerization in DevOps?",
        "options": [
            "Increased reliance on physical machines",
            "Consistent runtime environments across different systems",
            "Reduced need for automated testing",
            "Elimination of security vulnerabilities"
        ],
        "correct": 1
    },
    {
        "question": "Which DevOps methodology promotes frequent, small, and incremental changes to the codebase?",
        "options": [
            "Waterfall Model",
            "Agile Development",
            "Monolithic Development",
            "Big Bang Deployment"
        ],
        "correct": 1
    },
    {
        "question": "What is the main purpose of using Ansible in DevOps?",
        "options": [
            "Continuous Integration",
            "Infrastructure automation and configuration management",
            "Monitoring application performance",
            "Managing container orchestration"
        ],
        "correct": 1
    }
        # Additional questions would be added here
    ],
    "web": [
        {
            "question": "You've developed a web application that loads slowly for users. Initial investigation shows large JavaScript bundle sizes. How would you approach improving performance?",
            "options": [
                "Implement code splitting to load only necessary JavaScript",
                "Minify all JavaScript files",
                "Implement lazy loading for components not needed on initial render",
                "All of the above"
            ],
            "correct": 3
        },
        {
        "question": "Which HTTP status code indicates that a resource has been permanently moved to a new URL?",
        "options": [
            "301",
            "404",
            "500",
            "200"
        ],
        "correct": 0
    },
    {
        "question": "What is the primary purpose of using a Content Delivery Network (CDN) in web development?",
        "options": [
            "To store user data securely",
            "To speed up the delivery of web assets by caching content in multiple locations",
            "To prevent SQL injection attacks",
            "To create server-side logic for a web application"
        ],
        "correct": 1
    },
    {
        "question": "Which CSS property is used to create a responsive layout by distributing space between items?",
        "options": [
            "display: block",
            "flexbox (display: flex)",
            "position: absolute",
            "grid-template-columns"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following is NOT a JavaScript framework or library?",
        "options": [
            "React",
            "Vue",
            "Django",
            "Angular"
        ],
        "correct": 2
    },
    {
        "question": "What is the main advantage of using Server-Side Rendering (SSR) in web applications?",
        "options": [
            "It reduces server load by rendering all content on the client-side",
            "It improves SEO and initial page load time",
            "It removes the need for a backend server",
            "It makes JavaScript completely unnecessary"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following techniques can help prevent Cross-Site Scripting (XSS) attacks?",
        "options": [
            "Validating and sanitizing user input",
            "Using inline JavaScript within HTML files",
            "Disabling HTTPS on the server",
            "Using large image files"
        ],
        "correct": 0
    },
    {
        "question": "Which of the following best describes Progressive Web Applications (PWAs)?",
        "options": [
            "Web applications that work offline, have fast load times, and provide an app-like experience",
            "Web applications that require an internet connection at all times",
            "Web applications that only run on mobile devices",
            "Web applications that do not support push notifications"
        ],
        "correct": 0
    },
    {
        "question": "Which JavaScript feature allows asynchronous execution without blocking the main thread?",
        "options": [
            "Promises and async/await",
            "Synchronous loops",
            "Callback functions only",
            "document.getElementById"
        ],
        "correct": 0
    },
    {
        "question": "Which of the following HTTP methods is used to update an existing resource?",
        "options": [
            "GET",
            "POST",
            "PUT",
            "DELETE"
        ],
        "correct": 2
    }
        # Additional questions would be added here
    ],
    "cyber": [
        {
            "question": "Your organization's web application was found to be vulnerable to SQL injection attacks. What immediate action would you recommend?",
            "options": [
                "Take the application offline until it can be fixed",
                "Implement input validation and parameterized queries",
                "Switch to a NoSQL database",
                "Add a web application firewall"
            ],
            "correct": 1
        },
        {
        "question": "Which type of malware encrypts a user's files and demands payment for decryption?",
        "options": [
            "Trojan Horse",
            "Ransomware",
            "Spyware",
            "Adware"
        ],
        "correct": 1
    },
    {
        "question": "What is the primary purpose of a firewall in network security?",
        "options": [
            "To physically block unauthorized access to a network",
            "To filter incoming and outgoing traffic based on security rules",
            "To store encrypted user credentials",
            "To detect and remove viruses from a computer"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following is an example of a social engineering attack?",
        "options": [
            "Brute-force attack",
            "Phishing email",
            "SQL Injection",
            "Man-in-the-Middle attack"
        ],
        "correct": 1
    },
    {
        "question": "What does 'zero-day vulnerability' mean in cybersecurity?",
        "options": [
            "A vulnerability that has been patched immediately after discovery",
            "A vulnerability that has been exploited before a fix is available",
            "A security feature that protects against cyber threats",
            "A type of virus that spreads through email attachments"
        ],
        "correct": 1
    },
    {
        "question": "Which security protocol is commonly used to encrypt web traffic?",
        "options": [
            "FTP",
            "HTTP",
            "TLS/SSL",
            "ICMP"
        ],
        "correct": 2
    },
    {
        "question": "What is the purpose of multi-factor authentication (MFA)?",
        "options": [
            "To allow users to reset their passwords more easily",
            "To require multiple forms of verification for account access",
            "To replace the need for strong passwords",
            "To store passwords securely in a database"
        ],
        "correct": 1
    },
    {
        "question": "Which tool is commonly used for penetration testing and ethical hacking?",
        "options": [
            "Nmap",
            "Microsoft Excel",
            "Google Chrome",
            "VMware"
        ],
        "correct": 0
    },
    {
        "question": "What does a VPN (Virtual Private Network) do in terms of security?",
        "options": [
            "Provides faster internet speeds",
            "Hides a user's IP address and encrypts internet traffic",
            "Prevents all types of cyber attacks",
            "Blocks malicious emails"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following best describes a brute-force attack?",
        "options": [
            "An attack that tries every possible password combination until the correct one is found",
            "An attack that infects computers with malware",
            "An attack that tricks users into giving up personal information",
            "An attack that intercepts data in transit"
        ],
        "correct": 0
    }
        # Additional questions would be added here
    ],
    
    "data": [
        {
            "question": "You're analyzing customer purchase data and notice that the distribution of purchase amounts is highly skewed with many outliers. What approach would you take for a meaningful analysis?",
            "options": [
                "Remove all outliers from the dataset",
                "Use median and IQR instead of mean and standard deviation",
                "Apply a log transformation to normalize the distribution",
                "Both B and C could be appropriate"
            ],
            "correct": 3
        },
        {
        "question": "Which of the following is an example of a supervised learning algorithm?",
        "options": [
            "K-Means Clustering",
            "Principal Component Analysis (PCA)",
            "Linear Regression",
            "Apriori Algorithm"
        ],
        "correct": 2
    },
    {
        "question": "In data preprocessing, what is the main purpose of one-hot encoding?",
        "options": [
            "To normalize numerical values",
            "To convert categorical variables into a binary matrix",
            "To remove duplicate data",
            "To detect missing values in a dataset"
        ],
        "correct": 1
    },
    {
        "question": "Which metric is commonly used to evaluate classification models?",
        "options": [
            "Mean Squared Error (MSE)",
            "Accuracy",
            "R-Squared",
            "Silhouette Score"
        ],
        "correct": 1
    },
    {
        "question": "Which SQL clause is used to filter records based on a specified condition?",
        "options": [
            "ORDER BY",
            "WHERE",
            "GROUP BY",
            "HAVING"
        ],
        "correct": 1
    },
    {
        "question": "What is the purpose of dimensionality reduction techniques like PCA?",
        "options": [
            "To increase the number of features in a dataset",
            "To remove irrelevant or redundant features and improve model performance",
            "To create new artificial data points",
            "To replace categorical data with numerical values"
        ],
        "correct": 1
    },
    {
        "question": "Which type of database is best suited for handling large-scale, unstructured data?",
        "options": [
            "Relational Database (RDBMS)",
            "NoSQL Database",
            "Data Warehouse",
            "Flat File Database"
        ],
        "correct": 1
    },
    {
        "question": "What does 'data wrangling' refer to in data analysis?",
        "options": [
            "Creating new data from scratch",
            "Processing and transforming raw data into a usable format",
            "Storing data in cloud-based servers",
            "Deleting unnecessary datasets"
        ],
        "correct": 1
    },
    {
        "question": "Which visualization technique is most effective for showing the distribution of a numerical variable?",
        "options": [
            "Bar Chart",
            "Histogram",
            "Pie Chart",
            "Line Chart"
        ],
        "correct": 1
    },
    {
        "question": "Which of the following best describes an ETL (Extract, Transform, Load) process?",
        "options": [
            "A process used to train machine learning models",
            "A method for securely transmitting data over networks",
            "A pipeline used to extract data, transform it, and load it into a database or data warehouse",
            "A programming language used for data analysis"
        ],
        "correct": 2
    }
        # Additional questions would be added here
    ]
}

@app.route('/')
def index():
    """Render the main assessment page"""
    return render_template('index.html')

@app.route('/assess', methods=['POST'])
def assess():
    """Process the initial assessment and determine domain"""
    try:
        if model_available:
            # Get answers from form
            answers = []
            missing_questions = []
            
            for i in range(1, 11):  # Assuming 10 questions
                answer = request.form.get(f'q{i}')
                if answer is None:
                    missing_questions.append(i)
                else:
                    try:
                        answers.append(int(answer))
                    except ValueError:
                        missing_questions.append(i)
            
            if missing_questions:
                error_msg = f"Please answer questions: {', '.join(map(str, missing_questions))}"
                return render_template('index.html', error=error_msg), 400
            
            # Predict domain using the model
            domain = model.predict([answers])[0]
        else:
            # Fallback to random selection if model isn't available
            domain = random.choice(list(DOMAINS.keys()))
        
        # Store the domain in session
        session['domain'] = domain
        return redirect(url_for('results'))
    
    except Exception as e:
        print(f"Error in assessment: {e}")
        return render_template('index.html', error="An error occurred processing your answers"), 500

@app.route('/results')
def results():
    """Display the results of the domain assessment"""
    domain = session.get('domain', 'data')  # Default to data if no domain in session
    domain_info = DOMAINS.get(domain, DOMAINS['data'])
    
    return render_template(
        'results.html', 
        domain=domain.upper(),
        domain_name=domain_info['name'],
        domain_description=domain_info['description'],
        roles=domain_info['roles']
    )

@app.route('/job_role_assessment')
def job_role_assessment():
    """Present scenario-based questions for the determined domain"""
    domain = session.get('domain', 'data')
    # Get questions for the specific domain
    questions = SCENARIO_QUESTIONS.get(domain, [])
    
    # If no questions available for this domain
    if not questions:
        return redirect(url_for('results'))
        
    return render_template('job_role_assessment.html', 
                          domain=domain.upper(), 
                          domain_name=DOMAINS[domain]['name'],
                          questions=questions)

@app.route('/job_role_results', methods=['POST'])
def job_role_results():
    """Process and display results of the job role assessment"""
    domain = session.get('domain', 'data')
    questions = SCENARIO_QUESTIONS.get(domain, [])
    
    if not questions:
        return redirect(url_for('results'))
    
    correct_count = 0
    total_questions = len(questions)
    
    # Calculate score
    for i, question in enumerate(questions):
        user_answer = int(request.form.get(f'q{i}', -1))
        if user_answer == question['correct']:
            correct_count += 1
    
    score_percentage = (correct_count / total_questions) * 100
    
    # Determine suitability based on score
    if score_percentage >= 80:
        suitability = "Excellent fit"
    elif score_percentage >= 60:
        suitability = "Good fit"
    elif score_percentage >= 40:
        suitability = "Moderate fit"
    else:
        suitability = "Consider exploring other areas"
    
    return render_template(
        'job_role_results.html',
        domain=domain.upper(),
        domain_name=DOMAINS[domain]['name'],
        score=correct_count,
        total=total_questions,
        percentage=score_percentage,
        suitability=suitability,
        roles=DOMAINS[domain]['roles']
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if not model_available:
        return jsonify({"error": "Model not available"}), 503
    
    try:
        data = request.get_json()
        answers = data.get('answers', [])
        
        if len(answers) != 10:  # Assuming 10 questions required
            return jsonify({"error": "Exactly 10 answers required"}), 400
            
        domain = model.predict([answers])[0]
        return jsonify({
            "domain": domain,
            "domain_name": DOMAINS[domain]['name'],
            "roles": DOMAINS[domain]['roles']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.getenv('PORT', 5000))
