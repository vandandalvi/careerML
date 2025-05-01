import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Sample dataset - in production you'd use a much larger dataset
def generate_sample_data(num_samples=1000):
    np.random.seed(42)
    data = {
        'q1': np.random.randint(0, 5, num_samples),
        'q2': np.random.randint(0, 5, num_samples),
        'q3': np.random.randint(0, 5, num_samples),
        'q4': np.random.randint(0, 5, num_samples),
        'q5': np.random.randint(0, 5, num_samples),
        'q6': np.random.randint(0, 5, num_samples),
        'q7': np.random.randint(0, 5, num_samples),
        'q8': np.random.randint(0, 5, num_samples),
        'q9': np.random.randint(0, 5, num_samples),
        'q10': np.random.randint(0, 5, num_samples),
    }
    
    # Create some patterns in the data
    domains = []
    for i in range(num_samples):
        if data['q1'][i] == 0 and data['q2'][i] == 0:
            domains.append('aiml')
        elif data['q3'][i] == 1 and data['q4'][i] == 1:
            domains.append('devops')
        elif data['q5'][i] == 2 and data['q6'][i] == 2:
            domains.append('web')
        elif data['q7'][i] == 3 and data['q8'][i] == 3:
            domains.append('cyber')
        elif data['q9'][i] == 4 and data['q10'][i] == 4:
            domains.append('data')
        else:
            domains.append(np.random.choice(['aiml', 'devops', 'web', 'cyber', 'data']))
    
    data['domain'] = domains
    return pd.DataFrame(data)

# Generate and save the model
def train_and_save_model():
    df = generate_sample_data()
    X = df.drop('domain', axis=1)
    y = df['domain']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'model/career_model.pkl')
    print(f"Model trained with accuracy: {model.score(X_test, y_test):.2f}")

if __name__ == '__main__':
    train_and_save_model()