import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class BloodDonationMLModel:
    def __init__(self):
        self.donor_model = None
        self.matching_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_datasets(self):
        """Load the donor and recipient datasets"""
        print("Loading datasets...")
        self.donor_df = pd.read_csv('blood_donor_dataset.csv')
        self.recipient_df = pd.read_csv('blood_recipient_dataset.csv')
        print(f"✓ Loaded {len(self.donor_df)} donor records")
        print(f"✓ Loaded {len(self.recipient_df)} recipient records")
        return self.donor_df, self.recipient_df
    
    def preprocess_donor_eligibility(self):
        """Preprocess donor data for eligibility prediction"""
        print("\nPreprocessing donor data for eligibility prediction...")
        
        df = self.donor_df.copy()
        
        # Create binary target variable
        df['Eligible_Binary'] = df['Currently_Eligible'].apply(
            lambda x: 1 if x == 'Yes' else 0
        )
        
        # Select features for eligibility prediction
        feature_cols = [
            'Age', 'Gender', 'Weight_kg', 'Hemoglobin_g_dL', 
            'Days_Since_Last_Donation', 'Total_Donations',
            'HIV_Status', 'Hepatitis_B_Status', 'Hepatitis_C_Status',
            'IV_Drug_Use_History', 'Recent_Tattoo_Piercing', 'Recent_Travel',
            'Current_Medication', 'Recent_Surgery', 'Current_Illness',
            'Is_Pregnant', 'Is_Breastfeeding'
        ]
        
        X = df[feature_cols].copy()
        y = df['Eligible_Binary']
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'HIV_Status', 'Hepatitis_B_Status', 
                           'Hepatitis_C_Status', 'IV_Drug_Use_History',
                           'Recent_Tattoo_Piercing', 'Recent_Travel', 
                           'Current_Medication', 'Recent_Surgery', 
                           'Current_Illness', 'Is_Pregnant', 'Is_Breastfeeding']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[f'donor_{col}'] = le
        
        # Scale numerical features
        numerical_cols = ['Age', 'Weight_kg', 'Hemoglobin_g_dL', 
                         'Days_Since_Last_Donation', 'Total_Donations']
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        print(f"✓ Processed {len(X)} donor records")
        print(f"✓ Features: {len(feature_cols)}")
        print(f"✓ Eligible donors: {y.sum()} ({y.mean()*100:.1f}%)")
        
        return X, y, feature_cols
    
    def train_donor_eligibility_model(self, X, y, feature_cols):
        """Train model to predict donor eligibility"""
        print("\nTraining Donor Eligibility Model...")
        print("-" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        self.donor_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.donor_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.donor_model.predict(X_train)
        y_pred_test = self.donor_model.predict(X_test)
        
        # Evaluation
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\n📊 Training Accuracy: {train_accuracy:.4f}")
        print(f"📊 Testing Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.donor_model, X, y, cv=5, scoring='accuracy')
        print(f"📊 Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Classification report
        print("\n📋 Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['Not Eligible', 'Eligible']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.donor_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Eligible', 'Eligible'],
                   yticklabels=['Not Eligible', 'Eligible'])
        plt.title('Donor Eligibility Prediction - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('donor_eligibility_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'donor_eligibility_confusion_matrix.png'")
        
        # Feature Importance Plot
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Features for Donor Eligibility Prediction')
        plt.tight_layout()
        plt.savefig('donor_feature_importance.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance plot saved as 'donor_feature_importance.png'")
        plt.close('all')
        
        return X_test, y_test, y_pred_test
    
    def create_matching_dataset(self):
        """Create dataset for blood matching"""
        print("\n\nCreating Blood Matching Dataset...")
        print("-" * 60)
        
        # Get eligible donors
        eligible_donors = self.donor_df[
            self.donor_df['Currently_Eligible'] == 'Yes'
        ].copy()
        
        print(f"✓ Found {len(eligible_donors)} eligible donors")
        print(f"✓ Found {len(self.recipient_df)} recipients")
        
        matching_records = []
        
        # Create matching pairs (sample to keep dataset manageable)
        sample_recipients = self.recipient_df.sample(min(200, len(self.recipient_df)), random_state=42)
        
        for idx, recipient in sample_recipients.iterrows():
            # Sample donors for each recipient
            sample_donors = eligible_donors.sample(min(50, len(eligible_donors)), random_state=idx)
            
            for _, donor in sample_donors.iterrows():
                # Calculate compatibility score
                compatibility = self._calculate_compatibility(donor, recipient)
                
                # Create matching record
                record = {
                    'Donor_Blood_Type': donor['Blood_Type_ABO_Rh'],
                    'Recipient_Blood_Type': recipient['Blood_Type_ABO_Rh'],
                    'Donor_Rh_C': donor['Rh_C'],
                    'Recipient_Rh_C': recipient['Rh_C'],
                    'Donor_Rh_E': donor['Rh_E'],
                    'Recipient_Rh_E': recipient['Rh_E'],
                    'Donor_Kell': donor['Kell'],
                    'Recipient_Kell': recipient['Kell'],
                    'Blood_Type_Compatible': self._check_abo_compatibility(
                        donor['Blood_Type_ABO_Rh'], recipient['Blood_Type_ABO_Rh']
                    ),
                    'Extended_Match': compatibility['extended_match'],
                    'Recipient_Has_Antibodies': 1 if recipient['Antibody_Screen'] == 'Positive' else 0,
                    'Recipient_Total_Transfusions': recipient['Total_Transfusions'],
                    'Donor_Location': donor['Location'],
                    'Recipient_Location': recipient['Location'],
                    'Location_Match': 1 if donor['Location'] == recipient['Location'] else 0,
                    'Urgency_High': 1 if recipient['Urgency_Level'] == 'High' else 0,
                    'Compatibility_Score': compatibility['score'],
                    'Match_Quality': compatibility['quality']
                }
                matching_records.append(record)
        
        matching_df = pd.DataFrame(matching_records)
        print(f"\n✓ Created {len(matching_df)} matching records")
        print(f"✓ Match quality distribution:")
        print(matching_df['Match_Quality'].value_counts())
        
        return matching_df
    
    def _check_abo_compatibility(self, donor_type, recipient_type):
        """Check ABO blood type compatibility"""
        compatibility_matrix = {
            'O-': ['O-', 'O+', 'A-', 'A+', 'B-', 'B+', 'AB-', 'AB+'],
            'O+': ['O+', 'A+', 'B+', 'AB+'],
            'A-': ['A-', 'A+', 'AB-', 'AB+'],
            'A+': ['A+', 'AB+'],
            'B-': ['B-', 'B+', 'AB-', 'AB+'],
            'B+': ['B+', 'AB+'],
            'AB-': ['AB-', 'AB+'],
            'AB+': ['AB+']
        }
        
        return 1 if recipient_type in compatibility_matrix.get(donor_type, []) else 0
    
    def _calculate_compatibility(self, donor, recipient):
        """Calculate detailed compatibility between donor and recipient"""
        score = 0
        
        # ABO compatibility (most important)
        abo_compatible = self._check_abo_compatibility(
            donor['Blood_Type_ABO_Rh'], recipient['Blood_Type_ABO_Rh']
        )
        if not abo_compatible:
            return {'score': 0, 'quality': 'Incompatible', 'extended_match': 0}
        
        score += 40  # Base score for ABO compatibility
        
        # Extended antigen matching
        extended_matches = 0
        if donor['Rh_C'] == recipient['Rh_C']:
            score += 10
            extended_matches += 1
        if donor['Rh_E'] == recipient['Rh_E']:
            score += 10
            extended_matches += 1
        if donor['Rh_c'] == recipient['Rh_c']:
            score += 5
            extended_matches += 1
        if donor['Rh_e'] == recipient['Rh_e']:
            score += 5
            extended_matches += 1
        if donor['Kell'] == recipient['Kell']:
            score += 15
            extended_matches += 1
        if donor['Duffy'] == recipient['Duffy']:
            score += 10
            extended_matches += 1
        if donor['Kidd'] == recipient['Kidd']:
            score += 5
            extended_matches += 1
        
        # Location match bonus
        if donor['Location'] == recipient['Location']:
            score += 10
        
        # Determine quality
        if score >= 90:
            quality = 'Excellent'
        elif score >= 70:
            quality = 'Good'
        elif score >= 50:
            quality = 'Fair'
        else:
            quality = 'Poor'
        
        extended_match = 1 if extended_matches >= 5 else 0
        
        return {'score': score, 'quality': quality, 'extended_match': extended_match}
    
    def train_matching_model(self, matching_df):
        """Train model to predict match quality"""
        print("\n\nTraining Blood Matching Model...")
        print("-" * 60)
        
        # Prepare features
        X = matching_df.drop(['Match_Quality', 'Compatibility_Score'], axis=1).copy()
        y = matching_df['Match_Quality']
        
        # Encode categorical variables
        categorical_cols = ['Donor_Blood_Type', 'Recipient_Blood_Type', 
                           'Donor_Rh_C', 'Recipient_Rh_C',
                           'Donor_Rh_E', 'Recipient_Rh_E',
                           'Donor_Kell', 'Recipient_Kell',
                           'Donor_Location', 'Recipient_Location']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[f'matching_{col}'] = le
            else:
                X[col] = self.label_encoders[f'matching_{col}'].transform(X[col].astype(str))
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['match_quality_target'] = le_target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train Gradient Boosting model
        self.matching_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        
        self.matching_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.matching_model.predict(X_train)
        y_pred_test = self.matching_model.predict(X_test)
        
        # Evaluation
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        print(f"\n📊 Training Accuracy: {train_accuracy:.4f}")
        print(f"📊 Testing Accuracy: {test_accuracy:.4f}")
        
        # Cross-validation - skip for now due to large dataset
        print(f"📊 Cross-Validation: Skipped (using train-test split validation)")
        
        # Classification report
        print("\n📋 Classification Report (Test Set):")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=le_target.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=le_target.classes_,
                   yticklabels=le_target.classes_)
        plt.title('Blood Matching Quality Prediction - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('matching_confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\n✓ Confusion matrix saved as 'matching_confusion_matrix.png'")
        plt.close('all')
        
        return X_test, y_test, y_pred_test
    
    def save_models(self):
        """Save trained models and encoders"""
        print("\n\nSaving Models...")
        print("-" * 60)
        
        # Save donor eligibility model
        joblib.dump(self.donor_model, 'donor_eligibility_model.pkl')
        print("✓ Saved donor_eligibility_model.pkl")
        
        # Save matching model
        joblib.dump(self.matching_model, 'blood_matching_model.pkl')
        print("✓ Saved blood_matching_model.pkl")
        
        # Save label encoders
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        print("✓ Saved label_encoders.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        print("✓ Saved scaler.pkl")
        
        print("\n✅ All models saved successfully!")
    
    def load_models(self):
        """Load saved models"""
        self.donor_model = joblib.load('donor_eligibility_model.pkl')
        self.matching_model = joblib.load('blood_matching_model.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')
        self.scaler = joblib.load('scaler.pkl')
        print("✓ Models loaded successfully")
    
    def predict_donor_eligibility(self, donor_data):
        """Predict if a donor is eligible"""
        # Preprocess input
        # ... implementation for single donor prediction
        pass
    
    def find_best_matches(self, recipient_data, n_matches=10):
        """Find best donor matches for a recipient"""
        # Implementation for finding matches
        pass


def main():
    print("="*60)
    print("BLOOD DONATION ML MODEL TRAINING")
    print("="*60)
    
    # Initialize model
    ml_model = BloodDonationMLModel()
    
    # Load datasets
    donor_df, recipient_df = ml_model.load_datasets()
    
    # Train donor eligibility model
    X_donor, y_donor, feature_cols = ml_model.preprocess_donor_eligibility()
    ml_model.train_donor_eligibility_model(X_donor, y_donor, feature_cols)
    
    # Create matching dataset and train matching model
    matching_df = ml_model.create_matching_dataset()
    ml_model.train_matching_model(matching_df)
    
    # Save all models
    ml_model.save_models()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("  - donor_eligibility_model.pkl")
    print("  - blood_matching_model.pkl")
    print("  - label_encoders.pkl")
    print("  - scaler.pkl")
    print("  - donor_eligibility_confusion_matrix.png")
    print("  - donor_feature_importance.png")
    print("  - matching_confusion_matrix.png")
    print("\n🎯 Next Steps:")
    print("  1. Review the confusion matrices and feature importance plots")
    print("  2. Use the saved models for predictions")
    print("  3. Integrate models into your blood donation application")


if __name__ == "__main__":
    main()
