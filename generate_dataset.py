import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class BloodDonationDatasetGenerator:
    def __init__(self):
        # Blood types and their frequencies
        self.blood_types_abo = ['O+', 'O-', 'A+', 'A-', 'B+', 'B-', 'AB+', 'AB-']
        self.blood_type_freq = [0.37, 0.07, 0.36, 0.06, 0.08, 0.02, 0.03, 0.01]
        
        # Extended antigen profiles
        self.rh_variants = {
            'C': ['+', '-'],
            'c': ['+', '-'],
            'E': ['+', '-'],
            'e': ['+', '-']
        }
        
        self.kell = ['+', '-']
        self.duffy = ['Fy(a+b-)', 'Fy(a-b+)', 'Fy(a+b+)', 'Fy(a-b-)']
        self.kidd = ['Jk(a+b-)', 'Jk(a-b+)', 'Jk(a+b+)', 'Jk(a-b-)']
        
        # Indian cities
        self.cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
                       'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
        
        # Medications
        self.medications = ['None', 'Isotretinoin', 'Antibiotics', 'Blood Thinners', 
                           'Finasteride', 'Acne medication']
        
        # Travel destinations (malaria risk)
        self.travel_destinations = ['None', 'Africa', 'South America', 'Southeast Asia', 
                                   'Central America']
        
    def generate_donor_id(self):
        return f"D{str(uuid.uuid4())[:8].upper()}"
    
    def generate_patient_id(self):
        return f"P{str(uuid.uuid4())[:8].upper()}"
    
    def generate_extended_phenotype(self):
        """Generate extended antigen profile"""
        phenotype = {
            'Rh_C': np.random.choice(self.rh_variants['C']),
            'Rh_c': np.random.choice(self.rh_variants['c']),
            'Rh_E': np.random.choice(self.rh_variants['E']),
            'Rh_e': np.random.choice(self.rh_variants['e']),
            'Kell': np.random.choice(self.kell, p=[0.09, 0.91]),
            'Duffy': np.random.choice(self.duffy),
            'Kidd': np.random.choice(self.kidd)
        }
        return phenotype
    
    def generate_antibody_profile(self, num_transfusions):
        """Generate antibody profile based on transfusion history"""
        # More transfusions = higher chance of antibodies
        if num_transfusions < 10:
            antibody_prob = 0.05
        elif num_transfusions < 50:
            antibody_prob = 0.20
        else:
            antibody_prob = 0.35
        
        has_antibodies = np.random.random() < antibody_prob
        
        if has_antibodies:
            possible_antibodies = ['anti-E', 'anti-C', 'anti-Kell', 'anti-c', 
                                  'anti-Fya', 'anti-Jka', 'anti-D']
            num_antibodies = np.random.randint(1, 4)
            antibodies = np.random.choice(possible_antibodies, num_antibodies, replace=False)
            return ', '.join(antibodies)
        return 'None'
    
    def generate_donor_dataset(self, num_donors=1000):
        """Generate donor dataset"""
        donors = []
        
        for i in range(num_donors):
            # Basic demographics
            donor_id = self.generate_donor_id()
            age = np.random.randint(18, 66)
            gender = np.random.choice(['Male', 'Female'], p=[0.6, 0.4])
            weight = np.random.normal(70 if gender == 'Male' else 60, 10)
            weight = max(50, min(weight, 120))  # Clamp between 50-120 kg
            
            # Contact info
            name = f"Donor_{i+1}"
            location = np.random.choice(self.cities)
            contact = f"+91{random.randint(7000000000, 9999999999)}"
            
            # Blood type
            blood_type = np.random.choice(self.blood_types_abo, p=self.blood_type_freq)
            phenotype = self.generate_extended_phenotype()
            
            # Physiological parameters
            hemoglobin = np.random.normal(14.5 if gender == 'Male' else 13.0, 1.0)
            hemoglobin = max(12.0, min(hemoglobin, 18.0))
            blood_pressure = f"{np.random.randint(110, 140)}/{np.random.randint(70, 90)}"
            pulse = np.random.randint(60, 100)
            temperature = np.random.normal(98.6, 0.5)
            
            # Eligibility factors
            last_donation_days = np.random.choice([0, 60, 120, 180, 365], 
                                                  p=[0.3, 0.3, 0.2, 0.1, 0.1])
            last_donation_date = (datetime.now() - timedelta(days=int(last_donation_days))).strftime('%Y-%m-%d')
            
            # Absolute deferral factors
            hiv_status = np.random.choice(['Negative', 'Positive'], p=[0.998, 0.002])
            hepatitis_b = np.random.choice(['Negative', 'Positive'], p=[0.995, 0.005])
            hepatitis_c = np.random.choice(['Negative', 'Positive'], p=[0.997, 0.003])
            iv_drug_use = np.random.choice(['No', 'Yes'], p=[0.99, 0.01])
            
            # Temporary deferral factors
            recent_tattoo = np.random.choice(['No', 'Yes'], p=[0.85, 0.15])
            if recent_tattoo == 'Yes':
                tattoo_days_ago = np.random.randint(1, 120)
                tattoo_date = (datetime.now() - timedelta(days=int(tattoo_days_ago))).strftime('%Y-%m-%d')
            else:
                tattoo_date = 'None'
            
            recent_travel = np.random.choice(['No', 'Yes'], p=[0.75, 0.25])
            if recent_travel == 'Yes':
                travel_destination = np.random.choice(self.travel_destinations[1:])
                travel_days_ago = np.random.randint(1, 365)
                travel_return_date = (datetime.now() - timedelta(days=int(travel_days_ago))).strftime('%Y-%m-%d')
            else:
                travel_destination = 'None'
                travel_return_date = 'None'
            
            current_medication = np.random.choice(self.medications, p=[0.7, 0.05, 0.1, 0.05, 0.05, 0.05])
            
            recent_surgery = np.random.choice(['No', 'Yes'], p=[0.90, 0.10])
            if recent_surgery == 'Yes':
                surgery_days_ago = np.random.randint(1, 180)
                surgery_date = (datetime.now() - timedelta(days=int(surgery_days_ago))).strftime('%Y-%m-%d')
            else:
                surgery_date = 'None'
            
            current_illness = np.random.choice(['No', 'Yes'], p=[0.85, 0.15])
            illness_type = np.random.choice(['None', 'Cold', 'Flu', 'Fever', 'Infection'], 
                                           p=[0.85, 0.05, 0.04, 0.03, 0.03])
            
            # For females
            if gender == 'Female':
                is_pregnant = np.random.choice(['No', 'Yes'], p=[0.95, 0.05])
                is_breastfeeding = np.random.choice(['No', 'Yes'], p=[0.90, 0.10])
            else:
                is_pregnant = 'N/A'
                is_breastfeeding = 'N/A'
            
            # Donation history
            total_donations = np.random.randint(0, 50)
            
            # Calculate eligibility
            is_eligible = self._calculate_donor_eligibility(
                age, weight, hemoglobin, gender, hiv_status, hepatitis_b, 
                hepatitis_c, iv_drug_use, last_donation_days, recent_tattoo,
                tattoo_days_ago if recent_tattoo == 'Yes' else 999,
                recent_travel, travel_days_ago if recent_travel == 'Yes' else 999,
                current_medication, recent_surgery, 
                surgery_days_ago if recent_surgery == 'Yes' else 999,
                current_illness, is_pregnant
            )
            
            # Calculate next eligible date
            next_eligible_date = self._calculate_next_eligible_date(
                last_donation_days, recent_tattoo, tattoo_days_ago if recent_tattoo == 'Yes' else 999,
                recent_travel, travel_days_ago if recent_travel == 'Yes' else 999,
                recent_surgery, surgery_days_ago if recent_surgery == 'Yes' else 999,
                current_medication, current_illness
            )
            
            donor = {
                'Donor_ID': donor_id,
                'Name': name,
                'Age': age,
                'Gender': gender,
                'Weight_kg': round(weight, 1),
                'Contact': contact,
                'Location': location,
                'Blood_Type_ABO_Rh': blood_type,
                'Rh_C': phenotype['Rh_C'],
                'Rh_c': phenotype['Rh_c'],
                'Rh_E': phenotype['Rh_E'],
                'Rh_e': phenotype['Rh_e'],
                'Kell': phenotype['Kell'],
                'Duffy': phenotype['Duffy'],
                'Kidd': phenotype['Kidd'],
                'Hemoglobin_g_dL': round(hemoglobin, 1),
                'Blood_Pressure': blood_pressure,
                'Pulse_bpm': pulse,
                'Temperature_F': round(temperature, 1),
                'Last_Donation_Date': last_donation_date,
                'Days_Since_Last_Donation': last_donation_days,
                'Total_Donations': total_donations,
                'HIV_Status': hiv_status,
                'Hepatitis_B_Status': hepatitis_b,
                'Hepatitis_C_Status': hepatitis_c,
                'IV_Drug_Use_History': iv_drug_use,
                'Recent_Tattoo_Piercing': recent_tattoo,
                'Tattoo_Date': tattoo_date,
                'Recent_Travel': recent_travel,
                'Travel_Destination': travel_destination,
                'Travel_Return_Date': travel_return_date,
                'Current_Medication': current_medication,
                'Recent_Surgery': recent_surgery,
                'Surgery_Date': surgery_date,
                'Current_Illness': current_illness,
                'Illness_Type': illness_type,
                'Is_Pregnant': is_pregnant,
                'Is_Breastfeeding': is_breastfeeding,
                'Currently_Eligible': is_eligible,
                'Next_Eligible_Date': next_eligible_date,
                'Registration_Date': (datetime.now() - timedelta(days=int(np.random.randint(1, 1000)))).strftime('%Y-%m-%d')
            }
            
            donors.append(donor)
        
        return pd.DataFrame(donors)
    
    def generate_recipient_dataset(self, num_recipients=500):
        """Generate recipient (patient) dataset"""
        recipients = []
        
        for i in range(num_recipients):
            patient_id = self.generate_patient_id()
            age = np.random.randint(1, 70)  # Thalassemia patients can be of any age
            gender = np.random.choice(['Male', 'Female'])
            weight = np.random.normal(45 if age < 18 else 65, 15)
            weight = max(15, min(weight, 100))
            
            name = f"Patient_{i+1}"
            location = np.random.choice(self.cities)
            contact = f"+91{random.randint(7000000000, 9999999999)}"
            
            # Blood type
            blood_type = np.random.choice(self.blood_types_abo, p=self.blood_type_freq)
            phenotype = self.generate_extended_phenotype()
            
            # Clinical data
            diagnosis = np.random.choice(['Thalassemia Major', 'Thalassemia Intermedia', 
                                         'Sickle Cell Disease', 'Aplastic Anemia'],
                                        p=[0.5, 0.2, 0.2, 0.1])
            
            # Transfusion history
            total_transfusions = np.random.randint(5, 200)
            last_transfusion_days = np.random.randint(7, 45)
            last_transfusion_date = (datetime.now() - timedelta(days=int(last_transfusion_days))).strftime('%Y-%m-%d')
            
            # Transfusion frequency (days)
            transfusion_frequency = np.random.choice([14, 21, 28, 35], p=[0.3, 0.4, 0.2, 0.1])
            
            # Next scheduled transfusion
            next_transfusion_days = transfusion_frequency - last_transfusion_days
            if next_transfusion_days < 0:
                next_transfusion_days = 7  # Overdue, schedule soon
            next_transfusion_date = (datetime.now() + timedelta(days=int(next_transfusion_days))).strftime('%Y-%m-%d')
            
            # Antibody profile
            antibodies = self.generate_antibody_profile(total_transfusions)
            
            # Clinical parameters
            hemoglobin_target = np.random.uniform(9.5, 11.5)
            current_hemoglobin = np.random.uniform(7.0, hemoglobin_target)
            ferritin_level = np.random.randint(500, 5000)  # ng/mL, elevated due to transfusions
            
            # Hospital data
            treating_hospital = f"{np.random.choice(self.cities)} Thalassemia Center"
            
            # Blood requirement
            units_per_transfusion = np.random.randint(1, 4)
            
            # Urgency
            urgency = 'High' if next_transfusion_days <= 3 else 'Medium' if next_transfusion_days <= 7 else 'Normal'
            
            # Special requirements
            needs_irradiated = np.random.choice(['No', 'Yes'], p=[0.85, 0.15])
            needs_washed = np.random.choice(['No', 'Yes'], p=[0.90, 0.10])
            needs_leukoreduced = np.random.choice(['No', 'Yes'], p=[0.70, 0.30])
            
            recipient = {
                'Patient_ID': patient_id,
                'Name': name,
                'Age': age,
                'Gender': gender,
                'Weight_kg': round(weight, 1),
                'Contact': contact,
                'Location': location,
                'Blood_Type_ABO_Rh': blood_type,
                'Rh_C': phenotype['Rh_C'],
                'Rh_c': phenotype['Rh_c'],
                'Rh_E': phenotype['Rh_E'],
                'Rh_e': phenotype['Rh_e'],
                'Kell': phenotype['Kell'],
                'Duffy': phenotype['Duffy'],
                'Kidd': phenotype['Kidd'],
                'Diagnosis': diagnosis,
                'Total_Transfusions': total_transfusions,
                'Last_Transfusion_Date': last_transfusion_date,
                'Days_Since_Last_Transfusion': last_transfusion_days,
                'Transfusion_Frequency_Days': transfusion_frequency,
                'Next_Scheduled_Transfusion': next_transfusion_date,
                'Days_Until_Next_Transfusion': next_transfusion_days,
                'Antibody_Screen': 'Positive' if antibodies != 'None' else 'Negative',
                'Antibodies_Identified': antibodies,
                'Hemoglobin_Target_g_dL': round(hemoglobin_target, 1),
                'Current_Hemoglobin_g_dL': round(current_hemoglobin, 1),
                'Ferritin_Level_ng_mL': ferritin_level,
                'Treating_Hospital': treating_hospital,
                'Units_Required_Per_Transfusion': units_per_transfusion,
                'Urgency_Level': urgency,
                'Needs_Irradiated_Blood': needs_irradiated,
                'Needs_Washed_RBCs': needs_washed,
                'Needs_Leukoreduced': needs_leukoreduced,
                'Registration_Date': (datetime.now() - timedelta(days=int(np.random.randint(365, 3650)))).strftime('%Y-%m-%d')
            }
            
            recipients.append(recipient)
        
        return pd.DataFrame(recipients)
    
    def _calculate_donor_eligibility(self, age, weight, hemoglobin, gender, hiv, hep_b, 
                                     hep_c, iv_drug, last_donation_days, recent_tattoo,
                                     tattoo_days_ago, recent_travel, travel_days_ago,
                                     medication, recent_surgery, surgery_days_ago,
                                     illness, is_pregnant):
        """Calculate if donor is currently eligible"""
        
        # Absolute deferrals
        if hiv == 'Positive' or hep_b == 'Positive' or hep_c == 'Positive' or iv_drug == 'Yes':
            return 'Permanently Deferred'
        
        # Age and weight
        if age < 18 or age > 65 or weight < 50:
            return 'No'
        
        # Hemoglobin
        min_hb = 13.0 if gender == 'Male' else 12.5
        if hemoglobin < min_hb:
            return 'No - Low Hemoglobin'
        
        # Donation interval
        if last_donation_days < 56:
            return 'No - Too Soon'
        
        # Tattoo
        if recent_tattoo == 'Yes' and tattoo_days_ago < 120:
            return 'No - Recent Tattoo'
        
        # Travel
        if recent_travel == 'Yes' and travel_days_ago < 90:
            return 'No - Recent Travel'
        
        # Surgery
        if recent_surgery == 'Yes' and surgery_days_ago < 180:
            return 'No - Recent Surgery'
        
        # Medication
        if medication in ['Isotretinoin', 'Blood Thinners', 'Antibiotics']:
            return 'No - Medication'
        
        # Illness
        if illness == 'Yes':
            return 'No - Current Illness'
        
        # Pregnancy
        if is_pregnant == 'Yes':
            return 'No - Pregnant'
        
        return 'Yes'
    
    def _calculate_next_eligible_date(self, last_donation_days, recent_tattoo, tattoo_days_ago,
                                      recent_travel, travel_days_ago, recent_surgery, 
                                      surgery_days_ago, medication, illness):
        """Calculate next eligible date for donation"""
        
        deferral_days = []
        
        # Donation interval
        if last_donation_days < 56:
            deferral_days.append(56 - last_donation_days)
        
        # Tattoo
        if recent_tattoo == 'Yes' and tattoo_days_ago < 120:
            deferral_days.append(120 - tattoo_days_ago)
        
        # Travel
        if recent_travel == 'Yes' and travel_days_ago < 90:
            deferral_days.append(90 - travel_days_ago)
        
        # Surgery
        if recent_surgery == 'Yes' and surgery_days_ago < 180:
            deferral_days.append(180 - surgery_days_ago)
        
        # Medication (assume 30 days)
        if medication in ['Isotretinoin', 'Antibiotics']:
            deferral_days.append(30)
        
        # Illness (assume 7 days)
        if illness == 'Yes':
            deferral_days.append(7)
        
        if deferral_days:
            days_until_eligible = max(deferral_days)
            return (datetime.now() + timedelta(days=int(days_until_eligible))).strftime('%Y-%m-%d')
        
        return 'Currently Eligible'


# Generate datasets
generator = BloodDonationDatasetGenerator()

print("Generating Donor Dataset...")
donor_df = generator.generate_donor_dataset(num_donors=1000)

print("Generating Recipient Dataset...")
recipient_df = generator.generate_recipient_dataset(num_recipients=500)

# Save to CSV
donor_df.to_csv('blood_donor_dataset.csv', index=False)
recipient_df.to_csv('blood_recipient_dataset.csv', index=False)

print(f"\n✓ Donor Dataset: {len(donor_df)} records")
print(f"✓ Recipient Dataset: {len(recipient_df)} records")
print("\nFiles saved:")
print("  - blood_donor_dataset.csv")
print("  - blood_recipient_dataset.csv")

# Display summary statistics
print("\n" + "="*60)
print("DONOR DATASET SUMMARY")
print("="*60)
print(f"\nEligibility Distribution:")
print(donor_df['Currently_Eligible'].value_counts())

print(f"\nBlood Type Distribution:")
print(donor_df['Blood_Type_ABO_Rh'].value_counts().sort_index())

print("\n" + "="*60)
print("RECIPIENT DATASET SUMMARY")
print("="*60)
print(f"\nDiagnosis Distribution:")
print(recipient_df['Diagnosis'].value_counts())

print(f"\nUrgency Level Distribution:")
print(recipient_df['Urgency_Level'].value_counts())

print(f"\nAntibody Screen Distribution:")
print(recipient_df['Antibody_Screen'].value_counts())

# Display sample records
print("\n" + "="*60)
print("SAMPLE DONOR RECORD")
print("="*60)
print(donor_df.head(1).T)

print("\n" + "="*60)
print("SAMPLE RECIPIENT RECORD")
print("="*60)
print(recipient_df.head(1).T)
