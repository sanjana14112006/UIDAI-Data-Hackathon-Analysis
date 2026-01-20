# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Phase 1: Environment Setup & Data Extraction
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def read_all_from_zip(zip_path):
    all_chunks = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Loop through every file inside the ZIP
        for file_info in z.infolist():
            # Only process files that end in .csv
            if file_info.filename.endswith('.csv'):
                with z.open(file_info) as f:
                    print(f"Reading chunk: {file_info.filename}")
                    all_chunks.append(pd.read_csv(f))
    
    # Combine all chunks into one big table
    return pd.concat(all_chunks, ignore_index=True)

# 1. LOAD ALL DATASETS
path_base = r'C:\Users\sanja\OneDrive\Desktop\UIDAI DATA HACKATHON'

print("--- Loading Enrolments ---")
df_enrol = read_all_from_zip(f"{path_base}\\api_data_aadhar_enrolment.zip")

print("\n--- Loading Demographic Updates ---")
df_demo = read_all_from_zip(f"{path_base}\\api_data_aadhar_demographic.zip")

print("\n--- Loading Biometric Updates ---")
df_bio = read_all_from_zip(f"{path_base}\\api_data_aadhar_biometric.zip")

print("\n✅ All data combined and loaded successfully!")
print(f"Total Enrolment Rows: {len(df_enrol)}")
# Total Enrolment Rows: 1006029

print(df_enrol.head())
#          date          state          district  pincode  age_0_5  age_5_17  age_18_greater
# 0  02-03-2025      Meghalaya  East Khasi Hills   793121       11        61              37
# 1  09-03-2025      Karnataka   Bengaluru Urban   560043       14        33              39
# 2  09-03-2025  Uttar Pradesh      Kanpur Nagar   208001       29        82              12
# 3  09-03-2025  Uttar Pradesh           Aligarh   202133       62        29              15
# 4  09-03-2025      Karnataka   Bengaluru Urban   560016       14        16              21
print(df_enrol.shape)
# (1006029, 7)

print(df_enrol.info())  # data type of cols?
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1006029 entries, 0 to 1006028
# Data columns (total 7 columns):
#  #   Column          Non-Null Count    Dtype
# ---  ------          --------------    -----
#  0   date            1006029 non-null  object
#  1   state           1006029 non-null  object
#  2   district        1006029 non-null  object
#  3   pincode         1006029 non-null  int64
#  4   age_0_5         1006029 non-null  int64
#  5   age_5_17        1006029 non-null  int64
#  6   age_18_greater  1006029 non-null  int64
# dtypes: int64(4), object(3)
# memory usage: 53.7+ MB
# None

print(df_enrol.isnull().sum())  # Are there any missing values?
# date              0
# state             0
# district          0
# pincode           0
# age_0_5           0
# age_5_17          0
# age_18_greater    0
# dtype: int64

print(df_enrol.describe())  # How does the data look mathematically?
#             pincode       age_0_5      age_5_17  age_18_greater
# count  1.006029e+06  1.006029e+06  1.006029e+06    1.006029e+06
# mean   5.186415e+05  3.525709e+00  1.710074e+00    1.673441e-01
# std    2.056360e+05  1.753851e+01  1.436963e+01    3.220525e+00
# min    1.000000e+05  0.000000e+00  0.000000e+00    0.000000e+00
# 25%    3.636410e+05  1.000000e+00  0.000000e+00    0.000000e+00
# 50%    5.174170e+05  2.000000e+00  0.000000e+00    0.000000e+00
# 75%    7.001040e+05  3.000000e+00  1.000000e+00    0.000000e+00
# max    8.554560e+05  2.688000e+03  1.812000e+03    8.550000e+02


print(df_enrol.duplicated().sum())    # Are there duplicate values?
# 22957

# How is the correlation betweeen cols?
print(df_enrol.corr(numeric_only=True))  
#                  pincode   age_0_5  age_5_17  age_18_greater
# pincode         1.000000 -0.026274 -0.001946        0.016032
# age_0_5        -0.026274  1.000000  0.773063        0.334540
# age_5_17       -0.001946  0.773063  1.000000        0.492281
# age_18_greater  0.016032  0.334540  0.492281        1.000000






print(f"Total Demographic Rows: {len(df_demo)}")
print(df_demo.head())
#          date           state    district  pincode  demo_age_5_17  demo_age_17_
# 0  01-03-2025   Uttar Pradesh   Gorakhpur   273213             49           529
# 1  01-03-2025  Andhra Pradesh    Chittoor   517132             22           375
# 2  01-03-2025         Gujarat      Rajkot   360006             65           765
# 3  01-03-2025  Andhra Pradesh  Srikakulam   532484             24           314
# 4  01-03-2025       Rajasthan     Udaipur   313801             45           785

print(df_demo.shape)
# (2071700, 6)

print(df_demo.info())     # What is the data type of cols?
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 2071700 entries, 0 to 2071699
# Data columns (total 6 columns):
#  #   Column         Dtype
# ---  ------         -----
#  0   date           object
#  1   state          object
#  2   district       object
#  3   pincode        int64
#  4   demo_age_5_17  int64
#  5   demo_age_17_   int64
# dtypes: int64(3), object(3)
# memory usage: 94.8+ MB
# None

print(df_demo.isnull().sum())    # Are there any missing values?
# date             0
# state            0
# district         0
# pincode          0
# demo_age_5_17    0
# demo_age_17_     0
# dtype: int64

print(df_demo.describe())       # How does the data look mathematically?
#             pincode  demo_age_5_17  demo_age_17_
# count  2.071700e+06   2.071700e+06  2.071700e+06
# mean   5.278318e+05   2.347552e+00  2.144701e+01
# std    1.972933e+05   1.490355e+01  1.252498e+02
# min    1.000000e+05   0.000000e+00  0.000000e+00
# 25%    3.964690e+05   0.000000e+00  2.000000e+00
# 50%    5.243220e+05   1.000000e+00  6.000000e+00
# 75%    6.955070e+05   2.000000e+00  1.500000e+01
# max    8.554560e+05   2.690000e+03  1.616600e+04

print(df_demo.duplicated().sum())        # Are there duplicate values?
# 473601

print(df_demo.corr(numeric_only=True))    # How is the correlation betweeen cols?
#                 pincode  demo_age_5_17  demo_age_17_
# pincode        1.000000      -0.041052     -0.036542
# demo_age_5_17 -0.041052       1.000000      0.854358
# demo_age_17_  -0.036542       0.854358      1.000000




print(f"Total Biometric Rows: {len(df_bio)}")
print(df_bio.head())
#          date              state      district  pincode  bio_age_5_17  bio_age_17_
# 0  01-03-2025            Haryana  Mahendragarh   123029           280          577
# 1  01-03-2025              Bihar     Madhepura   852121           144          369
# 2  01-03-2025  Jammu and Kashmir         Punch   185101           643         1091
# 3  01-03-2025              Bihar       Bhojpur   802158           256          980
# 4  01-03-2025         Tamil Nadu       Madurai   625514           271          815

print(df_bio.shape)
# (1861108, 6)

print(df_bio.info())      # What is the data type of cols?
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 1861108 entries, 0 to 1861107
# Data columns (total 6 columns):
#  #   Column        Dtype
# ---  ------        -----
#  0   date          object
#  1   state         object
#  2   district      object
#  3   pincode       int64
#  4   bio_age_5_17  int64
#  5   bio_age_17_   int64
# dtypes: int64(3), object(3)
# memory usage: 85.2+ MB
# None

print(df_bio.isnull().sum())     # Are there any missing values?
# date            0
# state           0
# district        0
# pincode         0
# bio_age_5_17    0
# bio_age_17_     0
# dtype: int64

print(df_bio.describe())      # How does the data look mathematically?
#             pincode  bio_age_5_17   bio_age_17_
# count  1.861108e+06  1.861108e+06  1.861108e+06
# mean   5.217612e+05  1.839058e+01  1.909413e+01
# std    1.981627e+05  8.370421e+01  8.806502e+01
# min    1.100010e+05  0.000000e+00  0.000000e+00
# 25%    3.911750e+05  1.000000e+00  1.000000e+00
# 50%    5.224010e+05  3.000000e+00  4.000000e+00
# 75%    6.866362e+05  1.100000e+01  1.000000e+01
# max    8.554560e+05  8.002000e+03  7.625000e+03

print(df_bio.duplicated().sum())         # Are there duplicate values?
# 94896

print(df_bio.corr(numeric_only=True))     # How is the correlation betweeen cols?
#                pincode  bio_age_5_17  bio_age_17_
# pincode       1.000000     -0.060449    -0.036943
# bio_age_5_17 -0.060449      1.000000     0.786095
# bio_age_17_  -0.036943      0.786095     1.000000





# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Phase 2: Data Cleaning & Geographic Standardization
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# 1. PERMANENT DUPLICATE REMOVAL
# We clean each dataset individually before merging to ensure data integrity.
df_enrol.drop_duplicates(inplace=True)
df_demo.drop_duplicates(inplace=True)
df_bio.drop_duplicates(inplace=True)

# Resetting index to ensure a clean dataframe structure
df_enrol.reset_index(drop=True, inplace=True)
df_demo.reset_index(drop=True, inplace=True)
df_bio.reset_index(drop=True, inplace=True)

# 2. DATE STANDARDIZATION
# Convert 'date' to datetime objects so they align perfectly during the merge.
df_enrol['date'] = pd.to_datetime(df_enrol['date'], dayfirst=True)
df_demo['date'] = pd.to_datetime(df_demo['date'], dayfirst=True)
df_bio['date'] = pd.to_datetime(df_bio['date'], dayfirst=True)

print("Duplicates removed permanently.")
print(f"Final Row Counts -> Enrol: {len(df_enrol)}, Demo: {len(df_demo)}, Bio: {len(df_bio)}")
# Final Row Counts -> Enrol: 983072, Demo: 1598099, Bio: 1766212







# 3. THE MASTER MERGE
# We use a composite key of 4 columns to uniquely identify each service row.
merge_keys = ['date', 'state', 'district', 'pincode']

# Step A: Join Enrolment and Demographic Data
master_df = pd.merge(df_enrol, df_demo, on=merge_keys, how='outer')

# Step B: Join the result with Biometric Data
master_df = pd.merge(master_df, df_bio, on=merge_keys, how='outer')

# 4. HANDLING MISSING VALUES
# After an outer join, any missing activity is filled with 0.
master_df = master_df.fillna(0)

print("✅ Data Cleaning and Merging Complete!")
print(f"Final Merged Dataset Shape: {master_df.shape}")
# Final Merged Dataset Shape: (2330468, 11)
print(master_df.head())
#         date                        state  district  pincode  age_0_5  age_5_17  age_18_greater  demo_age_5_17  demo_age_17_  bio_age_5_17  bio_age_17_
# 0 2025-03-01    Andaman & Nicobar Islands  Andamans   744101      0.0       0.0             0.0            0.0           0.0          16.0        193.0
# 1 2025-03-01  Andaman and Nicobar Islands   Nicobar   744301      0.0       0.0             0.0           16.0         180.0         101.0         48.0
# 2 2025-03-01  Andaman and Nicobar Islands   Nicobar   744302      0.0       0.0             0.0            0.0           0.0          15.0         12.0
# 3 2025-03-01  Andaman and Nicobar Islands   Nicobar   744303      0.0       0.0             0.0            0.0           0.0          46.0         27.0
# 4 2025-03-01  Andaman and Nicobar Islands   Nicobar   744304      0.0       0.0             0.0            0.0           0.0          16.0         14.0


print(f"Total Unique States: {master_df['state'].nunique()}")
# Total Unique States: 68
print(f"Total Unique States: {master_df['state'].unique()}")
# Total Unique States: ['Andaman & Nicobar Islands' 'Andaman and Nicobar Islands'
#  'Andhra Pradesh' 'Arunachal Pradesh' 'Assam' 'Bihar' 'Chandigarh'
#  'Chhattisgarh' 'Dadra & Nagar Haveli' 'Dadra and Nagar Haveli'
#  'Dadra and Nagar Haveli and Daman and Diu' 'Daman & Diu' 'Daman and Diu'
#  'Delhi' 'Goa' 'Gujarat' 'Haryana' 'Himachal Pradesh' 'Jammu and Kashmir'
#  'Jharkhand' 'Karnataka' 'Kerala' 'Ladakh' 'Lakshadweep' 'Madhya Pradesh'
#  'Maharashtra' 'Manipur' 'Meghalaya' 'Mizoram' 'Nagaland' 'Odisha'
#  'Orissa' 'Pondicherry' 'Puducherry' 'Punjab' 'Rajasthan' 'Sikkim'
#  'Tamil Nadu' 'Telangana' 'Tripura' 'Uttar Pradesh' 'Uttarakhand'
#  'West Bengal' 'The Dadra And Nagar Haveli And Daman And Diu'
#  'Jammu And Kashmir' 'Jammu & Kashmir' 'ODISHA' 'WEST BENGAL' 'WESTBENGAL'
#  'West  Bengal' 'West bengal' 'Westbengal' 'andhra pradesh' 'odisha'
#  'west Bengal' '100000' 'West Bangal' 'Uttaranchal' 'Chhatisgarh'
#  'West Bengli' 'BALANAGAR' 'Darbhanga' 'Puttenahalli' 'Nagpur' 'Jaipur'
#  'Raja Annamalai Puram' 'Madanapalle' 'Tamilnadu']
print(f"Total Unique Districts: {master_df['district'].nunique()}")
# Total Unique Districts: 1029
print(f"Total Unique Districts: {master_df['district'].unique()}")
# Total Unique Districts: ['Andamans' 'Nicobar' 'North And Middle Andaman' ... 'Near meera hospital'
#  'Near Dhyana Ashram' 'Kadiri Road']



# 1. DEFINE COMPREHENSIVE MAPPING DICTIONARY
# This handles complex renames, historical names, and mergers (e.g., UT merger of 2020)
state_mapping = {
    # Union Territory Mergers & Variations
    'Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra And Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
    'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'Dadra & Nagar Haveli And Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
    
    # Common State Name Typos & Historical Fixes
    'Orissa': 'Odisha',
    'Odisha': 'Odisha',
    'Westbengal': 'West Bengal',
    'West Bengli': 'West Bengal',
    'West  Bengal': 'West Bengal',
    'West Bangal': 'West Bengal',
    'Tamilnadu': 'Tamil Nadu',
    'Chhatisgarh': 'Chhattisgarh',
    'Chattisgarh': 'Chhattisgarh',
    'Uttaranchal': 'Uttarakhand',
    'Pondicherry': 'Puducherry',
    'Telengana': 'Telangana',
    
    # Symbols & Spacing Fixes
    'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
    'Andaman And Nicobar Islands': 'Andaman and Nicobar Islands',
    'Jammu & Kashmir': 'Jammu and Kashmir',
    'Jammu And Kashmir': 'Jammu and Kashmir',
    'Delhi (National Capital Territory)': 'Delhi',
    'Nct Of Delhi': 'Delhi'
}

# 2. DEFINE THE ENCODING FUNCTION
def encode_geography(df):
    """
    Cleans and standardizes State and District columns to ensure data integrity.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df = df.copy()
    
    geo_columns = ['state', 'district']
    
    for col in geo_columns:
        if col in df.columns:
            # Step A: Remove leading/trailing spaces and weird characters
            df[col] = df[col].astype(str).str.strip()
            
            # Step B: Standardize to Title Case (handles 'ODISHA' vs 'odisha')
            df[col] = df[col].str.title()
    
    # Step C: Apply the complex mapping dictionary specifically to 'state'
    if 'state' in df.columns:
        df['state'] = df['state'].replace(state_mapping)
        
    return df

# 3. APPLY TO MASTER DATASET
master_df = encode_geography(master_df)


# 4. FINAL VERIFICATION
print(f"Total Standardized States after cleaning: {master_df['state'].nunique()}")
# Total Standardized States: 45
print("Standardized State List after cleaning:", master_df['state'].unique())
# Standardized State List: ['Andaman and Nicobar Islands' 'Andhra Pradesh' 'Arunachal Pradesh'
#  'Assam' 'Bihar' 'Chandigarh' 'Chhattisgarh'
#  'Dadra and Nagar Haveli and Daman and Diu'
#  'Dadra And Nagar Haveli And Daman And Diu' 'Delhi' 'Goa' 'Gujarat'
#  'Haryana' 'Himachal Pradesh' 'Jammu and Kashmir' 'Jharkhand' 'Karnataka'
#  'Kerala' 'Ladakh' 'Lakshadweep' 'Madhya Pradesh' 'Maharashtra' 'Manipur'
#  'Meghalaya' 'Mizoram' 'Nagaland' 'Odisha' 'Puducherry' 'Punjab'
#  'Rajasthan' 'Sikkim' 'Tamil Nadu' 'Telangana' 'Tripura' 'Uttar Pradesh'
#  'Uttarakhand' 'West Bengal' '100000' 'Balanagar' 'Darbhanga'
#  'Puttenahalli' 'Nagpur' 'Jaipur' 'Raja Annamalai Puram' 'Madanapalle']




# 1. DEFINE DISTRICT MAPPING DICTIONARY
# This merges specific local areas/typos into the actual district names
district_mapping = {
    # Bangalore & Urban Transitions
    'Puttenahalli': 'Bengaluru Urban',
    'Bengaluru Urban': 'Bengaluru Urban',
    'Bengaluru Rural': 'Bengaluru Rural',
    'Bengaluru South': 'Bengaluru Urban',
    
    # Hyderabad & Telangana Fragmentation
    'Balanagar': 'Hyderabad',
    'Hanumakonda': 'Hanamkonda',
    'Jangoan': 'Jangaon',
    'Medchal-Malkajgiri': 'Medchal-Malkajgiri',
    'Medchal?Malkajgiri': 'Medchal-Malkajgiri',
    'Medchalâ\x88\x92Malkajgiri': 'Medchal-Malkajgiri',
    
    # Tamil Nadu & Andhra Updates
    'Raja Annamalai Puram': 'Chennai',
    'Madanapalle': 'Chittoor',
    'Visakhapatanam': 'Visakhapatnam',
    'Tuticorin': 'Thoothukkudi',
    
    # Cleaning Numeric/Non-District Garbage
    '100000': 'Unknown',
    'Near Meera Hospital': 'Unknown',
    'Near Dhyana Ashram': 'Unknown',
    'Kadiri Road': 'Unknown',
    'Dist : Thane': 'Thane',
    '?': 'Unknown',
    
    # West Bengal Specific Unification
    'West Bengli': 'West Bengal',
    'Naihati Anandabazar': 'North 24 Parganas',
    'Domjur': 'Howrah',
    'Dinajpur Uttar': 'Uttar Dinajpur',
    'Dinajpur Dakshin': 'Dakshin Dinajpur',
    'South 24 Pargana': 'South 24 Parganas',
    
    # Maharashtra & Chhattisgarh Formatting
    'Manendragarhchirmiribharatpur': 'Manendragarh-Chirmiri-Bharatpur',
    'Manendragarh–Chirmiri–Bharatpur': 'Manendragarh-Chirmiri-Bharatpur',
    'Raigarh(Mh)': 'Raigarh',
    'Ahilyanagar': 'Ahmednagar'
}

# 2. UPDATED ENCODING FUNCTION
def encode_geography_v2(df):
    """
    Advanced standardization for both State and District columns.
    """
    df = df.copy()
    
    # Step A: Universal Case and Space Standardization
    # This automatically fixes 'NAGPUR' vs 'Nagpur' or ' jaipur' vs 'Jaipur'
    for col in ['state', 'district']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
    
    # Step B: Apply State Mapping (from previous step)
    if 'state' in df.columns:
        df['state'] = df['state'].replace(state_mapping)
        
    # Step C: Apply District Mapping to reduce count
    if 'district' in df.columns:
        df['district'] = df['district'].replace(district_mapping)
        
        # Step D: Filter out non-district garbage (optional but recommended)
        # We can flag districts with common "street" or "near" keywords
        garbage_keywords = ['Near', 'Road', 'Hospital', 'Ashram', 'Lane', 'Cross']
        df.loc[df['district'].str.contains('|'.join(garbage_keywords), na=False), 'district'] = 'Other'

    return df

# 3. APPLY AND VERIFY
master_df = encode_geography_v2(master_df)
# --- STEP: CREATE RAW METRICS (Required for all Indices) ---
# We must sum the individual age groups into total columns first

# 1. Total New Enrolments
master_df['total_enrol'] = master_df['age_0_5'] + master_df['age_5_17'] + master_df['age_18_greater']

# 2. Total Updates (Demographic + Biometric)
master_df['total_updates'] = (master_df['demo_age_5_17'] + master_df['demo_age_17_'] + 
                             master_df['bio_age_5_17'] + master_df['bio_age_17_'])

# 3. Total Combined Activity
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']

print("✅ Raw metrics (total_enrol, total_updates, total_activity) created successfully.")

print(f"Total Unique Districts: {master_df['district'].nunique()}")
# Total Unique Districts: 995
print(f"Total Unique Districts: {master_df['district'].unique()}")
# Total Unique Districts: ['Andamans' 'Nicobar' 'North And Middle Andaman' 'South Andaman'
#  'Adilabad' 'Alluri Sitharama Raju' 'Anakapalli' 'Anantapur' 'Ananthapur'
#  'Ananthapuramu' 'Annamayya' 'Bapatla' 'Chittoor' 'Cuddapah'
#  'Dr. B. R. Ambedkar Konaseema' 'East Godavari' 'Eluru' 'Guntur'
#  'Hyderabad' 'K.V. Rangareddy' 'Kakinada' 'Karim Nagar' 'Karimnagar'
#  'Khammam' 'Krishna' 'Kurnool' 'Mahabub Nagar' 'Mahbubnagar' 'Medak'  
#  'N. T. R' 'Nalgonda' 'Nandyal' 'Nellore' 'Nizamabad' 'Palnadu'
#  'Parvathipuram Manyam' 'Prakasam' 'Rangareddi'
#  'Sri Potti Sriramulu Nellore' 'Sri Sathya Sai' 'Srikakulam' 'Tirupati'
#  'Visakhapatnam' 'Vizianagaram' 'Warangal' 'West Godavari' 'Y. S. R'
#  'Anjaw' 'Changlang' 'Dibang Valley' 'East Kameng' 'East Siang'
#  'Kra Daadi' 'Kurung Kumey' 'Lohit' 'Longding' 'Lower Dibang Valley'
#  'Lower Siang' 'Lower Subansiri' 'Namsai' 'Pakke Kessang' 'Papum Pare'
#  'Shi-Yomi' 'Siang' 'Tawang' 'Tirap' 'Upper Siang' 'Upper Subansiri'
#  'West Kameng' 'West Siang' 'Baksa' 'Barpeta' 'Biswanath' 'Bongaigaon'
#  'Cachar' 'Charaideo' 'Chirang' 'Darrang' 'Dhemaji' 'Dhubri' 'Dibrugarh'
#  'Goalpara' 'Golaghat' 'Hailakandi' 'Hojai' 'Jorhat' 'Kamrup'
#  'Kamrup Metro' 'Karbi Anglong' 'Karimganj' 'Kokrajhar' 'Lakhimpur'
#  'Majuli' 'Marigaon' 'Nagaon' 'Nalbari' 'North Cachar Hills' 'Sibsagar'
#  'Sonitpur' 'South Salmara Mankachar' 'Tinsukia' 'Udalguri'
#  'West Karbi Anglong' 'Araria' 'Arwal' 'Aurangabad' 'Banka' 'Begusarai'
#  'Bhagalpur' 'Bhojpur' 'Buxar' 'Darbhanga' 'East Champaran' 'Gaya'
#  'Gopalganj' 'Jamui' 'Jehanabad' 'Kaimur (Bhabua)' 'Katihar' 'Khagaria'
#  'Kishanganj' 'Lakhisarai' 'Madhepura' 'Madhubani' 'Munger' 'Muzaffarpur'
#  'Nalanda' 'Nawada' 'Patna' 'Purnea' 'Purnia' 'Rohtas' 'Saharsa'
#  'Samastipur' 'Saran' 'Sheikhpura' 'Sheohar' 'Sitamarhi' 'Siwan' 'Supaul'
#  'Vaishali' 'West Champaran' 'Chandigarh' 'Balod' 'Baloda Bazar'
#  'Balrampur' 'Bastar' 'Bemetara' 'Bijapur' 'Bilaspur'
#  'Dakshin Bastar Dantewada' 'Dantewada' 'Dhamtari' 'Durg' 'Gariyaband'
#  'Gaurela-Pendra-Marwahi' 'Janjgir-Champa' 'Jashpur' 'Kabeerdham' 'Kanker'
#  'Kawardha' 'Kondagaon' 'Korba' 'Koriya' 'Mahasamund'
#  'Mohalla-Manpur-Ambagarh Chowki' 'Mohla-Manpur-Ambagarh Chouki' 'Mungeli'
#  'Narayanpur' 'Raigarh' 'Raipur' 'Rajnandgaon' 'Sakti' 'Sukma' 'Surajpur'
#  'Surguja' 'Uttar Bastar Kanker' 'Dadra & Nagar Haveli'
#  'Dadra And Nagar Haveli' 'Daman' 'Diu' 'Central Delhi' 'East Delhi'
#  'Najafgarh' 'New Delhi' 'North Delhi' 'North East' 'North East Delhi'
#  'North West Delhi' 'Shahdara' 'South Delhi' 'South East Delhi'
#  'South West Delhi' 'West Delhi' 'North Goa' 'South Goa' 'Ahmadabad'
#  'Ahmedabad' 'Amreli' 'Anand' 'Arvalli' 'Banaskantha' 'Bharuch'
#  'Bhavnagar' 'Botad' 'Chhotaudepur' 'Dahod' 'Devbhumi Dwarka' 'Dohad'
#  'Gandhinagar' 'Gir Somnath' 'Jamnagar' 'Junagadh' 'Kachchh' 'Kheda'
#  'Mahesana' 'Mahisagar' 'Morbi' 'Narmada' 'Navsari' 'Panchmahals' 'Patan'
#  'Porbandar' 'Rajkot' 'Sabarkantha' 'Surat' 'Surendra Nagar' 'Tapi'
#  'The Dangs' 'Vadodara' 'Valsad' 'Ambala' 'Bhiwani' 'Charkhi Dadri'
#  'Faridabad' 'Fatehabad' 'Gurgaon' 'Hisar' 'Jhajjar' 'Jind' 'Kaithal'
#  'Karnal' 'Kurukshetra' 'Mahendragarh' 'Mewat' 'Palwal' 'Panchkula'
#  'Panipat' 'Rewari' 'Rohtak' 'Sirsa' 'Sonipat' 'Yamuna Nagar'
#  'Yamunanagar' 'Chamba' 'Hamirpur' 'Kangra' 'Kinnaur' 'Kullu' 'Mandi'
#  'Shimla' 'Sirmaur' 'Solan' 'Una' 'Anantnag' 'Badgam' 'Bandipore'
#  'Baramula' 'Budgam' 'Doda' 'Ganderbal' 'Jammu' 'Kargil' 'Kathua'
#  'Kishtwar' 'Kulgam' 'Kupwara' 'Leh' 'Pulwama' 'Punch' 'Rajouri' 'Ramban'
#  'Reasi' 'Samba' 'Shupiyan' 'Srinagar' 'Udhampur' 'Bokaro' 'Chatra'
#  'Deoghar' 'Dhanbad' 'Dumka' 'East Singhbhum' 'Garhwa' 'Garhwa *'
#  'Giridih' 'Godda' 'Gumla' 'Hazaribag' 'Hazaribagh' 'Jamtara' 'Khunti'
#  'Kodarma' 'Koderma' 'Latehar' 'Lohardaga' 'Pakaur' 'Pakur' 'Palamau'
#  'Palamu' 'Pashchimi Singhbhum' 'Purbi Singhbhum' 'Ramgarh' 'Ranchi'
#  'Sahebganj' 'Sahibganj' 'Seraikela-Kharsawan' 'Simdega' 'West Singhbhum'
#  'Bagalkot' 'Bagalkot *' 'Ballari' 'Bangalore' 'Bangalore Rural'
#  'Belagavi' 'Belgaum' 'Bellary' 'Bengaluru' 'Bidar' 'Chamarajanagar'
#  'Chamrajanagar' 'Chamrajnagar' 'Chickmagalur' 'Chikkaballapur'
#  'Chikkamagaluru' 'Chikmagalur' 'Chitradurga' 'Dakshina Kannada'
#  'Davanagere' 'Davangere' 'Dharwad' 'Gadag' 'Gadag *' 'Gulbarga' 'Hasan'
#  'Hassan' 'Haveri' 'Haveri *' 'Kalaburagi' 'Kodagu' 'Kolar' 'Koppal'
#  'Mandya' 'Mysore' 'Mysuru' 'Raichur' 'Ramanagar' 'Shimoga' 'Shivamogga'
#  'Tumakuru' 'Tumkur' 'Udupi' 'Uttara Kannada' 'Vijayanagara' 'Vijayapura'
#  'Yadgir' 'Alappuzha' 'Ernakulam' 'Idukki' 'Kannur' 'Kasaragod' 'Kasargod'
#  'Kollam' 'Kottayam' 'Kozhikode' 'Malappuram' 'Palakkad' 'Pathanamthitta'
#  'Thiruvananthapuram' 'Thrissur' 'Wayanad' 'Lakshadweep' 'Agar Malwa'
#  'Alirajpur' 'Anuppur' 'Ashok Nagar' 'Balaghat' 'Barwani' 'Betul' 'Bhind'
#  'Bhopal' 'Burhanpur' 'Chhatarpur' 'Chhindwara' 'Damoh' 'Datia' 'Dewas'
#  'Dhar' 'Dindori' 'East Nimar' 'Guna' 'Gwalior' 'Harda' 'Harda *'
#  'Hoshangabad' 'Indore' 'Jabalpur' 'Jhabua' 'Katni' 'Khandwa' 'Khargone'
#  'Maihar' 'Mandla' 'Mandsaur' 'Mauganj' 'Morena' 'Narmadapuram'
#  'Narsimhapur' 'Narsinghpur' 'Neemuch' 'Niwari' 'Panna' 'Raisen' 'Rajgarh'
#  'Ratlam' 'Rewa' 'Sagar' 'Satna' 'Sehore' 'Seoni' 'Shahdol' 'Shajapur'
#  'Sheopur' 'Shivpuri' 'Sidhi' 'Singrauli' 'Tikamgarh' 'Ujjain' 'Umaria'
#  'Vidisha' 'West Nimar' 'Ahmadnagar' 'Ahmed Nagar' 'Akola' 'Amravati'
#  'Beed' 'Bhandara' 'Bid' 'Buldana' 'Buldhana' 'Chandrapur'
#  'Chatrapati Sambhaji Nagar' 'Chhatrapati Sambhajinagar' 'Dharashiv'
#  'Dhule' 'Gadchiroli' 'Gondiya' 'Hingoli' 'Jalgaon' 'Jalna' 'Kolhapur'
#  'Latur' 'Mumbai' 'Mumbai City' 'Mumbai Suburban' 'Nagpur' 'Nanded'
#  'Nandurbar' 'Nashik' 'Osmanabad' 'Palghar' 'Parbhani' 'Pune' 'Raigad'
#  'Ratnagiri' 'Sangli' 'Satara' 'Sindhudurg' 'Solapur' 'Thane' 'Wardha'
#  'Washim' 'Yavatmal' 'Bishnupur' 'Chandel' 'Churachandpur' 'Imphal East'
#  'Imphal West' 'Jiribam' 'Kakching' 'Senapati' 'Tamenglong' 'Thoubal'
#  'Ukhrul' 'East Garo Hills' 'East Jaintia Hills' 'East Khasi Hills'
#  'North Garo Hills' 'Ri Bhoi' 'South Garo Hills' 'South West Garo Hills'
#  'South West Khasi Hills' 'West Garo Hills' 'West Jaintia Hills'
#  'West Khasi Hills' 'Aizawl' 'Champhai' 'Kolasib' 'Lawngtlai' 'Lunglei'
#  'Mamit' 'Mammit' 'Saiha' 'Saitual' 'Serchhip' 'Chumukedima' 'Dimapur'
#  'Kiphire' 'Kohima' 'Longleng' 'Mokokchung' 'Mon' 'Niuland' 'Noklak'
#  'Peren' 'Phek' 'Tseminyu' 'Tuensang' 'Wokha' 'Zunheboto' 'Anugul' 'Angul'
#  'Balangir' 'Baleshwar' 'Baleswar' 'Bargarh' 'Baudh' 'Bhadrak' 'Boudh'
#  'Cuttack' 'Debagarh' 'Dhenkanal' 'Gajapati' 'Ganjam' 'Jajpur'
#  'Jagatsinghapur' 'Jagatsinghpur' 'Jajapur' 'Jharsuguda' 'Kalahandi'
#  'Kandhamal' 'Kendrapara' 'Kendujhar' 'Khorda' 'Khordha' 'Koraput'
#  'Malkangiri' 'Mayurbhanj' 'Nabarangapur' 'Nayagarh' 'Nuapada' 'Puri'
#  'Rayagada' 'Sambalpur' 'Sonapur' 'Subarnapur' 'Sundargarh' 'Sundergarh'
#  'Pondicherry' 'Karaikal' 'Puducherry' 'Amritsar' 'Barnala' 'Bathinda'
#  'Faridkot' 'Fatehgarh Sahib' 'Fazilka' 'Ferozepur' 'Firozpur' 'Gurdaspur'
#  'Hoshiarpur' 'Jalandhar' 'Kapurthala' 'Ludhiana' 'Malerkotla' 'Mansa'
#  'Moga' 'Muktsar' 'Pathankot' 'Patiala' 'Rupnagar' 'S.A.S Nagar(Mohali)'
#  'Sas Nagar (Mohali)' 'Sangrur' 'Shaheed Bhagat Singh Nagar'
#  'Sri Muktsar Sahib' 'Tarn Taran' 'Ajmer' 'Alwar' 'Banswara' 'Baran'
#  'Barmer' 'Bharatpur' 'Bhilwara' 'Bikaner' 'Bundi' 'Chittaurgarh'
#  'Chittorgarh' 'Churu' 'Dausa' 'Dholpur' 'Dungarpur' 'Ganganagar'
#  'Hanumangarh' 'Jaipur' 'Jaisalmer' 'Jalor' 'Jhalawar' 'Jhunjhunu'
#  'Jhunjhunun' 'Jodhpur' 'Karauli' 'Kota' 'Nagaur' 'Pali' 'Pratapgarh'
#  'Rajsamand' 'Sawai Madhopur' 'Sikar' 'Sirohi' 'Tonk' 'Udaipur' 'East'
#  'East Sikkim' 'North' 'North Sikkim' 'South' 'South Sikkim' 'West'
#  'West Sikkim' 'Ariyalur' 'Chengalpattu' 'Chennai' 'Coimbatore'
#  'Cuddalore' 'Dharmapuri' 'Dindigul' 'Erode' 'Kallakurichi' 'Kancheepuram'
#  'Kanniyakumari' 'Kanyakumari' 'Karur' 'Krishnagiri' 'Madurai'
#  'Mayiladuthurai' 'Nagapattinam' 'Namakkal' 'Perambalur' 'Pudukkottai'
#  'Ramanathapuram' 'Ranipet' 'Salem' 'Sivaganga' 'Tenkasi' 'Thanjavur'
#  'The Nilgiris' 'Theni' 'Thiruvallur' 'Thiruvarur' 'Thoothukkudi'
#  'Tiruchirappalli' 'Tirunelveli' 'Tirupattur' 'Tiruppur' 'Tiruvallur'
#  'Tiruvannamalai' 'Vellore' 'Villupuram' 'Viluppuram' 'Virudhunagar'
#  'Bhadradri Kothagudem' 'Hanumakonda' 'Jagitial' 'Jangaon' 'Jangoan'
#  'Jayashankar Bhupalpally' 'Jogulamba Gadwal' 'Kamareddy' 'Komaram Bheem'
#  'Mahabubabad' 'Mahabubnagar' 'Mancherial' 'Medchal-Malkajgiri'
#  'Medchal?Malkajgiri' 'Mulugu' 'Nagarkurnool' 'Narayanpet' 'Nirmal'
#  'Peddapalli' 'Rajanna Sircilla' 'Rangareddy' 'Sangareddy' 'Siddipet'
#  'Suryapet' 'Vikarabad' 'Wanaparthy' 'Warangal Rural' 'Warangal Urban'
#  'Yadadri.' 'Dhalai' 'Dhalai  *' 'Gomati' 'Khowai' 'North Tripura'
#  'Sepahijala' 'South Tripura' 'Unakoti' 'West Tripura' 'Agra' 'Aligarh'
#  'Allahabad' 'Ambedkar Nagar' 'Amethi' 'Amroha' 'Auraiya' 'Ayodhya'
#  'Azamgarh' 'Baghpat' 'Bahraich' 'Ballia' 'Banda' 'Bara Banki' 'Barabanki'
#  'Bareilly' 'Basti' 'Bhadohi' 'Bijnor' 'Budaun' 'Bulandshahr' 'Chandauli'
#  'Chitrakoot' 'Deoria' 'Etah' 'Etawah' 'Faizabad' 'Farrukhabad' 'Fatehpur'
#  'Firozabad' 'Gautam Buddha Nagar' 'Ghaziabad' 'Ghazipur' 'Gonda'
#  'Gorakhpur' 'Hapur' 'Hardoi' 'Hathras' 'Jalaun' 'Jaunpur' 'Jhansi'
#  'Kannauj' 'Kanpur Dehat' 'Kanpur Nagar' 'Kasganj' 'Kaushambi' 'Kheri'
#  'Kushinagar' 'Lalitpur' 'Lucknow' 'Maharajganj' 'Mahoba' 'Mainpuri'
#  'Mathura' 'Mau' 'Meerut' 'Mirzapur' 'Moradabad' 'Muzaffarnagar'
#  'Pilibhit' 'Prayagraj' 'Rae Bareli' 'Rampur' 'Saharanpur' 'Sambhal'
#  'Sant Kabir Nagar' 'Sant Ravidas Nagar' 'Shahjahanpur' 'Shamli'
#  'Shrawasti' 'Siddharthnagar' 'Sitapur' 'Sonbhadra' 'Sultanpur' 'Unnao'
#  'Varanasi' 'Almora' 'Bageshwar' 'Chamoli' 'Champawat' 'Dehradun'
#  'Haridwar' 'Nainital' 'Pauri Garhwal' 'Pithoragarh' 'Rudraprayag'
#  'Tehri Garhwal' 'Udham Singh Nagar' 'Uttarkashi' 'Alipurduar' 'Bankura'
#  'Barddhaman' 'Bardhaman' 'Birbhum' 'Cooch Behar' 'Dakshin Dinajpur'
#  'Darjeeling' 'Darjiling' 'East Midnapore' 'Haora' 'Hooghly' 'Howrah'
#  'Jalpaiguri' 'Jhargram' 'Kalimpong' 'Koch Bihar' 'Kolkata' 'Malda'
#  'Maldah' 'Murshidabad' 'Nadia' 'North 24 Parganas'
#  'North Twenty Four Parganas' 'Paschim Bardhaman' 'Paschim Medinipur'
#  'Purba Bardhaman' 'Purba Medinipur' 'Purulia' 'Puruliya'
#  'South 24 Parganas' 'South Dinajpur' 'South Twenty Four Parganas'
#  'Uttar Dinajpur' 'West Midnapore' 'Purbi Champaran' 'Gurugram'
#  'Bengaluru Urban' 'Coochbehar' 'Dinajpur Uttar' 'Spsr Nellore'
#  'Banas Kantha' 'Kanchipuram' 'Pashchim Champaran'
#  'Manendragarh-Chirmiri-Bharatpur' 'Mumbai( Sub Urban )' 'North Dinajpur'
#  'Visakhapatanam' 'Kamle' 'Dima Hasao' 'Sivasagar'
#  'Gaurella Pendra Marwahi' 'Khairagarh Chhuikhadan Gandai' 'Dang' 'Nuh'
#  'Lahul & Spiti' 'Lahul And Spiti' 'Bengaluru Rural' 'Ashoknagar'
#  'Pandhurna' 'S.A.S Nagar' 'Medchal Malkajgiri' 'Shravasti'
#  'Siddharth Nagar' 'Medinipur West' 'K.V.Rangareddy' 'Panch Mahals'
#  'Sabar Kantha' 'Surendranagar' 'Shopian' 'East Singhbum' 'Ramanagara'
#  'Ahmednagar' 'Shamator' 'Nabarangpur' 'Nawanshahr' 'Dhaulpur' 'Jalore'
#  'Ranga Reddy' 'Kushi Nagar' 'Kushinagar *' 'Dinajpur Dakshin' 'Nicobars'
#  'Leparada' 'Bajali' 'Sribhumi' 'Tamulpur District' 'Aurangabad(Bh)'
#  'Bhabua' 'Monghyr' 'Purba Champaran' 'Samstipur' 'Sheikpura'
#  'Janjgir - Champa' 'Janjgir Champa' 'Manendragarh–Chirmiri–Bharatpur'
#  'Sarangarh-Bilaigarh' 'Bardez' 'Tiswadi' 'Bengaluru South'
#  'Chamarajanagar *' 'Gondia' 'Gondiya *' 'Nandurbar *' 'Raigarh(Mh)'
#  'Washim *' 'Eastern West Khasi Hills' 'Khawzawl' 'Meluri' 'Yanam' 'Deeg'
#  'Namchi' 'Tirupathur' 'Medchal−Malkajgiri' 'Bulandshahar'
#  'Jyotiba Phule Nagar' 'Mahrajganj' 'Raebareli'
#  'Sant Ravidas Nagar Bhadohi' 'Garhwal' 'Hardwar' 'Hawrah' 'Hooghiy'
#  'Hugli' 'Medinipur' 'South 24 Pargana' 'Unknown' 'Pherzawl' 'Hnahthial'
#  'Mangan' 'Tuticorin' 'Warangal (Urban)' 'Burdwan' 'Udupi *'
#  'Jaintia Hills' 'Bagpat' 'Mahoba *' 'West Medinipur' 'Jhajjar *'
#  'Leh (Ladakh)' 'Rajauri' 'Lahaul And Spiti' 'Hingoli *' 'East Midnapur'
#  'South Dumdum(M)' 'Bally Jagachha' 'Anugal' 'Baghpat *' 'Mohali'
#  'Bijapur(Kar)' 'Tiruvarur' 'Domjur' 'Bokaro *' 'Jajapur  *'
#  'North East   *' 'Namakkal   *' 'Bandipur' 'Salumbar'
#  'Gautam Buddha Nagar *' 'Bicholim' 'Naihati Anandabazar' 'Anugul  *'
#  'Akhera' 'Kendrapara *' 'South  Twenty Four Parganas' 'Auraiya *'
#  'Jyotiba Phule Nagar *' 'Phalodi' 'Balotra' 'Didwana-Kuchaman'
#  'Khairthal-Tijara' 'Kotputli-Behror' 'Beawar' '?' 'Poonch' 'Bhadrak(R)'
#  'Khordha  *' 'Ahilyanagar' 'Idpl Colony' 'Dist : Thane' 'Other'
#  'Udham Singh Nagar *' 'Balianta' 'Chandauli *' 'Kangpokpi'
#  'Medchalâ\x88\x92Malkajgiri' 'Chitrakoot *']












# 1. Mandatory Biometric Update (MBU) Compliance Score
# Aadhaar regulations require children to update biometrics at ages 5 and 15. A district where many children are enrolling but few are updating biometrics is at risk of mass authentication failures in the future.
# Logic: Ratio of school-age biometric updates to school-age enrolments.
# Formula: master_df['mbu_compliance'] = master_df['bio_age_5_17'] / (master_df['age_5_17'] + 1)
# Insight: Identifies "Policy Gaps." A low score indicates that school-age children in that district are missing their mandatory revalidation.

import matplotlib.pyplot as plt
import seaborn as sns

# 1. STANDARDIZE NAMES (To ensure regional accuracy)
master_df['state'] = master_df['state'].str.strip().str.title()
master_df['district'] = master_df['district'].str.strip().str.title()

# 2. CALCULATE MBU COMPLIANCE SCORE
# We use bio_age_5_17 (updates) and age_5_17 (enrolments)
# Adding +1 to the denominator prevents DivisionByZero errors
master_df['mbu_compliance'] = master_df['bio_age_5_17'] / (master_df['age_5_17'] + 1)

# 3. IDENTIFY TOP AND BOTTOM PERFORMING DISTRICTS
mbu_ranking = master_df.groupby(['state', 'district'])['mbu_compliance'].mean().sort_values(ascending=False).reset_index()

print("--- Districts with High MBU Compliance (Healthy) ---")
print(mbu_ranking.head(5))
# --- Districts with High MBU Compliance (Healthy) ---
#                state          district  mbu_compliance
# 0              Delhi  North East Delhi      114.971366
# 1     Madhya Pradesh             Sidhi       75.695660
# 2  Jammu And Kashmir          Shupiyan       64.934514
# 3     Madhya Pradesh         Singrauli       63.936402
# 4     Madhya Pradesh             Damoh       57.961394

print("\n--- Districts with Low MBU Compliance (Critical Policy Gaps) ---")
print(mbu_ranking.tail(5))
# --- Districts with Low MBU Compliance (Critical Policy Gaps) ---
#                state             district  mbu_compliance
# 1020     West Bengal               Domjur             0.0
# 1021     West Bengal     Dinajpur Dakshin             0.0
# 1022  Andhra Pradesh       Visakhapatanam             0.0
# 1023     West Bengal       Medinipur West             0.0
# 1024     West Bengal  Naihati Anandabazar             0.0

# 4. VISUALIZE THE COMPLIANCE GAP BY STATE
plt.figure(figsize=(12, 10))

# Grouping and sorting data for the plot
plot_data = mbu_ranking.groupby('state')['mbu_compliance'].mean().sort_values(ascending=False).reset_index()

# THE FIX: Assign 'state' to 'hue' and set 'legend=False' to satisfy new standards
sns.barplot(
    data=plot_data, 
    x='mbu_compliance', 
    y='state', 
    hue='state',      # Explicitly map color to the 'state' column
    palette='magma',  # Keep your chosen color theme
    legend=False      # Removes the redundant legend that hue would create
)

plt.title('State-wise Mandatory Biometric Update (MBU) Compliance', fontsize=16)
plt.xlabel('Compliance Score (Ratio of Updates to Enrolments)', fontsize=12)

# Reference line for ideal performance
plt.axvline(1.0, color='red', linestyle='--', label='Ideal Ratio (1:1)')

plt.legend() # This will only show the label for the red dashed line
plt.tight_layout()
plt.savefig('mbu_compliance_map.png', bbox_inches='tight')
plt.show()





# 2. The "Digital Mobility" Index
# Demographic updates (address and mobile) are strong indicators of people moving for work or updating IDs for digital services.
# Logic: Adult demographic updates relative to total adult activity.
# Formula: master_df['mobility_index'] = master_df['demo_age_17_'] / (master_df['age_18_greater'] + master_df['demo_age_17_'] + 1)
# Insight: High scores suggest an Urban/Migrant Hub where people frequently update addresses. Low scores suggest a static rural population................. 

# 1. Calculate the Index
master_df['mobility_index'] = master_df['demo_age_17_'] / (master_df['age_18_greater'] + master_df['demo_age_17_'] + 1)

# 2. Group by District to find the Hubs
mobility_ranking = master_df.groupby(['state', 'district'])['mobility_index'].mean().sort_values(ascending=False).reset_index()

# 3. Print the results
print("Top 5 'Migrant Hubs' (High Digital Mobility):")
print(mobility_ranking.head(5))
# Top 5 'Migrant Hubs' (High Digital Mobility):
#           state                      district  mobility_index
# 0  Chhattisgarh  Mohla-Manpur-Ambagarh Chouki        0.716222
# 1       Haryana                         Mewat        0.705974
# 2         Delhi                   North Delhi        0.699714
# 3         Assam       South Salmara Mankachar        0.699557
# 4     Rajasthan              Khairthal-Tijara        0.694914
print("\nTop 5 'Static Districts' (Low Digital Mobility):")
print(mobility_ranking.tail(5))
# Top 5 'Static Districts' (Low Digital Mobility):
#                state          district  mobility_index
# 1018     West Bengal           Burdwan             0.0
# 1019  Madhya Pradesh        Ashoknagar             0.0
# 1020     Maharashtra        Ahmednagar             0.0
# 1021     West Bengal  Dinajpur Dakshin             0.0
# 1022     West Bengal    Dinajpur Uttar             0.0






# 3. Service Saturation Ratio
# This helps distinguish between "New Growth" areas and "Mature" areas.
# Logic: Total updates divided by total activity (Enrolments + Updates).
# Formula: master_df['saturation_ratio'] = master_df['total_updates'] / (master_df['total_activity'] + 1)
# Insight:
# High Ratio (>0.8): Mature Market. Most people already have Aadhaar; activity is just maintenance.
# Low Ratio (<0.3): Emerging Market. High volume of new enrolments; needs more enrolment kits.

import matplotlib.pyplot as plt
import seaborn as sns

# 1. STANDARDIZE & PREPARE DATA
master_df['state'] = master_df['state'].str.strip().str.title()

# Ensure required columns exist
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']
master_df['saturation_ratio'] = master_df['total_updates'] / (master_df['total_activity'] + 1)

# 2. AGGREGATE BY STATE
state_maturity = master_df.groupby('state')['saturation_ratio'].mean().sort_values(ascending=False).reset_index()

# 3. CREATE THE VISUALIZATION
plt.figure(figsize=(12, 10))
sns.set_theme(style="whitegrid")

# Create a bar chart with a color palette that indicates 'Maturity'
barplot = sns.barplot(
    data=state_maturity, 
    x='saturation_ratio', 
    y='state', 
    palette='RdYlGn' # Green = Highly Mature (High Updates), Red = Emerging (High Enrolment)
)

# 4. ADD STRATEGIC THRESHOLD LINES
plt.axvline(0.8, color='green', linestyle='--', alpha=0.6, label='Mature Threshold (>0.8)')
plt.axvline(0.3, color='red', linestyle='--', alpha=0.6, label='Emerging Threshold (<0.3)')

# 5. TITLES AND ANNOTATIONS
plt.title('Market Maturity Map: Service Saturation by State', fontsize=18, pad=20)
plt.xlabel('Saturation Ratio (Updates / Total Activity)', fontsize=14)
plt.ylabel('State', fontsize=14)
plt.legend(loc='lower right')

# Add text labels for clarity
plt.text(0.82, len(state_maturity)-1, 'Maintenance Focused', color='green', fontweight='bold')
plt.text(0.05, len(state_maturity)-1, 'Growth Focused', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('market_maturity_map.png', bbox_inches='tight')
plt.show()








# 4. Late Adopter Density
# Why are adults (18+) still enrolling in 2025? This identifies marginalized or remote populations finally entering the system.
# Logic: New adult enrolments as a percentage of total new enrolments.
# Formula: master_df['late_adopter_ratio'] = master_df['age_18_greater'] / (master_df['total_enrol'] + 1)
# Insight: High density in specific Pincodes suggests a need for Inclusion Drives in those specific neighborhoods.............. 

# 1. Calculate the ratio
master_df['late_adopter_ratio'] = master_df['age_18_greater'] / (master_df['total_enrol'] + 1)
# 2. Identify the top 5 Pincodes where adults are enrolling the most
priority_zones = master_df.groupby('pincode')['late_adopter_ratio'].mean().sort_values(ascending=False).head(5)
print("--- Priority Inclusion Zones (High Late Adopter Density) ---")
print(priority_zones)
# --- Priority Inclusion Zones (High Late Adopter Density) ---
# pincode
# 100000    0.601422
# 384520    0.500000
# 793116    0.266668
# 793009    0.258771
# 464570    0.250000
# Name: late_adopter_ratio, dtype: float64









# 5. The District Health Index (DHI)
# You have already implemented a version of this in your code. It combines multiple metrics into one "Quality of Service" score.
# Formula (Revised):
# $$DHI = (Normalized\ MBU\ Compliance \times 0.4) + (Normalized\ Saturation\ Ratio \times 0.6)$$
# Insight: This single number allows the government to rank 700+ districts instantly....................


# 1. STANDARDIZE NAMES (Crucial for accurate grouping)  
# This prevents 'ODISHA' and 'odisha' from being calculated as separate entities.
master_df['state'] = master_df['state'].str.strip().str.title()
master_df['district'] = master_df['district'].str.strip().str.title()

# 2. FEATURE ENGINEERING: COMPONENT RATIOS
# Mandatory Biometric Update (MBU) Compliance: Focuses on the 5-17 age gap
master_df['mbu_compliance'] = master_df['bio_age_5_17'] / (master_df['age_5_17'] + 1)

# Saturation Ratio: Measures system maturity (Updates vs Total Activity)
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']
master_df['saturation_ratio'] = master_df['total_updates'] / (master_df['total_activity'] + 1)

# 3. NORMALIZATION FUNCTION
# This converts raw ratios to a 0-1 scale so they can be weighted together.
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

# 4. CALCULATE REVISED DHI (The Weighted Scoring Logic)
# We apply the 40/60 weighting to the normalized components.
master_df['health_score'] = (normalize(master_df['mbu_compliance']) * 40) + \
                            (normalize(master_df['saturation_ratio']) * 60)

# 5. RANK THE DISTRICTS (Aggregated View)
# We take the mean score over the available dates to get a stable performance rank.
district_rankings = master_df.groupby(['state', 'district'])['health_score'].mean().sort_values(ascending=False).reset_index()

# 6. DISPLAY RESULTS
print("--- Top 10 Districts by Health Score (High Performance) ---")
print(district_rankings.head(10))
# --- Top 10 Districts by Health Score (High Performance) ---
#           state                       district  health_score
# 0  Chhattisgarh  Manendragarhchirmiribharatpur     59.427732
# 1  Chhattisgarh                        Raigarh     52.803306
# 2   Maharashtra                     Gadchiroli     52.431304
# 3       Manipur                        Thoubal     52.132293
# 4   Maharashtra                       Yavatmal     51.932698
# 5       Manipur                    Imphal West     51.766200
# 6  Chhattisgarh                    Rajnandgaon     51.688894
# 7  Chhattisgarh                 Janjgir-Champa     51.678170
# 8  Chhattisgarh                       Kawardha     51.451937
# 9   Maharashtra                     Chandrapur     51.190631


print("\n--- Bottom 10 Districts by Health Score (Critical Risk Zones) ---")
print(district_rankings.tail(10))
# --- Bottom 10 Districts by Health Score (Critical Risk Zones) ---
#                   state            district  health_score
# 1085          Karnataka          Ramanagara           0.0
# 1086      Uttar Pradesh         Kushi Nagar           0.0
# 1087          Karnataka     Bengaluru Urban           0.0
# 1088      Uttar Pradesh           Shravasti           0.0
# 1089      Uttar Pradesh     Siddharth Nagar           0.0
# 1090            Gujarat                Dang           0.0
# 1091        West Bengal  24 Paraganas South           0.0
# 1092        West Bengal  24 Paraganas North           0.0
# 1093  Jammu And Kashmir             Shopian           0.0
# 1094        West Bengal          Coochbehar           0.0









from sklearn.preprocessing import MaxAbsScaler
from sklearn.cluster import KMeans

# 1. INITIALIZE SCALER
# MaxAbsScaler preserves sparsity, which is vital for your 0-filled joined data
ma_scaler = MaxAbsScaler()

# 2. CALCULATE DISTRICT HEALTH INDEX (DHI)
# We select features as a list to satisfy the 2D array requirement
dhi_features = ['mbu_compliance', 'saturation_ratio']
scaled_dhi = ma_scaler.fit_transform(master_df[dhi_features])

# Apply the 40/60 weighted logic using the scaled columns
master_df['health_score'] = (scaled_dhi[:, 0] * 40) + (scaled_dhi[:, 1] * 60)

# 3. PREPARE INPUT FOR MACHINE LEARNING
# We aggregate by district first to get behavioral averages
ml_features = ['mbu_compliance', 'saturation_ratio', 'mobility_index', 'late_adopter_ratio']
X_cluster = master_df.groupby(['state', 'district'])[ml_features].mean().reset_index()

# Scale the aggregated data for K-Means
X_scaled = ma_scaler.fit_transform(X_cluster[ml_features])

# 4. TRAIN K-MEANS MODEL
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
X_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

# Map clusters to the profiles you identified earlier
cluster_map = {0: 'Mature Hubs', 1: 'Emerging Zones', 2: 'Policy Risk'}
X_cluster['District_Profile'] = X_cluster['Cluster'].map(cluster_map)

print("✅ Scaling error resolved: MaxAbsScaler applied to DHI and K-Means.")










# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # UNIVARIATE ANALYSIS  :
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BAR GRAPH :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a clean version for plotting
plot_df = master_df[~master_df['district'].isin(['Unknown', 'Other', '100000'])]
import matplotlib.pyplot as plt
import seaborn as sns

# Set a larger figure size so all states fit comfortably
plt.figure(figsize=(10, 15)) 

# By passing the column to 'y', seaborn automatically makes the bars horizontal
sns.countplot(data=master_df, y='state', order=master_df['state'].value_counts().index)

plt.title('Aadhaar Activity Count by State')
plt.xlabel('Count')
plt.ylabel('State')
plt.show()




# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='date')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by date')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='state')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by State')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='district')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by district')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='pincode')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by pincode')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='age_0_5')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by age_0_5')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='age_5_17')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by age_5_17')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='age_18_greater')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by age_18_greater')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='demo_age_5_17')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by demo_age_5_17')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='demo_age_17_')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by demo_age_17_')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='bio_age_5_17')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by bio_age_5_17')
# plt.show()


# plt.figure(figsize=(15, 8))
# # Standard vertical countplot
# sns.countplot(data=master_df, x='bio_age_17_')
# # This is the "straightening" magic: rotate labels 90 degrees
# plt.xticks(rotation=90) 
# plt.title('Aadhaar Activity Count by bio_age_17_')
# plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------








# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PIE CHART :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# master_df['date'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
master_df['state'].value_counts().plot(kind='pie',autopct='%.2f')
plt.show()
# master_df['district'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['pincode'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['age_0_5'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['age_5_17'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['age_18_greater'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['demo_age_5_17'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['demo_age_17_'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['bio_age_5_17'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()
# master_df['bio_age_17_'].value_counts().plot(kind='pie',autopct='%.2f')
# plt.show()




# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HISTOGRAM :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# plt.hist(master_df['age_0_5'])
# plt.show()
# plt.hist(master_df['age_5_17'])
# plt.show()
# plt.hist(master_df['age_18_greater'])
# plt.show()
# plt.hist(master_df['demo_age_5_17'])
# plt.show()
# plt.hist(master_df['demo_age_17_'])
# plt.show()
# plt.hist(master_df['bio_age_5_17'])
# plt.show()
# plt.hist(master_df['bio_age_17_'])  
# plt.show()
plt.hist(master_df['date'])
plt.show()

# After your plotting code, add this line:
plt.xticks(rotation=45, ha='right')
plt.show()

plt.hist(master_df['district'])
plt.show()
plt.hist(master_df['pincode'])
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Displot :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# sns.distplot(master_df['age_0_5'])
# plt.show()
# sns.distplot(master_df['age_5_17'])
# plt.show()
# sns.distplot(master_df['age_18_greater'])
# plt.show()
# sns.distplot(master_df['demo_age_5_17'])
# plt.show()
# sns.distplot(master_df['demo_age_17_'])
# plt.show()
# sns.distplot(master_df['bio_age_5_17'])
# plt.show()
# sns.distplot(master_df['bio_age_17_'])
# plt.show()
sns.distplot(master_df['pincode'])
plt.show()











# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BIVARIATE ANALYSIS  :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 1. Bivariate Analysis: Investigating Key Relationships
# Bivariate analysis helps us understand how one service affects another within the same region.

# A. New Enrolments vs. Demographic Updates (By State)
# This helps identify if states with high growth are also maintaining their data.
# Aggregate data by state for a clean comparison
state_bivariate = master_df.groupby('state')[['total_enrol', 'demo_age_17_']].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.regplot(data=state_bivariate, x='total_enrol', y='demo_age_17_', scatter_kws={'alpha':0.5})
plt.title('Bivariate Analysis: Total Enrolments vs. Adult Demographic Updates')
plt.xlabel('Total New Enrolments')
plt.ylabel('Adult Demographic Updates (17+)')
plt.show()

# Insight: A strong correlation suggests a balanced ecosystem. Outliers (dots far from the line) represent states with "asymmetric activity"—high growth but low maintenance.

# B. The "Youth Gap": age_5_17 vs. bio_age_5_17
# This specifically targets the Mandatory Biometric Update (MBU) trend.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=master_df.sample(10000), x='age_5_17', y='bio_age_5_17', alpha=0.3)
plt.title('Relationship: School-Age Enrolments vs. Biometric Updates')
plt.show()
# Insight: In an ideal scenario, these should move together. A "flat" line here indicates a Service Gap where children are enrolling but not revalidating their biometrics.







# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Approach A  :
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure data is ready
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']

# 1. SETUP VISUALIZATION
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="white")

# 2. CREATE THE JOINTPLOT
g = sns.jointplot(
    data=master_df, 
    x='age_0_5', 
    y='demo_age_17_', 
    kind='hex', 
    cmap='Blues',
    gridsize=25
)

# 3. SET LABELS AND TITLES
# set_axis_labels ensures axis names are properly anchored
g.set_axis_labels('New Infant Enrolments (Age 0-5)', 'Adult Demographic Updates (Contact Proxy)', fontsize=12)

# Use fig.suptitle for figure-level titles; adjust 'y' to prevent overlap
g.fig.suptitle('Approach A: Family Engagement Analysis', fontsize=16, y=1.05)

# 4. FIX CUT-OFF CORNERS
# tight_layout adjusts params so subplots fit the figure area
g.fig.tight_layout() 

# 5. SAVE WITH COMPLETE BOUNDING BOX
# bbox_inches='tight' recomputes the box to include all text
g.savefig('approach_a_complete.png', dpi=300, bbox_inches='tight')
plt.show()








# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Approach B  :
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create the missing column
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']

# Now the scatterplot will work
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=master_df, 
    x='demo_age_17_', 
    y='bio_age_17_', 
    hue='state', 
    size='total_activity', # This will now find the column
    sizes=(20, 200),
    alpha=0.6,
    palette='Set2'
)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. SETUP FIGURE
# A wider figsize helps accommodate a legend on the right
plt.figure(figsize=(14, 8))
sns.set_theme(style="whitegrid")

# 2. CREATE SCATTER PLOT
# 'total_activity' used for size must exist in master_df
scatter = sns.scatterplot(
    data=master_df, 
    x='demo_age_17_', 
    y='bio_age_17_', 
    hue='state', 
    size='total_activity',
    sizes=(30, 300),
    alpha=0.6,
    palette='Set2'
)

# 3. POSITION LEGEND OUTSIDE TO PREVENT OVERLAP
# bbox_to_anchor shifts the legend outside axes boundaries
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title='States')

# 4. TITLES AND LABELS
plt.title('Approach B: Digital (Demo) vs. Physical (Bio) Service Profiles', fontsize=16, pad=20)
plt.xlabel('Adult Demographic Updates (Digital/Contact Proxy)', fontsize=12)
plt.ylabel('Adult Biometric Updates (Physical Authentication Proxy)', fontsize=12)

# 5. FIX CUT-OFFS
# subplots_adjust leaves extra space on the right for the legend
plt.subplots_adjust(right=0.8) 

# 6. SAVE WITH COMPLETE BOUNDING BOX
plt.savefig('approach_b_complete.png', dpi=300, bbox_inches='tight')
plt.show()














# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MULTIVARIATE ANALYSIS  :
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2. Multivariate Analysis: Unlocking Complex Trends
# Multivariate analysis allows you to look at Geography, Time, and Service Volume simultaneously.

# A. The Correlation Heatmap (Feature Relationships)
# This shows how all your variables interact with each other.
plt.figure(figsize=(12, 10))
# Calculate correlation on numeric columns
corr = master_df[['age_0_5', 'age_5_17', 'age_18_greater', 'demo_age_5_17', 
                 'demo_age_17_', 'bio_age_5_17', 'bio_age_17_', 'health_score']].corr()

sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt='.2f')
plt.title('Multivariate Correlation Heatmap: Aadhaar Service Interdependencies')
plt.show()
# Insight: High correlation between age_0_5 and demo_age_17_ might suggest that parents are updating their own Aadhaar details (like mobile numbers) while enrolling their newborns.



# B. Activity Density: State vs. Date vs. Total Activity
# This fulfills the Trivariate requirement by adding the dimension of Time.
# Pivot data for heatmap: States (Y), Dates (X), Total Activity (Color)
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']
# Filter for top 10 states to keep the visual clean
top_10 = master_df.groupby('state')['total_activity'].sum().nlargest(10).index
pivot_df = master_df[master_df['state'].isin(top_10)].pivot_table(
    index='state', columns='date', values='total_activity', aggfunc='sum'
)

plt.figure(figsize=(15, 8))
sns.heatmap(pivot_df, cmap='YlOrRd')
plt.title('Multivariate Temporal Analysis: Service Demand Peaks by State')
plt.show()
# Insight: This reveals "Service Spikes". If a specific date shows a dark red band across multiple states, it might correlate with a national policy change or a deadline for government benefits.


import matplotlib.pyplot as plt
import seaborn as sns

# 1. PREPARE DATA
# We use the columns we created earlier
plt.figure(figsize=(14, 10))

# 2. CREATE MULTIVARIATE BUBBLE CHART
# X: Growth, Y: Maintenance, Size/Color: System Health
scatter = sns.scatterplot(
    data=master_df, 
    x='total_enrol', 
    y='total_updates', 
    hue='health_score', 
    size='health_score',
    sizes=(20, 500), # Sizes based on Health Score
    palette='Spectral',
    alpha=0.6,
    edgecolor='w'
)

# 3. ADD QUADRANT LINES (Means)
plt.axvline(master_df['total_enrol'].mean(), color='grey', linestyle='--', alpha=0.5)
plt.axhline(master_df['total_updates'].mean(), color='grey', linestyle='--', alpha=0.5)

# 4. TITLES AND FORMATTING
plt.title('Multivariate Analysis: Service Maturity Matrix', fontsize=18)
plt.xlabel('Enrolment Volume (Growth)', fontsize=14)
plt.ylabel('Update Volume (Maintenance)', fontsize=14)
plt.legend(title='Health Score', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust limits to focus on the bulk of data (removing extreme outliers)
plt.xlim(0, master_df['total_enrol'].quantile(0.99))
plt.ylim(0, master_df['total_updates'].quantile(0.99))

plt.tight_layout()
plt.savefig('multivariate_maturity_matrix.png', bbox_inches='tight')
plt.show()

 













  

































































































































































# Ensure the plots are readable and high-quality
plt.rcParams['figure.dpi'] = 300
sns.set_theme(style="whitegrid")

# --- VISUALIZATION 1: Box Plot (State-wise Health Distribution) ---
# This shows which states have consistent performance vs. huge district gaps.
plt.figure(figsize=(14, 8))
# Filter for Top 15 states by activity to keep the graph clean
top_states = master_df.groupby('state')['total_enrol'].sum().nlargest(15).index
filtered_df = master_df[master_df['state'].isin(top_states)]

sns.boxplot(data=filtered_df, x='health_score', y='state', palette='coolwarm')

plt.title('Distribution of District Health Scores by State', fontsize=16)
plt.xlabel('Health Score (0-100)', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.savefig('state_health_boxplot.png', bbox_inches='tight')
plt.show()

# --- VISUALIZATION 2: Scatter Plot (Trivariate Analysis) ---
# Variables: 1. Total Enrolment (X), 2. Total Updates (Y), 3. Health Score (Color)
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=master_df, 
    x='total_enrol', 
    y='total_updates', 
    hue='health_score', 
    palette='viridis',
    alpha=0.6,
    edgecolor=None
)

plt.title('Trivariate Analysis: Enrolment vs. Updates vs. Health Score', fontsize=16)
plt.xlabel('Total Enrolments (Service Growth)', fontsize=12)
plt.ylabel('Total Updates (System Maintenance)', fontsize=12)

# Adjusting axes to handle outliers if necessary
plt.xlim(0, master_df['total_enrol'].quantile(0.99))
plt.ylim(0, master_df['total_updates'].quantile(0.99))

plt.savefig('trivariate_performance_scatter.png', bbox_inches='tight')
plt.show()

print("Graphs saved as 'state_health_boxplot.png' and 'trivariate_performance_scatter.png'")