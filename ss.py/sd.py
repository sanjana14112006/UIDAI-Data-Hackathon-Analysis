
# 1. CREATE RAW METRICS
# We sum up the key 'health' indicators
master_df['total_enrol'] = master_df['age_0_5'] + master_df['age_5_17'] + master_df['age_18_greater']
master_df['total_updates'] = master_df['demo_age_5_17'] + master_df['demo_age_17_'] + \
                             master_df['bio_age_5_17'] + master_df['bio_age_17_']

# 2. CALCULATE COMPONENT RATIOS
# Youth Compliance Ratio: How many MBUs per school-age enrolment
master_df['youth_compliance'] = master_df['bio_age_5_17'] / (master_df['age_5_17'] + 1)

# Maintenance Ratio: Total Updates relative to Total Activity
master_df['maintenance_ratio'] = master_df['total_updates'] / (master_df['total_enrol'] + master_df['total_updates'] + 1)

# 3. NORMALIZE AND CREATE THE HEALTH SCORE (0-100)
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())

# We weight Maintenance slightly higher as it shows system health
master_df['health_score'] = (normalize(master_df['youth_compliance']) * 40) + \
                            (normalize(master_df['maintenance_ratio']) * 60)

# 4. RANK THE DISTRICTS
district_health = master_df.groupby(['state', 'district'])['health_score'].mean().sort_values(ascending=False).reset_index()

print("Top 5 'Healthiest' Districts (High Compliance):")
print(district_health.head(5))
#           state                       district  health_score
# 0  Chhattisgarh  ManendragarhChirmiriBharatpur     59.427732
# 1  Chhattisgarh                        Raigarh     52.803306
# 2   Maharashtra                     Gadchiroli     52.431304
# 3       Manipur                        Thoubal     52.132293
# 4   Maharashtra                       Yavatmal     51.932698

print("\nBottom 5 'At-Risk' Districts (Service Gaps Found):")
print(district_health.tail(5))
#               state            district  health_score
# 1127          Delhi      North East   *           0.0
# 1128  Uttar Pradesh           Shravasti           0.0
# 1129  Uttar Pradesh     Siddharth Nagar           0.0
# 1130    West Bengal  24 Paraganas South           0.0
# 1129  Uttar Pradesh     Siddharth Nagar           0.0
# 1129  Uttar Pradesh     Siddharth Nagar           0.0
# 1130    West Bengal  24 Paraganas South           0.0
# 1131    West Bengal  24 Paraganas North           0.0







# Phase 3: Advanced Feature Engineering (The DHI Model)

# A. MBU COMPLIANCE BY STATE
plt.figure(figsize=(12, 10))
plot_data = master_df.groupby('state')['mbu_compliance'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=plot_data, x='mbu_compliance', y='state', hue='state', palette='magma', legend=False)
plt.axvline(1.0, color='red', linestyle='--', label='Ideal Compliance')
plt.title('National MBU Compliance Map')
plt.show()

# B. MARKET MATURITY MATRIX
plt.figure(figsize=(12, 10))
state_mat = master_df.groupby('state')['saturation_ratio'].mean().sort_values(ascending=False).reset_index()
sns.barplot(data=state_mat, x='saturation_ratio', y='state', hue='state', palette='RdYlGn', legend=False)
plt.axvline(0.8, color='green', label='Mature Market')
plt.axvline(0.3, color='red', label='Emerging Market')
plt.title('State-wise Market Maturity')
plt.show()




# Phase 4: Exploratory Analysis & Strategic Visualization

# 1. DERIVED RAW METRICS
master_df['total_enrol'] = master_df['age_0_5'] + master_df['age_5_17'] + master_df['age_18_greater']
master_df['total_updates'] = master_df['demo_age_5_17'] + master_df['demo_age_17_'] + \
                             master_df['bio_age_5_17'] + master_df['bio_age_17_']
master_df['total_activity'] = master_df['total_enrol'] + master_df['total_updates']

# 2. BEHAVIORAL INDICES
master_df['mbu_compliance'] = master_df['bio_age_5_17'] / (master_df['age_5_17'] + 1)
master_df['saturation_ratio'] = master_df['total_updates'] / (master_df['total_activity'] + 1)
master_df['mobility_index'] = master_df['demo_age_17_'] / (master_df['age_18_greater'] + master_df['demo_age_17_'] + 1)
master_df['late_adopter_ratio'] = master_df['age_18_greater'] / (master_df['total_enrol'] + 1)

# 3. CALCULATE DISTRICT HEALTH INDEX (DHI)
scaler = MinMaxScaler()
master_df['norm_mbu'] = scaler.fit_transform(master_df[['mbu_compliance']])
master_df['norm_sat'] = scaler.fit_transform(master_df[['saturation_ratio']])
master_df['health_score'] = (master_df['norm_mbu'] * 40) + (master_df['norm_sat'] * 60)



# Phase 5: Towards Model Building (Unsupervised Clustering)

from sklearn.cluster import KMeans

# 1. PREPARE MODELING FEATURES
features = ['mbu_compliance', 'saturation_ratio', 'mobility_index', 'late_adopter_ratio']
X = master_df.groupby(['state', 'district'])[features].mean().dropna()

# 2. SIMPLE CLUSTERING (Identifying 3 Types of Districts)
kmeans = KMeans(n_clusters=3, random_state=42)
X['Cluster'] = kmeans.fit_predict(scaler.fit_transform(X))

# Cluster 0: High Update/Mature, Cluster 1: High Growth/Emerging, Cluster 2: Low Compliance/At-Risk
print("--- Cluster Profiles ---")
print(X.groupby('Cluster').mean())

# 3. EXPORT FINAL RANKINGS
district_rankings = master_df.groupby(['state', 'district'])['health_score'].mean().sort_values(ascending=False).reset_index()
district_rankings.to_csv('UIDAI_District_Health_Report.csv', index=False)




# # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# # Pandas Profiling :
# # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# import pandas as pd
# from pandas_profiling import ProfileReport
# prof = ProfileReport(master_df)
# prof.to_file(output_file='output.html')

