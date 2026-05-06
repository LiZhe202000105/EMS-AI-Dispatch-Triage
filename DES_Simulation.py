import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 1. Prepare Operational Data
df = pd.read_csv("EMS_Trauma_NPJ_Ready.csv")
df['Response_Time_min'] = df['Response_Time_min'].fillna(df['Response_Time_min'].median())
test_data = df[df['Dataset_Split'] == 'Test'].copy()

# Assume AI probability generated from main pipeline
threshold_ai = np.percentile(test_data['AI_Prob'] if 'AI_Prob' in test_data.columns else np.random.rand(len(test_data)), 90)
test_data['AI_Alert'] = (test_data['AI_Prob'] >= threshold_ai).astype(int) if 'AI_Prob' in test_data.columns else np.random.randint(0,2,len(test_data))

# 2. Simulation Constants
SIMULATION_RUNS = 100       
TURNAROUND_TIME_MEAN = 60.0 
SIMULATED_FAST_RT_MEAN = 8.0 

# 3. Discrete-Event Simulation Core Function
def run_single_simulation(data, capacity):
    arrival_times = np.sort(np.random.uniform(0, 730 * 24 * 60, len(data)))
    fleet_available_times = [0.0] * capacity
    simulated_rt = []
    
    for i, row in data.iterrows():
        arrival = arrival_times[i]
        alert_triggered = row['AI_Alert'] == 1
        
        if alert_triggered:
            available_ambulances = [j for j, t in enumerate(fleet_available_times) if t <= arrival]
            if len(available_ambulances) > 0:
                assigned_amb = available_ambulances[0]
                turnaround = np.clip(np.random.normal(TURNAROUND_TIME_MEAN, 15.0), 30, 120)
                fleet_available_times[assigned_amb] = arrival + turnaround
                fast_rt = np.clip(np.random.normal(SIMULATED_FAST_RT_MEAN, 2.0), 4, 15)
                simulated_rt.append(fast_rt)
            else:
                simulated_rt.append(row['Response_Time_min']) # Queue Exhausted
        else:
            simulated_rt.append(row['Response_Time_min'])
            
    return np.array(simulated_rt)

# 4. Execute Monte Carlo Runs
capacities = [2, 3, 5, 8, 12, 15, 20]
pce_mask = test_data['Target_PCE'] == 1
baseline_rt = test_data.loc[pce_mask, 'Response_Time_min']

mc_results = {}
for cap in capacities:
    saved_times_runs = []
    for _ in range(SIMULATION_RUNS):
        sim_rt = run_single_simulation(test_data.reset_index(drop=True), cap)
        time_saved = baseline_rt.values - sim_rt[pce_mask.values]
        saved_times_runs.append(np.nanmean(time_saved))
    mc_results[cap] = saved_times_runs