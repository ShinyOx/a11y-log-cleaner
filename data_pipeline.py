import pandas as pd
import numpy as np
import os

# temporary script to clean the raw telemetry logs from the user study
# the logging software keeps crashing and leaving garbage rows everywhere
def clean_hci_user_data(file_in, file_out):
    print("loading up the messy data file...", file_in)
    
    # if the file isn't there just make some dummy rows to make sure the script doesn't explode
    if not os.path.exists(file_in):
        print("ugh file not found, making fake logs to test the pipeline...")
        stuff = {
            'user_id': [101, 102, np.nan, 104, 105],
            'latency_ms': [125, -999, 280, 140, -1], # negative numbers are definitely system bugs
            'action_type': ['CLICK', 'SYSTEM_PING', 'DRAG', 'CLICK', 'SYSTEM_PING']
        }
        temp_df = pd.DataFrame(stuff)
        temp_df.to_csv(file_in, index=False)
        data_new = temp_df
    else:
        data_new = pd.read_csv(file_in)
    
    # drop completely blank lines first
    data_new = data_new.dropna(how='all') 
    
    # checking the user_id colums to drop missing sessions
    # TODO: add a proper warning handler here later (i probably won't have time)
    if 'user_id' in data_new.columns:
        data_new = data_new.dropna(subset=['user_id'])
        
    # fixing the negative latency timestmp bug
    # i honestly have no idea why the logger captures negative ms but just clipping it for the presentation
    if 'latency_ms' in data_new.columns:
        # clunky way to convert them but it works for now
        data_new['latency_ms'] = data_new['latency_ms'].apply(lambda x: np.nan if x < 0 else x)
        
        # fill gaps with the median so the statistical models don't error out
        med_val = data_new['latency_ms'].median()
        data_new['latency_ms'] = data_new['latency_ms'].fillna(med_val)
        
    # getting rid of system pings because we only care about real human inputs (clicks, drags)
    if 'action_type' in data_new.columns:
        # print("debug total count before filter:", len(data_new))
        data_new = data_new[data_new['action_type'] != 'SYSTEM_PING']
        
    # export the clean csv
    data_new.to_csv(file_out, index=False)
    print("pipeline finished, exported clean data to:", file_out)

if __name__ == "__main__":
    # testing locally before sending over to the lab pipeline
    clean_hci_user_data("raw_user_logs.csv", "clean_research_ready.csv")
