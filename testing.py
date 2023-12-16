import os

# List of test_suffix values
test_suffix_list = ['_low', '_medium', '_high', '_test1', '_test2']  # Add more suffix values as needed
observation_suffix_list = ['1']
reward_suffix_list = ['3', '4']

# Path to the original script
original_script_path = 'sumoAgent_rand_train_test.py'  # Replace with the actual path

# Function to run the script for a specific test_suffix
def run_script_with_suffix(test_suffix, observation_suffix, reward_suffix):
    command = f'python3 {original_script_path} --test_suffix={test_suffix} --rew_suffix={reward_suffix}'
    os.system(command)

# Iterate through the test_suffix_list and run the script for each value
for test_suffix in test_suffix_list:
    for observation_suffix in observation_suffix_list:
        for reward_suffix in reward_suffix_list:
            run_script_with_suffix(test_suffix, observation_suffix, reward_suffix)
