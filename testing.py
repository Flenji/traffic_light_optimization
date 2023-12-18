import os
import xmltodict
import xml.etree.ElementTree as ET



# Function to run the script for a specific test_suffix
def run_script_with_suffix(test_suffix, observation_suffix, reward_suffix, seed):
    command = f'python3 {original_script_path} --test_suffix={test_suffix} --rew_suffix={reward_suffix} --obs_suffix={observation_suffix} --seed={seed}'
    #command = f'python3 {original_script_path} --test_suffix={test_suffix} --rew_suffix={reward_suffix} --seed={seed}'
    os.system(command)

def compute_average_metrics(filenames, metrics_name):
    # Step 2: Aggregate Data
    aggregate_data = {}

    # Step 3: Sum Values
    for xml_file in filenames:
        with open(xml_file, "r") as file:
            xml_content = file.read()
            xml_dict = xmltodict.parse(xml_content)

            for edge_data in xml_dict["meandata"]["interval"]["edge"]:
                edge_id = edge_data["@id"]
                if edge_id not in aggregate_data:
                    aggregate_data[edge_id] = {
                        "traveltime": 0,
                        "density": 0,
                        "occupancy": 0,
                        "waitingTime": 0,
                        "timeLoss": 0,
                        "speed": 0,
                    }

                # Sum values for each edge
                aggregate_data[edge_id]["traveltime"] += float(edge_data["@traveltime"])
                if "@density" in edge_data:
                    aggregate_data[edge_id]["density"] += float(edge_data["@density"])
                if "@occupancy" in edge_data:
                    aggregate_data[edge_id]["occupancy"] += float(edge_data["@occupancy"])
                if "@waitingTime" in edge_data:
                    aggregate_data[edge_id]["waitingTime"] += float(edge_data["@waitingTime"])
                if "@timeLoss" in edge_data:
                    aggregate_data[edge_id]["timeLoss"] += float(edge_data["@timeLoss"])
                aggregate_data[edge_id]["speed"] += float(edge_data["@speed"])
        os.remove(xml_file)

    # Step 4: Create Aggregated XML
    aggregated_xml = {
        "meandata": {
            "@xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "@xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/meandata_file.xsd",
            "interval": {
                "@begin": "0.00",
                "@end": "3600.00",
                "@id": "edgeData",
                "edge": [],
            },
        }
    }

    for edge_id, edge_values in aggregate_data.items():
        aggregated_xml["meandata"]["interval"]["edge"].append(
            {
                "@id": edge_id,
                "@traveltime": str(edge_values["traveltime"]/5),
                "@density": str(edge_values["density"]/5),
                "@occupancy": str(edge_values["occupancy"]/5),
                "@waitingTime": str(edge_values["waitingTime"]/5),
                "@timeLoss": str(edge_values["timeLoss"]/5),
                "@speed": str(edge_values["speed"]/5),
            }
        )

    # Convert aggregated XML dictionary to string
    aggregated_xml_string = xmltodict.unparse(aggregated_xml, pretty=True)

    # Save the aggregated XML to a file
    with open(os.path.join("./metrics", metrics_name), "w") as result_file:
        result_file.write(aggregated_xml_string)

### Specific-trained agents

# List of test_suffix values
seeds = [11, 980, 450, 24, 111]
test_suffix_list = ['_test1', '_test2']  # Add more suffix values as needed
observation_suffix_list = ['1', '2']
reward_suffix_list = ['3', '4']
foldername = "./metrics"

# Path to the original script
original_script_path = 'sumoAgent_train_test.py'  # Replace with the actual path

# Iterate through the test_suffix_list and run the script for each value
for test_suffix in test_suffix_list:
    for observation_suffix in observation_suffix_list:
        for reward_suffix in reward_suffix_list:
            for seed in seeds:
                run_script_with_suffix(test_suffix, observation_suffix, reward_suffix, seed)
            metrics_name = "metrics_1w_obs"+observation_suffix+"_rew"+reward_suffix+test_suffix+".xml"
            filenames = [os.path.join(foldername, file) for file in os.listdir(foldername) if file.endswith(".xml") 
                         and "_rew"+reward_suffix in file and "_obs"+observation_suffix in file and test_suffix in file]
            compute_average_metrics(filenames, metrics_name)
"""
### Random-trained agents

# List of test_suffix values
seeds = [11, 980, 450, 24, 111]
test_suffix_list = ['_low', '_medium', '_high', '_test1', '_test2']  # Add more suffix values as needed
observation_suffix_list = ['2']
reward_suffix_list = ['3', '4']
foldername = "./metrics"

# Path to the original script
original_script_path = 'sumoAgent_rand_train_test.py'  # Replace with the actual path

# Iterate through the test_suffix_list and run the script for each value
for test_suffix in test_suffix_list:
    for observation_suffix in observation_suffix_list:
        for reward_suffix in reward_suffix_list:
            for seed in seeds:
                run_script_with_suffix(test_suffix, observation_suffix, reward_suffix, seed)
            metrics_name = "metrics_reward"+reward_suffix+"_randtraining"+test_suffix+".xml"
            filenames = [os.path.join(foldername, file) for file in os.listdir(foldername) if file.endswith(".xml") 
                         and "_reward"+reward_suffix in file and test_suffix in file]
            compute_average_metrics(filenames, metrics_name)

"""
### Static-TLS
"""
# List of test_suffix values
seeds = [11, 980, 450, 24, 111]
test_suffix_list = ['_low', '_medium', '_high', '_test1', '_test2']  # Add more suffix values as needed
foldername = "./metrics"
for test_suffix in test_suffix_list:
    for seed in seeds:
        # Specify the path to your XML file
        xml_file_path = 'additional.xml'

        # Read the XML content from the file
        with open(xml_file_path, 'r') as file:
            xml_content = file.read()

        # Parse the XML content
        root = ET.fromstring(xml_content)

        # Find the specific element
        edge_data_element = root.find(".//edgeData[@id='edgeData']")

        # Modify the 'file' attribute
        edge_data_element.set('file', 'metrics/metrics_static'+test_suffix+'_'+str(seed)+'.xml')

        # Convert the modified XML back to a string
        modified_xml_content = ET.tostring(root).decode()

        # Save the modified content back to the file
        with open(xml_file_path, 'w') as file:
            file.write(modified_xml_content)

        command = f'sumo -b 0 -e 3600 -n Networks/single_agent_networks/1w/1w.net.xml -r Networks/single_agent_networks/1w/1w{test_suffix}.rou.xml -a additional.xml --seed {seed}'
        os.system(command)
    metrics_name = "metrics_static"+test_suffix+".xml"
    filenames = [os.path.join(foldername, file) for file in os.listdir(foldername) if file.endswith(".xml") 
                and "static" in file and test_suffix in file]
    compute_average_metrics(filenames, metrics_name)
"""