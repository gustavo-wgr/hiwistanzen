import pickle
import os
import shutil
import numpy as np

def format():
    # Path to the original pickle file
    original_pickle_path = "synth_data/generated_data.pkl"

    # Directory to save the individual pickle files
    output_dir = "synth_data/individual_samples"

    # The .txt file that will list the paths of the created pickle files
    paths_txt_file = "synth_data/sample_paths.txt"

    # Remove existing folder and text file if they already exist
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if os.path.exists(paths_txt_file):
        os.remove(paths_txt_file)

    # Recreate the folder for the new pickle files
    os.makedirs(output_dir, exist_ok=True)

    # Define the mapping from numeric labels to class names
    label_to_class = {
        0: 'R05',
        1: 'R10',
        2: 'R15',
        3: 'R20',
        4: 'R25',
        5: 'R30',
        6: 'R35',
        7: 'R40',
        8: 'R45',
        9: 'R50'
    }

    # Load the original data
    with open(original_pickle_path, "rb") as f:
        data = pickle.load(f)

    inputs = data["inputs"]  # shape: (100, 2800)
    labels = data["labels"]  # could be shape: (100,) or (100, num_classes)

    # Open the text file in write mode so we can store the paths
    with open(paths_txt_file, "w") as txt_f:
        # Iterate over each sample
        for i in range(len(inputs)):
            single_input = inputs[i]      # shape: (2800,)
            
            # Check if the label is already one-hot encoded (i.e. an array)
            # If so, get the numeric label by argmax; otherwise, convert directly.
            if isinstance(labels[i], np.ndarray) and np.ndim(labels[i]) > 0:
                numeric_label = int(np.argmax(labels[i]))
            else:
                numeric_label = int(labels[i])
            
            # Use the mapping to get the class name
            class_name = label_to_class[numeric_label]
            
            # Create one-hot encoding from the numeric label (assuming 10 classes)
            one_hot_label = np.zeros(10, dtype=np.float32)
            one_hot_label[numeric_label] = 1
            
            # Construct the filename using the class name for reference.
            filename = f"{class_name}_{i}.pkl"
            output_path = os.path.join(output_dir, filename)
            abs_output_path = os.path.abspath(output_path)

            # Save both the input and the one-hot encoded label to the pickle file.
            with open(output_path, "wb") as out_f:
                pickle.dump(single_input, out_f)

            # Write the absolute path to the text file
            txt_f.write(abs_output_path + "\n")