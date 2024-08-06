import numpy as np 
import matplotlib
from matplotlib import pyplot as plt
import os
import nibabel as nib
import nrrd

HOSPITAL_DATA_ID_TO_LABEL = {
    0: "background",
    1: "liver (1)",
    2: "left kidney (2)",
    3: "spleen (3)",
    4: "pancreas (4)",
    5: "right kidney (5)",
}

def check_labels(source_folder, output_folder, dataset_name):
    """
    Go through all the files in the source folder and count unique labels
    Labels are in format .nii.gz
    We are in the files with name segm_combined.nii.gz
    """
    
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Dictionary to store the unique labels
    unique_labels = {}
    
    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):
        
        # Get all .nii.gz files
        # files_nii = [f for f in files if f.endswith('.nii.gz') or f.endswith('.nrrd')]
        files_nii = [f for f in files if f.endswith('.nii.gz')]

        # For each .nii.gz file
        for file in files_nii:
            
            # Load the .nii.gz file
            filepath = os.path.join(root, file)
            data = nib.load(filepath).get_fdata()

            # In filepath take path from segmentations/ (not included) till the end
            path_file_name = os.path.relpath(filepath, source_folder)

            output_folder_path = os.path.join(output_folder, path_file_name)

            # Change the data name file from segm_combined.nii.gz to data.npy
            os.makedirs(os.path.dirname(output_folder_path), exist_ok=True)

            # Update data values if an island is detected
            data_updated = data.copy()

            for i in range(data_updated.shape[2]):
                data_updated[:,:,i] = change_value_island(data_updated[:,:,i], 2, 5)


            np.save(output_folder_path, data_updated)
            
            # If the file is "segm_combined.nii.gz"
            if file == 'segm_combined.nii.gz':
                
                # Get the unique labels and their counts
                unique, counts = np.unique(data_updated, return_counts=True)
                print(f"Path: {filepath}")  
                print(f"Unique labels: {unique}")
                print(f"Shape: {data_updated.shape}")

                # Store the unique labels and their counts in the dictionary
                for i in range(len(unique)):
                    if unique[i] in unique_labels:
                        unique_labels[unique[i]] += counts[i]
                    else:
                        unique_labels[unique[i]] = counts[i]
            
    # Plot the unique labels and their counts
    plt.figure(figsize=(10, 5))
    plt.bar(unique_labels.keys(), unique_labels.values())
    plt.xlabel('Labels')
    plt.ylabel('Counts')
    plt.title(f'Unique Labels for {dataset_name}')
    # plt.savefig(os.path.join(output_folder, f"{dataset_name}_unique_labels.png"))
    plt.show()
    
    return unique_labels


def change_value_island(data, initial_value, target_value, detected_island=False):
    """
    Change the value of the island to a new value
    """
    # Detect in data values that are equal to initial_value
    mask = data == initial_value
    mask_updated = np.zeros_like(mask)

    # Detect the island
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] == True:
                mask = flood_fill(mask, mask_updated, i, j)
                detected_island = True
                break

    if detected_island:
        data[mask] = target_value

    return data

def flood_fill(mask, mask_updated, x, y):
    """
    Flood fill algorithm
    """

    if x < 0 or x >= mask_updated.shape[0] or y < 0 or y >= mask_updated.shape[1]:
        return mask_updated
    
    if mask[x,y] == True:
        mask_updated[x, y] = True

    stack = [(x, y)]
    while stack:
        x, y = stack.pop()
        mask_updated[x, y] = True
        if x > 0 and mask[x-1, y] == True and mask_updated[x-1, y] == False:
            stack.append((x-1, y))
            mask_updated[x-1, y] = True
        if x < mask_updated.shape[0] - 1 and mask[x+1, y] == True and mask_updated[x+1, y] == False:
            stack.append((x+1, y))
            mask_updated[x+1, y] = True
        if y > 0 and mask[x, y-1] == True and mask_updated[x, y-1] == False:
            stack.append((x, y-1))
            mask_updated[x, y-1] = True
        if y < mask_updated.shape[1] - 1 and mask[x, y+1] == True and mask_updated[x, y+1] == False:
            stack.append((x, y+1))
            mask_updated[x, y+1] = True

     

    return mask_updated


def visualize_hospital_data(source_folder, number_labels):
    """
    Visualize the hospital data
    """
    norm = plt.Normalize(0,19) # Normalize the colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red","orange","yellow","green","blue",
                                                                    "purple","pink","brown","gray","olive","cyan","magenta",
                                                                    "darkred","darkorange","limegreen","darkgreen","darkblue",
                                                                    "gold","chocolate","peachpuff","darkgray","crimson","darkcyan",
                                                                    "darkmagenta"]) # Create a colormap
    
    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):
        
        # Get all .nii.gz files
        files_nii = [f for f in files if f.endswith('.npy')]

        # For each .nii.gz file
        for file in files_nii:
            
            # Load the .nii.gz file
            data = np.load(os.path.join(root, file))

            list_slices_labels = []
            for i in range(data.shape[2]):
                if len(np.unique(data[:,:,i])) -1 >= number_labels:
                    list_slices_labels.append(i)

            slice_random = np.random.choice(list_slices_labels)
                        
            # See data in matplotlib
            plt.figure()
            plt.imshow(data[:,:,slice_random], cmap=cmap, norm=norm)
            plt.title(f"Patient: {root.split('/')[-1]} - Slice: {slice_random}")

            # Make legend for each label
            labels = np.unique(data)
            colors = [cmap(norm(label)) for label in labels]
            patches = [matplotlib.patches.Patch(color=colors[i], label=HOSPITAL_DATA_ID_TO_LABEL[labels[i]]) for i in range(len(labels))]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


            plt.show()


if __name__ == "__main__":
    # source_folder = "./original_datasets/Hospital_Data/segmentations"
    # output_folder = "./original_datasets/Hospital_Data_preprocessed/labels"
    # dataset_name = "Hospital_Data"
    
    # unique_labels = check_labels(source_folder, output_folder, dataset_name)
    # print(unique_labels)

    source_folder = "./original_datasets/Hospital_Data_preprocessed/labels"
    visualize_hospital_data(source_folder, number_labels=5)