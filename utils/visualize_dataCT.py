import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


WORD_MAP_ID_TO_LABEL = {
    1: "liver",
    2: "spleen",
    3: "left kidney",
    4: "right kidney",
    5: "stomach",
    6: "gallbladder",
    7: "pancreas",
    8: "duodenum",
    9: "colon",
    10: "intestine",
    11: "left adrenal gland",
    12: "rectum",
    13: "bladder",
    14: "right adrenal gland"
}

ABDOMENCT1K_MAP_ID_TO_LABEL = {
    0: "background",
    1: "liver",
    2: "right kidney",
    3: "spleen",
    4: "pancreas",
    5: "left kidney",
}

TOTALSEGMENTATOR_MAP_ID_TO_LABEL = {
    0: "background",
    1: "spleen",
    2: "right kidney",
    3: "left kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "pancreas",
    9: "right adrenal gland",
    10: "left adrenal gland",
    11: "esophagus",
    12: "trachea",
    13: "small bowel",
    14: "duodenum",
    15: "colon",
    16: "urinary bladder",
    17: "heart",
    18: "left lung",
    19: "right lung"
}



def visualize_slice2d(source_folder, dataset_name, data_split=None, label_to_visualize=None, slice_idx=None):
    norm = plt.Normalize(0,19) # Normalize the colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red","orange","yellow","green","blue",
                                                                    "purple","pink","brown","gray","olive","cyan","magenta",
                                                                    "darkred","darkorange","limegreen","darkgreen","darkblue",
                                                                    "gold","chocolate","peachpuff","darkgray","crimson","darkcyan",
                                                                    "darkmagenta"]) # Create a colormap

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(source_folder, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        if files_npy: # If there are .npy files
            if data_split is not None:
                if data_split not in root:
                    continue
            data = np.load(os.path.join(root, "data.npy")) # "data.npy" is the original data
            label = np.load(os.path.join(root, "label.npy")) # "label.npy" is the original label

            if label_to_visualize is None:
                if slice_idx is None:
                    # Select a random slice to visualize
                    slice_final_idx = np.random.randint(0, label.shape[3])
                else:
                    slice_final_idx = slice_idx
            else:
                find_slice = False
                slices_label = []
                for slice in range(label.shape[3]):
                    # Find a slice that contains the label to visualize
                    if label_to_visualize in np.unique(label[0,:,:,slice]):
                        slices_label.append(slice)
                        find_slice = True
                if find_slice:
                    slice_final_idx = slices_label[np.random.randint(len(slices_label))]
                else:
                    print(f"No slice contains label {label_to_visualize} in {root}")
                    continue

            print("===============================================")
            print(f"Visualizing {root}...")
            print(f"Original data shape: {data.shape}")
            print(f"Original label shape: {label.shape}")

            fig, ax = plt.subplots(1,2, figsize=(10,10))
            fig.suptitle(f"Visualizing {label_to_visualize} in {root}")

            # Visualize the original data and label
            ax[0].imshow(data[0, :, :, slice_final_idx], cmap='gray')
            ax[0].set_title(f"Original data - Slice {slice_final_idx}")
            ax[0].axis('off')
            ax[1].imshow(label[0, :, :, slice_final_idx], cmap=cmap, norm=norm)
            ax[1].set_title(f"Original labels - Slice {slice_final_idx}")
            ax[1].axis('off')

            # Make legend for the label
            colors = [cmap(norm(i)) for i in range(0,20)]

            if dataset_name == "WORD":
                labels = [WORD_MAP_ID_TO_LABEL[i] for i in range(1,len(WORD_MAP_ID_TO_LABEL)+1)]
            elif dataset_name == "ABDOMENCT1K":
                labels = [ABDOMENCT1K_MAP_ID_TO_LABEL[i] for i in range(0,len(ABDOMENCT1K_MAP_ID_TO_LABEL))]
            elif dataset_name == "TOTAL_SEGMENTATOR":
                labels = [TOTALSEGMENTATOR_MAP_ID_TO_LABEL[i] for i in range(0,len(TOTALSEGMENTATOR_MAP_ID_TO_LABEL))]

            patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]

            ax[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.show() # Show the visualization




def visualize_orig_transf_slice2d(original_dataset, transformed_dataset, dataset_name, data_split=None, 
                                  label_to_visualize=None, slice_idx=None):
    norm = plt.Normalize(0,19) # Normalize the colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black","red","orange","yellow","green","blue",
                                                                    "purple","pink","brown","gray","olive","cyan","magenta",
                                                                    "darkred","darkorange","limegreen","darkgreen","darkblue",
                                                                    "gold","chocolate","peachpuff","darkgray","crimson","darkcyan",
                                                                    "darkmagenta"]) # Create a colormap

    # Go through all the files in the source folder
    for root, dirs, files in os.walk(original_dataset, topdown=True):
        files_npy = [f for f in files if f.endswith('.npy')] # Get all .npy files
        if files_npy: # If there are .npy files
            if data_split is not None:
                if data_split not in root:
                    continue
            data = np.load(os.path.join(root, "data.npy")) # "data.npy" is the original data
            label = np.load(os.path.join(root, "label.npy")) # "label.npy" is the original label

            if label_to_visualize is None:
                if slice_idx is None:
                    # Select a random slice to visualize
                    slice_idx_final = np.random.randint(label.shape[2])

            else:
                find_slice = False
                slices_label = []
                for slice in range(label.shape[3]):
                    # Find a slice that contains the label to visualize
                    if label_to_visualize in np.unique(label[0,:,:,slice]):
                        slices_label.append(slice)
                        find_slice = True
                if find_slice:
                    slice_idx_final = slices_label[np.random.randint(len(slices_label))]
                else:
                    print(f"No slice contains label {label_to_visualize} in {root}")
                    continue
            
            # Get the same slice from the transformed dataset
            transformed_root = root.replace(original_dataset, transformed_dataset)
            transformed_data = np.load(os.path.join(transformed_root, "data.npy"))
            transformed_label = np.load(os.path.join(transformed_root, "label.npy"))

            if original_dataset == "./original_datasets/word_224_224_160_CT":
                slice_transf_idx = slice_idx_final + 48
            elif original_dataset == "./original_datasets/totalsegmentator_TAPv2_160_160_256_CT":
                slice_transf_idx = slice_idx_final + 0


            print("===============================================")
            print(f"Visualizing {root}...")
            print(f"Original data shape: {data.shape}")
            print(f"Original label shape: {label.shape}")
            print(f"Transformed data shape: {transformed_data.shape}")
            print(f"Transformed label shape: {transformed_label.shape}")

            fig, ax = plt.subplots(2,2, figsize=(10,10))
            fig.suptitle(f"Visualizing {label_to_visualize} in {root}")
            ax[0,0].imshow(data[0, :, :, slice_idx_final], cmap='gray')
            ax[0,0].set_title(f"Original data - Slice {slice_idx_final}")
            ax[0,0].axis('off')
            ax[0,1].imshow(label[0, :, :, slice_idx_final], cmap=cmap, norm=norm)
            ax[0,1].set_title(f"Original labels - Slice {slice_idx_final}")
            ax[0,1].axis('off')
            ax[1,0].imshow(transformed_data[0, :, :, slice_transf_idx], cmap='gray')
            ax[1,0].set_title(f"Transformed data - Slice {slice_transf_idx}")
            ax[1,0].axis('off')
            ax[1,1].imshow(transformed_label[0, :, :, slice_transf_idx], cmap=cmap, norm=norm)
            ax[1,1].set_title(f"Transformed labels - Slice {slice_transf_idx}")
            ax[1,1].axis('off')

            # Make legend for the label
            colors = [cmap(norm(i)) for i in range(0,20)]

            if dataset_name == "WORD":
                labels = [WORD_MAP_ID_TO_LABEL[i] for i in range(1,len(WORD_MAP_ID_TO_LABEL)+1)]
            elif dataset_name == "ABDOMENCT1K":
                labels = [ABDOMENCT1K_MAP_ID_TO_LABEL[i] for i in range(0,len(ABDOMENCT1K_MAP_ID_TO_LABEL))]
            elif dataset_name == "TOTAL_SEGMENTATOR":
                labels = [TOTALSEGMENTATOR_MAP_ID_TO_LABEL[i] for i in range(0,len(TOTALSEGMENTATOR_MAP_ID_TO_LABEL))]

            patches = [matplotlib.patches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
                       
            ax[1,1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

            plt.show() # Show the visualization




if __name__ == "__main__":
    # visualize_slice2d("./original_datasets/word_224_224_160_CT", "WORD", data_split="train", label_to_visualize=None, slice_idx=None)
    # visualize_slice2d("./original_datasets/totalsegmentator_TAPv2_160_160_256_CT", "TOTAL_SEGMENTATOR", data_split="train", label_to_visualize=None, slice_idx=None)
    
    # visualize_slice2d("./original_datasets/abdomenCT-1k_224_224_160_CT", "ABDOMENCT1K", data_split="train", label_to_visualize=None, slice_idx=150)
    # visualize_slice2d("./padded_datasets/abdomenCT-1k_224_224_160_CT", "ABDOMENCT1K", data_split="train", label_to_visualize=None, slice_idx=None)
    # visualize_slice2d("./downsampled_datasets/abdomenCT-1k_64_64_64_CT", "ABDOMENCT1K", data_split="train")

    # visualize_slice2d("./padded_datasets/word_224_224_256_CT", "WORD", data_split="train", label_to_visualize=None, slice_idx=159)
    # visualize_slice2d("./padded_datasets/total_segmentator_224_224_256_CT", "TOTAL_SEGMENTATOR", data_split="train", label_to_visualize=1, slice_idx=None)

    visualize_slice2d("./original_datasets/Hospital_Data_preprocessed", "ABDOMENCT1K", data_split="train", label_to_visualize=1, slice_idx=None)
