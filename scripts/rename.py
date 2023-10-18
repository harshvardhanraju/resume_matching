import os

# Specify the folder path and the base string
folder_path = "g:\jd_Cv\Profiles-20231016T064917Z-001\Profiles\Vmware Admin"  # Change to the actual folder path
base_string = "Vmware Admin"

# Function to rename files
def rename_files_in_folder(folder_path, base_string):
    # Ensure the folder path is valid
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    # Get a list of files in the folder
    files = os.listdir(folder_path)

    # Initialize a counter to keep track of the file number
    file_number = 1

    # Iterate through the files and rename them
    for file_name in files:
        # Construct the new file name based on the format "BASE_resume_num"
        new_file_name = f"{base_string}_resume_{file_number}"
        file_extension = os.path.splitext(file_name)[1]  # Get the file extension
        new_file_name_with_extension = f"{new_file_name}{file_extension}"

        # Construct the full file paths
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(folder_path, new_file_name_with_extension)

        # Rename the file
        os.rename(old_file_path, new_file_path)

        # Increment the file number for the next file
        file_number += 1

    print(f"Renamed {file_number - 1} files in '{folder_path}'.")

# Call the function to rename files in the specified folder
rename_files_in_folder(folder_path, base_string)
