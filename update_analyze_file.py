import os
import shutil

save_folder = 'save'
current_directory = os.getcwd()

for folder in os.listdir(save_folder):
    if os.path.isdir(os.path.join(save_folder, folder)):
        file_to_replace = 'analyze_' + folder + '.ipynb'
        source_file = os.path.join(current_directory, 'analyze_data.ipynb')
        destination_file = os.path.join(save_folder, folder, file_to_replace)
        
        if os.path.exists(destination_file):
            os.remove(destination_file)
        
        shutil.copy(source_file, destination_file)