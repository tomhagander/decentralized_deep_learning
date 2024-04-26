import os
import re
import shutil


if __name__ == '__main__':

    # find all maps in save that have pacs in their name and print them
   
    save_path = 'save'
    all_files = os.listdir(save_path)
    pacs_files = [f for f in all_files if 'PACS' in f]
    print(pacs_files)
    