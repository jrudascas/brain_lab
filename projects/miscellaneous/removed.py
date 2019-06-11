import os

folder_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Recovery/Resting/output'

cont = 1
for (root, dirs, files) in os.walk(folder_path):
    for file in sorted(files):
        if file == 'fmri_cleaned.nii':
            try:
                os.remove(root + '/' + file)
            except Exception:
                continue