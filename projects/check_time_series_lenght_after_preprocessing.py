import os
import os.path as path
import nibabel as nib

preprocessing_path = '/home/brainlab/Desktop/Rudas/Data/Propofol/Deep/Task/output/datasink/preprocessing'

for subject in sorted(os.listdir(preprocessing_path)):
    subject_path = path.join(preprocessing_path, subject, 'swfmri_art_removed.nii')
    img_data = nib.load(subject_path).get_data()
    print(subject + str(img_data.shape))

