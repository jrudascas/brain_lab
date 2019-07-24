from os.path import join as opj
from nipype.interfaces.fsl import (BET, ExtractROI, FAST, FLIRT, MCFLIRT, SliceTimer, Threshold)
from .interfaces.ExtractConfounds import ExtractConfounds
from .interfaces.SignalExtraction import SignalExtraction
from .interfaces.ArtifacRemotion import ArtifacRemotion
from .interfaces.N4Bias import N4Bias
from .interfaces.Reslicing import Reslicing
from .interfaces.Descomposition import Descomposition
from .interfaces.ExtractB0 import ExtractB0
from .interfaces.Denoise import Denoise
from .interfaces.ModelDTI import ModelDTI
from .interfaces.Tractography import Tractography
from .interfaces.MedianOtsu import MedianOtsu
from .interfaces.Registration import Registration
from .interfaces.RegistrationAtlas import RegistrationAtlas
from nipype.interfaces.fsl import Eddy
from nipype.interfaces.fsl import EddyCorrect
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.algorithms.rapidart import ArtifactDetect
from nipype import Workflow, Node
from nipype.interfaces.spm import Normalize12
from nipype.algorithms.misc import Gunzip
from .tools.utils import *

import nipype.interfaces.spm as spm
import os


class Pipeline(object):

    # Initializer / Instance Attributes
    def __init__(self, paths, parameters, subject_list):
        self.paths = paths
        self.parameters = parameters
        self.subject_list = subject_list

    def run(self):
        raise NotImplementedError


class PipelineDWI(Pipeline):

    def run(self):
        experiment_dir = opj(self.paths['input_path'], 'output/')
        output_dir = 'datasink'
        working_dir = 'workingdir'

        subject_list = self.subject_list
        iso_size = self.parameters['iso_size']

        # Infosource - a function free node to iterate over the list of subject names
        infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
        infosource.iterables = [('subject_id', subject_list)]

        # SelectFiles - to grab the data (alternativ to DataGrabber)
        anat_file = opj('{subject_id}/data/T1w/', 'T1w_acpc_dc_restore_1.25.nii.gz')
        dwi_file = opj('{subject_id}/data/T1w/Diffusion/', 'data.nii.gz')
        bvec_file = opj('{subject_id}/data/T1w/Diffusion/', 'bvecs')
        bval_file = opj('{subject_id}/data/T1w/Diffusion/', 'bvals')

        templates = {'anat': anat_file,
                     'dwi': dwi_file,
                     'bvec': bvec_file,
                     'bval': bval_file}

        selectfiles = Node(SelectFiles(templates, base_directory=self.paths['input_path']), name="selectfiles")

        # Datasink - creates output folder for important outputs
        datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")

        substitutions = [('_subject_id_', 'sub-')]

        datasink.inputs.substitutions = substitutions

        preproc = Workflow(name='preproc')
        preproc.base_dir = opj(experiment_dir, working_dir)

        # BET - Skullstrip anatomical anf funtional images
        bet_t1 = Node(BET(frac=0.5, robust=True, mask=True, output_type='NIFTI_GZ'), name="bet_t1") # FSL

        denoise_t1 = Node(Denoise(), name="denoising_t1") # Dipy

        reslicing = Node(Reslicing(vox_sz=iso_size), name="reslicing") #Dipy

        #registration_atlas = Node(RegistrationAtlas(reference=self.paths['reference'], atlas_to_apply=self.paths['image_parcellation_path']), name="registration_atlas")
        registration_atlas = Node(RegistrationAtlas(reference=self.paths['reference']), name="registration_atlas")
        registration_atlas.iterables = [('atlas_to_apply', self.paths['image_parcellation_path'])]

        #registration_t1 = Node(Registration(reference=self.paths['reference']), name="registration_t1")

        #registration_dwi = Node(Registration(reference='/home/brainlab/Desktop/Rudas/Data/Parcellation/MNI152_T2_2mm.nii.gz'), name="registration_dwi")

        tractography = Node(Tractography(), name='tractography') # Dipy

        model_dti = Node(ModelDTI(), name="model_dti") # Dipy

        denoise_dwi = Node(Denoise(), name="denoising_dwi") # Dipy

        extract_b0 = Node(ExtractB0(), name="extract_b0")

        n4bias = Node(N4Bias(out_file='t1_n4bias.nii.gz'), name='n4bias') # SimpeITK

        eddycorrection = Node(EddyCorrect(ref_num = 0), 'eddycorrection') # FSL

        median_otsu = Node(MedianOtsu(), 'median_otsu') # Dipy

        '''
        normalize_t1 = Node(Normalize12(jobtype='estwrite',
                                        tpm=self.paths['template_spm_path'],
                                        write_voxel_sizes=[iso_size, iso_size, iso_size],
                                        write_bounding_box=[[-90, -126, -72], [90, 90, 108]]),
                            name="normalize_t1")

        normalize_masks = Node(Normalize12(jobtype='estwrite',
                                           tpm=self.paths['template_spm_path'],
                                           write_voxel_sizes=[iso_size, iso_size, iso_size],
                                           write_bounding_box=[[-90, -126, -72], [90, 90, 108]]),
                               name="normalize_masks")
        
        # FAST - Image Segmentation
        segmentation = Node(FAST(output_type='NIFTI'), name="segmentation")

        # FLIRT - pre-alignment of functional images to anatomical images
        coreg_pre = Node(FLIRT(dof=6, output_type='NIFTI_GZ'), name="linear_warp_estimation")

        # Threshold - Threshold WM probability image
        threshold = Node(Threshold(thresh=0.5, args='-bin', output_type='NIFTI_GZ'), name="wm_mask_threshold")

        gunzip1 = Node(Gunzip(), name="gunzip1")
        gunzip2 = Node(Gunzip(), name="gunzip2")
        '''

        # Create a coregistration workflow
        coregwf = Workflow(name='coreg_fmri_to_t1')
        coregwf.base_dir = opj(experiment_dir, working_dir)

        # FLIRT - coregistration of functional images to anatomical images with BBR
        coreg_bbr = Node(FLIRT(dof=6, cost='bbr', schedule=opj(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch'),
                               output_type='NIFTI_GZ'),
                         name="nonlinear_warp_estimation")

        # Apply coregistration warp to functional images
        applywarp = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI'),
                         name="registration_dwi")

        applywarp_mean = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI_GZ'),
                              name="registration_mean_b0")

        # Connect all components of the coregistration workflow
        '''
        coregwf.connect([(denoise_t1, bet_t1, [('out_file', 'in_file')]),
                         (bet_t1, n4bias, [('out_file', 'in_file')]),
                         (n4bias, segmentation, [('out_file', 'in_files')]),
                         (segmentation, threshold, [(('partial_volume_files', get_latest), 'in_file')]),
                         (n4bias, coreg_pre, [('out_file', 'reference')]),
                         (threshold, coreg_bbr, [('out_file', 'wm_seg')]),
                         (coreg_pre, coreg_bbr, [('out_matrix_file', 'in_matrix_file')]),
                         (coreg_bbr, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                         (n4bias, applywarp, [('out_file', 'reference')]),
                         (coreg_bbr, applywarp_mean, [('out_matrix_file', 'in_matrix_file')]),
                         (n4bias, applywarp_mean, [('out_file', 'reference')]),
                         ])
        '''

        # Connect all components of the preprocessing workflow
        preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                         #(selectfiles, coregwf, [('anat', 'denoising_t1.in_file'),
                         #                        ('anat', 'nonlinear_warp_estimation.reference')]),
                         #(selectfiles, extract_b0, [('dwi', 'dwi_path'), ('bval', 'bval_path'), ('bvec', 'bvec_path')]),
                         #(extract_b0, coregwf, [('out_file', 'linear_warp_estimation.in_file'),
                         #                    ('out_file', 'nonlinear_warp_estimation.in_file'),
                         #                    ('out_file', 'registration_mean_b0.in_file')]),
                         #(selectfiles, coregwf, [('dwi', 'registration_dwi.in_file')]),
                         #(coregwf, eddycorrection, [('registration_dwi.out_file', 'in_file')]),
                         #(eddycorrection, denoise_dwi, [('eddy_corrected', 'in_file')]),
                         #(denoise_dwi, median_otsu, [('out_file', 'in_file')]),

                         (selectfiles, denoise_t1, [('anat', 'in_file')]),
                         (denoise_t1, bet_t1, [('out_file', 'in_file')]),
                         (bet_t1, n4bias, [('out_file', 'in_file')]),

                         (selectfiles, eddycorrection, [('dwi', 'in_file')]),
                         (eddycorrection, reslicing, [('eddy_corrected', 'in_file')]),
                         (reslicing, denoise_dwi, [('out_file', 'in_file')]),
                         (denoise_dwi, median_otsu, [('out_file', 'in_file')]),

                         (median_otsu, extract_b0, [(('out_file', get_first), 'in_file')]),
                         (selectfiles, extract_b0, [('bval', 'bval_path'), ('bvec', 'bvec_path')]),
                         (extract_b0, registration_atlas, [('out_file', 'image_to_align')]),
                         (median_otsu, model_dti, [(('out_file', get_first), 'in_file'), (('out_file', get_latest), 'mask_file')]),
                         (selectfiles, model_dti, [('bval', 'bval_path'), ('bvec', 'bvec_path')]),
                         (median_otsu, tractography, [(('out_file', get_first), 'in_file'), (('out_file', get_latest), 'mask_file')]),
                         (registration_atlas,tractography,[('out_file', 'image_parcellation_path')]),
                         (selectfiles, tractography, [('bval', 'bval_path'), ('bvec', 'bvec_path')])
                         ])
        preproc.run()


class PipelinefMRI(Pipeline):

    def run(self):
        matlab_cmd = self.paths['spm_path'] + ' ' + self.paths['mcr_path'] + '/ script'
        spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

        print('SPM version: ' + str(spm.SPMCommand().version))

        experiment_dir = opj(self.paths['input_path'], 'output/')
        output_dir = 'datasink'
        working_dir = 'workingdir'

        subject_list = self.subject_list

        # list of subject identifiers
        fwhm = self.parameters['fwhm']  # Smoothing widths to apply (Gaussian kernel size)
        tr = self.parameters['tr']  # Repetition time
        init_volume = self.parameters['init_volume']  # Firts volumen identification which will use in the pipeline
        iso_size = self.parameters['iso_size']  # Isometric resample of functional images to voxel size (in mm)
        low_pass = self.parameters['low_pass']
        high_pass = self.parameters['high_pass']

        # ExtractROI - skip dummy scans
        extract = Node(ExtractROI(t_min=init_volume, t_size=-1, output_type='NIFTI'), name="extract") #FSL

        # MCFLIRT - motion correction
        mcflirt = Node(MCFLIRT(mean_vol=True, save_plots=True, output_type='NIFTI'), name="motion_correction") #FSL

        # SliceTimer - correct for slice wise acquisition
        slicetimer = Node(SliceTimer(index_dir=False, interleaved=True, output_type='NIFTI', time_repetition=tr),
                          name="slice_timing_correction") #FSL

        # Smooth - image smoothing

        denoise = Node(Denoise(), name="denoising") #Interfaces with dipy

        smooth = Node(spm.Smooth(fwhm=fwhm), name="smooth") #SPM

        n4bias = Node(N4Bias(out_file='t1_n4bias.nii.gz'), name='n4bias') #Interface with SimpleITK

        descomposition = Node(Descomposition(n_components=20, low_pass=0.1, high_pass=0.01, tr=tr), name='descomposition') #Interface with nilearn

        # Artifact Detection - determines outliers in functional images
        art = Node(ArtifactDetect(norm_threshold=2,
                                  zintensity_threshold=3,
                                  mask_type='spm_global',
                                  parameter_source='FSL', use_differences=[True, False], plot_type='svg'),
                   name="artifact_detection") #Rapidart

        extract_confounds_ws_csf = Node(ExtractConfounds(out_file='ev_without_gs.csv'), name='extract_confounds_ws_csf') #Interfece

        extract_confounds_gs = Node(ExtractConfounds(out_file='ev_with_gs.csv',
                                                     delimiter=','),
                                    name='extract_confounds_global_signal')

        signal_extraction = Node(SignalExtraction(time_series_out_file='time_series.csv',
                                                  correlation_matrix_out_file='correlation_matrix.png',
                                                  labels_parcellation_path=self.paths['labels_parcellation_path'],
                                                  mask_mni_path=self.paths['mask_mni_path'],
                                                  tr=tr,
                                                  low_pass=low_pass,
                                                  high_pass=high_pass,
                                                  plot=False),
                                 name='signal_extraction')
        signal_extraction.iterables = [('image_parcellation_path', self.paths['image_parcellation_path'])]

        art_remotion = Node(ArtifacRemotion(out_file='fmri_art_removed.nii'), name='artifact_remotion') #This interface requires implementation

        # BET - Skullstrip anatomical anf funtional images
        bet_t1 = Node(BET(frac=0.5, robust=True, mask=True, output_type='NIFTI_GZ'), name="bet_t1") #FSL

        # FAST - Image Segmentation
        segmentation = Node(FAST(output_type='NIFTI'), name="segmentation") #FSL

        # Normalize - normalizes functional and structural images to the MNI template
        normalize_fmri = Node(Normalize12(jobtype='estwrite',
                                          tpm=self.paths['template_spm_path'],
                                          write_voxel_sizes=[iso_size, iso_size, iso_size],
                                          write_bounding_box=[[-90, -126, -72], [90, 90, 108]]),
                              name="normalize_fmri") #SPM

        gunzip = Node(Gunzip(), name="gunzip")

        normalize_t1 = Node(Normalize12(jobtype='estwrite',
                                        tpm=self.paths['template_spm_path'],
                                        write_voxel_sizes=[iso_size, iso_size, iso_size],
                                        write_bounding_box=[[-90, -126, -72], [90, 90, 108]]),
                            name="normalize_t1")

        normalize_masks = Node(Normalize12(jobtype='estwrite',
                                           tpm=self.paths['template_spm_path'],
                                           write_voxel_sizes=[iso_size, iso_size, iso_size],
                                           write_bounding_box=[[-90, -126, -72], [90, 90, 108]]),
                               name="normalize_masks")

        # Threshold - Threshold WM probability image
        threshold = Node(Threshold(thresh=0.5,
                                   args='-bin',
                                   output_type='NIFTI_GZ'),
                         name="wm_mask_threshold")

        # FLIRT - pre-alignment of functional images to anatomical images
        coreg_pre = Node(FLIRT(dof=6, output_type='NIFTI_GZ'), name="linear_warp_estimation")

        # FLIRT - coregistration of functional images to anatomical images with BBR
        coreg_bbr = Node(FLIRT(dof=6, cost='bbr', schedule=opj(os.getenv('FSLDIR'), 'etc/flirtsch/bbr.sch'),
                               output_type='NIFTI_GZ'),
                         name="nonlinear_warp_estimation")

        # Apply coregistration warp to functional images
        applywarp = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI'),
                         name="registration_fmri")

        # Apply coregistration warp to mean file
        applywarp_mean = Node(FLIRT(interp='spline', apply_isoxfm=iso_size, output_type='NIFTI_GZ'),
                              name="registration_mean_fmri")

        # Infosource - a function free node to iterate over the list of subject names
        infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
        infosource.iterables = [('subject_id', subject_list)]

        # SelectFiles - to grab the data (alternativ to DataGrabber)
        anat_file = opj('{subject_id}', 'data/unprocessed/3T/T1w_acpc_dc_restore_1.25.nii')
        func_file = opj('{subject_id}', 'data/unprocessed/3T/rfMRI_REST1_LR/3T_rfMRI_REST1_LR.nii.gz')

        #anat_file = opj('{subject_id}/anat/', 'data.nii')
        #func_file = opj('{subject_id}/func/', 'data.nii')

        templates = {'anat': anat_file, 'func': func_file}

        selectfiles = Node(SelectFiles(templates, base_directory=self.paths['input_path']), name="selectfiles")

        # Datasink - creates output folder for important outputs
        datasink = Node(DataSink(base_directory=experiment_dir, container=output_dir), name="datasink")

        # Create a coregistration workflow
        coregwf = Workflow(name='coreg_fmri_to_t1')
        coregwf.base_dir = opj(experiment_dir, working_dir)

        # Create a preprocessing workflow
        preproc = Workflow(name='preproc')
        preproc.base_dir = opj(experiment_dir, working_dir)

        # Connect all components of the coregistration workflow

        coregwf.connect([(bet_t1, n4bias, [('out_file', 'in_file')]),
                         (n4bias, segmentation, [('out_file', 'in_files')]),
                         (segmentation, threshold, [(('partial_volume_files', get_latest), 'in_file')]),
                         (n4bias, coreg_pre, [('out_file', 'reference')]),
                         (threshold, coreg_bbr, [('out_file', 'wm_seg')]),
                         (coreg_pre, coreg_bbr, [('out_matrix_file', 'in_matrix_file')]),
                         (coreg_bbr, applywarp, [('out_matrix_file', 'in_matrix_file')]),
                         (n4bias, applywarp, [('out_file', 'reference')]),
                         (coreg_bbr, applywarp_mean, [('out_matrix_file', 'in_matrix_file')]),
                         (n4bias, applywarp_mean, [('out_file', 'reference')]),
                         ])

        ## Use the following DataSink output substitutions
        substitutions = [('_subject_id_', 'sub-')]
        #                 ('_fwhm_', 'fwhm-'),
        #                 ('_roi', ''),
        #                 ('_mcf', ''),
        #                 ('_st', ''),
        #                 ('_flirt', ''),
        #                 ('.nii_mean_reg', '_mean'),
        #                 ('.nii.par', '.par'),
        #                 ]
        # subjFolders = [('fwhm-%s/' % f, 'fwhm-%s_' % f) for f in fwhm]

        # substitutions.extend(subjFolders)
        datasink.inputs.substitutions = substitutions

        # Connect all components of the preprocessing workflow
        preproc.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                         (selectfiles, extract, [('func', 'in_file')]),
                         (extract, mcflirt, [('roi_file', 'in_file')]),
                         (mcflirt, slicetimer, [('out_file', 'in_file')]),
                         (selectfiles, denoise, [('anat', 'in_file')]),
                         (denoise, coregwf, [('out_file', 'bet_t1.in_file'),
                                             ('out_file', 'nonlinear_warp_estimation.reference')]),




                         (mcflirt, coregwf, [('mean_img', 'linear_warp_estimation.in_file'),
                                             ('mean_img', 'nonlinear_warp_estimation.in_file'),
                                             ('mean_img', 'registration_mean_fmri.in_file')]),




                         (slicetimer, coregwf, [('slice_time_corrected_file', 'registration_fmri.in_file')]),
                         (coregwf, art, [('registration_fmri.out_file', 'realigned_files')]),
                         (mcflirt, art, [('par_file', 'realignment_parameters')]),
                         (art, art_remotion, [('outlier_files', 'outlier_files')]),
                         (coregwf, art_remotion, [('registration_fmri.out_file', 'in_file')]),
                         (coregwf, gunzip, [('n4bias.out_file', 'in_file')]),




                         (selectfiles, normalize_fmri, [('anat', 'image_to_align')]),
                         (art_remotion, normalize_fmri, [('out_file', 'apply_to_files')]),








                         (selectfiles, normalize_t1, [('anat', 'image_to_align')]),
                         (gunzip, normalize_t1, [('out_file', 'apply_to_files')]),

                         (selectfiles, normalize_masks, [('anat', 'image_to_align')]),






                         (coregwf, normalize_masks, [(('segmentation.partial_volume_files', get_wm_csf), 'apply_to_files')]),








                         (normalize_fmri, smooth, [('normalized_files', 'in_files')]),
                         (smooth, extract_confounds_ws_csf, [('smoothed_files', 'in_file')]),
                         (normalize_masks, extract_confounds_ws_csf, [('normalized_files', 'list_mask')]),
                         (mcflirt, extract_confounds_ws_csf, [('par_file', 'file_concat')]),

                         # (smooth, extract_confounds_gs, [('smoothed_files', 'in_file')]),
                         # (normalize_t1, extract_confounds_gs, [(('normalized_files',change_to_list), 'list_mask')]),
                         # (extract_confounds_ws_csf, extract_confounds_gs, [('out_file', 'file_concat')]),

                         (smooth, signal_extraction, [('smoothed_files', 'in_file')]),
                         # (extract_confounds_gs, signal_extraction, [('out_file', 'confounds_file')]),
                         (extract_confounds_ws_csf, signal_extraction, [('out_file', 'confounds_file')]),

                         #(smooth, descomposition, [('smoothed_files', 'in_file')]),
                         #(extract_confounds_ws_csf, descomposition, [('out_file', 'confounds_file')]),

                         # (extract_confounds_gs, datasink, [('out_file', 'preprocessing.@confounds_with_gs')]),
                         (denoise, datasink, [('out_file', 'preprocessing.@t1_denoised')]),
                         (extract_confounds_ws_csf, datasink, [('out_file', 'preprocessing.@confounds_without_gs')]),
                         (smooth, datasink, [('smoothed_files', 'preprocessing.@smoothed')]),
                         (normalize_fmri, datasink, [('normalized_files', 'preprocessing.@fmri_normalized')]),
                         (normalize_t1, datasink, [('normalized_files', 'preprocessing.@t1_normalized')]),
                         (normalize_masks, datasink, [('normalized_files', 'preprocessing.@masks_normalized')]),
                         (signal_extraction, datasink, [('time_series_out_file', 'preprocessing.@time_serie')]),
                         (signal_extraction, datasink,
                          [('correlation_matrix_out_file', 'preprocessing.@correlation_matrix')])])
                         #(signal_extraction, datasink,
                         # [('fmri_cleaned_out_file', 'preprocessing.@fmri_cleaned_out_file')])])
                         #,
                         #(descomposition, datasink, [('out_file', 'preprocessing.@descomposition')]),
                         #(descomposition, datasink, [('plot_files', 'preprocessing.@descomposition_plot_files')])
                         #])

        preproc.write_graph(graph2use='colored', format='png', simple_form=True)
        preproc.run()
