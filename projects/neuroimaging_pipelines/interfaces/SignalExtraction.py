from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class SignalExtractionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    time_series_out_file = File(mandatory=True)
    correlation_matrix_out_file = File(genfile=True)
    image_parcellation_path = traits.String(mandatory=True)
    labels_parcellation_path = traits.Either(
        traits.String(),
        traits.ArrayOrNone())
    mask_mni_path = traits.String(mandatory=True)
    tr = traits.Float(mandatory=True)
    low_pass = traits.Float(default_value=None)
    high_pass = traits.Float(default_value=None)
    confounds_file = File(mandatory=True)
    plot = traits.Bool(default_value=False, mandatory = False)


class SignalExtractionOutputSpec(TraitedSpec):
    time_series_out_file = File(genfile=True)
    correlation_matrix_out_file = File(genfile=True)
    #fmri_cleaned_out_file = File(genfile=True)


class SignalExtraction(BaseInterface):
    input_spec = SignalExtractionInputSpec
    output_spec = SignalExtractionOutputSpec

    def _run_interface(self, runtime):

        from nilearn.input_data import NiftiLabelsMasker
        from nilearn.image import clean_img
        import numpy as np
        import nibabel as nib

        '''
        image_cleaned = clean_img(self.inputs.in_file,
                                  sessions=None,
                                  detrend=True,
                                  standardize=True,
                                  low_pass=self.inputs.low_pass,
                                  high_pass=self.inputs.high_pass,
                                  t_r=self.inputs.tr,
                                  confounds=self.inputs.confounds_file,
                                  ensure_finite=True,
                                  mask_img=self.inputs.mask_mni_path)

        nib.save(image_cleaned, 'fmri_cleaned.nii')
        '''
        masker = NiftiLabelsMasker(labels_img=self.inputs.image_parcellation_path,
                                   standardize=True,
                                   detrend=True,
                                   low_pass=self.inputs.low_pass,
                                   high_pass=self.inputs.high_pass,
                                   t_r=self.inputs.tr,
                                   memory='nilearn_cache',
                                   verbose=0)

        time_series = masker.fit_transform(self.inputs.in_file, confounds=self.inputs.confounds_file)

        np.savetxt(self.inputs.time_series_out_file, time_series, fmt='%10.2f', delimiter=',')

        labels = []
        if self.inputs.labels_parcellation_path is not None:
            file_labels = open(self.inputs.labels_parcellation_path, 'r')

            for line in file_labels.readlines():
                labels.append(line)
            file_labels.close()
        else:
            labels = list(range(time_series.shape[-1]))

        if self.inputs.plot:
            from nilearn import plotting
            from nilearn.connectome import ConnectivityMeasure
            import matplotlib
            import matplotlib.pyplot as plt
            fig, ax = matplotlib.pyplot.subplots()

            font = {'family': 'normal',
                    'size': 5}

            matplotlib.rc('font', **font)

            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([time_series])[0]

            # Mask the main diagonal for visualization:
            np.fill_diagonal(correlation_matrix, 0)
            plotting.plot_matrix(correlation_matrix, figure=fig, labels=labels, vmax=1.0, vmin=-1.0, reorder=True)

            fig.savefig(self.inputs.correlation_matrix_out_file, dpi=1200)


        return runtime

    def _list_outputs(self):
        return {'time_series_out_file': os.path.abspath(self.inputs.time_series_out_file),
                'correlation_matrix_out_file': os.path.abspath(self.inputs.correlation_matrix_out_file)}
                #'fmri_cleaned_out_file': os.path.abspath('fmri_cleaned.nii')}

    def _gen_filename(self, name):
        if name == 'time_series_out_file':
            return os.path.abspath(self.inputs.time_series_out_file)
        if name == 'correlation_matrix_out_file':
            return os.path.abspath(self.inputs.correlation_matrix_out_file)
        return None