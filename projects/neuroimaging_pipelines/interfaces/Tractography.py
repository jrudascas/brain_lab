from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class TractographyInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    bval_path = File(exists=True, mandatory=True)
    bvec_path = File(exists=True, mandatory=True)
    image_parcellation_path = traits.String(mandatory=True)


class TractographyOutputSpec(TraitedSpec):
    out_file = File(genfile=True)
    jij_file = File(genfile=True)
    figure_file = File(genfile=True)


class Tractography(BaseInterface):
    input_spec = TractographyInputSpec
    output_spec = TractographyOutputSpec

    def _run_interface(self, runtime):
        from dipy.reconst.shm import CsaOdfModel
        from dipy.direction import peaks_from_model
        from dipy.data import default_sphere
        from dipy.core.gradients import gradient_table
        from dipy.tracking import utils
        import nibabel as nib
        from dipy.core.gradients import gradient_table
        import dipy.reconst.dti as dti
        from dipy.reconst.dti import color_fa, fractional_anisotropy, quantize_evecs
        from dipy.data import get_sphere
        from dipy.tracking.eudx import EuDX
        from dipy.tracking.utils import connectivity_matrix
        import numpy as np
        from dipy.tracking.streamline import Streamlines
        from nilearn import plotting
        import matplotlib

        dwi_img = nib.load(self.inputs.in_file)
        dwi_data = dwi_img.get_data()
        dwi_affine = dwi_img.affine

        mask_img = nib.load(self.inputs.mask_file)
        mask_data = mask_img.get_data().astype(bool)
        mask_affine = mask_img.affine

        step_size = 0.5
        max_angle = 30
        density = 2
        lenght_threshold = 10
        sh_order = 6
        min_separation_angle = 30
        relative_peak_threshold = .5
        threshold_tissue_classifier = .05

        gtab = gradient_table(self.inputs.bval_path, self.inputs.bvec_path, b0_threshold=50)

        # seeds = utils.seeds_from_mask(mask_data, density=density, affine=np.eye(4))

        '''
        #DTI reconstruction and EuDx tracking
        tensor_model = dti.TensorModel(gtab)
        tensor_fitted = tensor_model.fit(data, mask)
        FA = fractional_anisotropy(tensor_fitted.evals)

        peak_indices = quantize_evecs(tensor_fitted.evecs, default_sphere.vertices)

        streamlines_generator = EuDX(FA.astype('f8'), peak_indices, 
                                    odf_vertices=default_sphere.vertices, 
                                    a_low=threshold_tissue_classifier,
                                    step_sz=step_size,
                                    seeds=1000000)
        '''

        # QBall reconstruction and EuDX tracking
        csa_model = CsaOdfModel(gtab, sh_order=sh_order)

        csa_peaks = peaks_from_model(model=csa_model,
                                     data=dwi_data,
                                     sphere=default_sphere,
                                     relative_peak_threshold=relative_peak_threshold,
                                     min_separation_angle=min_separation_angle,
                                     mask=mask_data)

        streamlines_generator = EuDX(csa_peaks.peak_values, csa_peaks.peak_indices,
                                     odf_vertices=default_sphere.vertices,
                                     a_low=threshold_tissue_classifier, step_sz=step_size,
                                     seeds=1000000)

        self.save(streamlines_generator, streamlines_generator.affine, mask_data.shape, 'tractography2.trk',
                  lenght_threshold)

        strem = Streamlines(streamlines_generator, buffer_size=512)
        labels = nib.load(self.inputs.image_parcellation_path).get_data().astype(int)

        M, grouping = connectivity_matrix(strem, labels, affine=streamlines_generator.affine,
                                          return_mapping=True,
                                          mapping_as_streamlines=True)

        M = M[1:, 1:]  # Removing firsts column and row (Index = 0 is the background)
        M[range(M.shape[0]), range(M.shape[0])] = 0  # Removing element over the diagonal

        np.savetxt('Jij.csv', M, delimiter=',', fmt='%d')

        fig, ax = matplotlib.pyplot.subplots()
        plotting.plot_matrix(M, colorbar=True, figure=fig)
        fig.savefig('Jij.png', dpi=1200)

        return runtime

    def save(self, strem_generator, affine_, shape_, path_output, h):
        from dipy.io.streamline import save_trk
        stream = [s for s in strem_generator if s.shape[0] > h]

        save_trk(path_output, list(stream), affine_, shape_)

    def _list_outputs(self):
        return {'out_file': os.path.abspath('tractography2.trk'),
                'jij_file': os.path.abspath('Jij.csv'),
                'figure_file': os.path.abspath('Jij.png')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None
