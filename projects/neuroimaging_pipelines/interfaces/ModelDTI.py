from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ModelDTIInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    mask_file = File(exists=True, mandatory=True)
    bval_path = File(exists=True, mandatory=True)
    bvec_path = File(exists=True, mandatory=True)


class ModelDTIOutputSpec(TraitedSpec):
    out_file_fa = File(genfile=True)
    out_file_fa_color = File(genfile=True)
    out_file_md = File(genfile=True)
    out_file_rd = File(genfile=True)
    out_file_ad = File(genfile=True)
    out_file_trk = File(genfile=True)


class ModelDTI(BaseInterface):

    input_spec = ModelDTIInputSpec
    output_spec = ModelDTIOutputSpec

    def _run_interface(self, runtime):

        import nibabel as nib
        from dipy.io import read_bvals_bvecs
        from dipy.core.gradients import gradient_table
        import dipy.reconst.dti as dti
        from dipy.reconst.dti import color_fa, fractional_anisotropy, quantize_evecs
        import numpy as np

        img = nib.load(self.inputs.in_file)
        data = img.get_data()
        affine = img.affine
        mask = nib.load(self.inputs.mask_file)
        mask = mask.get_data()
        bvals, bvecs = read_bvals_bvecs(self.inputs.bval_path, self.inputs.bvec_path)
        gtab = gradient_table(bvals, bvecs, b0_threshold=50)

        tensor_model = dti.TensorModel(gtab)
        tensor_fitted = tensor_model.fit(data, mask)

        FA = fractional_anisotropy(tensor_fitted.evals)
        FA[np.isnan(FA)] = 0

        nib.save(nib.Nifti1Image(FA.astype(np.float32), affine), 'FA.nii.gz')
        FA2 = np.clip(FA, 0, 1)
        RGB = color_fa(FA2, tensor_fitted.evecs)
        nib.save(nib.Nifti1Image(np.array(255 * RGB, 'uint8'), affine), 'FA_RGB.nii.gz')

        MD = dti.mean_diffusivity(tensor_fitted.evals)
        nib.save(nib.Nifti1Image(MD.astype(np.float32), affine), 'MD.nii.gz')

        AD = dti.axial_diffusivity(tensor_fitted.evals)
        nib.save(nib.Nifti1Image(AD.astype(np.float32), affine), 'AD.nii.gz')

        RD = dti.radial_diffusivity(tensor_fitted.evals)
        nib.save(nib.Nifti1Image(RD.astype(np.float32), affine), 'RD.nii.gz')
        from dipy.data import get_sphere
        from dipy.tracking.eudx import EuDX

        sphere = get_sphere('symmetric724')
        peak_indices = quantize_evecs(tensor_fitted.evecs, sphere.vertices)

        eu = EuDX(FA.astype('f8'), peak_indices, seeds=1000000, odf_vertices=sphere.vertices, a_low=0.15)
        tensor_streamlines = [streamline for streamline in eu]

        hdr = nib.trackvis.empty_header()
        hdr['voxel_size'] = img.get_header().get_zooms()[:3]
        hdr['voxel_order'] = 'LAS'
        hdr['dim'] = FA.shape
        tensor_streamlines_trk = ((sl, None, None) for sl in tensor_streamlines)

        nib.trackvis.write('_tractography_EuDx.trk', tensor_streamlines_trk, hdr, points_space='voxel')
        return runtime


    def _list_outputs(self):
        return {'out_file_fa': os.path.abspath('FA.nii.gz'),
                'out_file_fa_color': os.path.abspath('FA_RGB.nii.gz'),
                'out_file_md': os.path.abspath('MD.nii.gz'),
                'out_file_rd': os.path.abspath('RD.nii.gz'),
                'out_file_ad': os.path.abspath('AD.nii.gz'),
                'out_file_trk': os.path.abspath('_tractography_EuDx.trk')}


    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None