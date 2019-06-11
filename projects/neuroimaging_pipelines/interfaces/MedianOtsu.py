from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface, OutputMultiPath
import os


class MedianOtsuInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)


class MedianOtsuOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(genfile=True))


class MedianOtsu(BaseInterface):
    input_spec = MedianOtsuInputSpec
    output_spec = MedianOtsuOutputSpec

    def _run_interface(self, runtime):

        from dipy.segment.mask import median_otsu
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        data = img.get_data()

        dwi_masked, mask = median_otsu(data, 2, 1)

        nib.save(nib.Nifti1Image(dwi_masked, img.affine), 'dwi_brain.nii.gz')
        nib.save(nib.Nifti1Image(mask.astype(np.float32), img.affine), 'dwi_mask.nii.gz')

        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = []
        outputs['out_file'].append(os.path.abspath('dwi_brain.nii.gz'))
        outputs['out_file'].append(os.path.abspath('dwi_mask.nii.gz'))

        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None