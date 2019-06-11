from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ExtractB0InputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    bval_path = File(exists=True, mandatory=True)
    bvec_path = File(exists=True, mandatory=True)


class ExtractB0OutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class ExtractB0(BaseInterface):
    input_spec = ExtractB0InputSpec
    output_spec = ExtractB0OutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        from dipy.io import read_bvals_bvecs
        from dipy.core.gradients import gradient_table
        import numpy as np

        file_name = 'mean_b0.nii.gz'

        bvals, bvecs = read_bvals_bvecs(self.inputs.bval_path, self.inputs.bvec_path)
        gtab = gradient_table(bvals, bvecs, b0_threshold=50)

        img = nib.load(self.inputs.in_file)
        b0s = img.get_data()[..., gtab.b0s_mask]

        mean_b0 = np.mean(b0s, -1)
        nib.save(nib.Nifti1Image(mean_b0, img.affine), file_name)

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('mean_b0.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None