from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class DenoiseInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)


class DenoiseOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class Denoise(BaseInterface):
    input_spec = DenoiseInputSpec
    output_spec = DenoiseOutputSpec

    def _run_interface(self, runtime):

        import nibabel as nib
        from dipy.denoise.nlmeans import nlmeans
        from dipy.denoise.noise_estimate import estimate_sigma
        import numpy as np

        img = nib.load(self.inputs.in_file)

        data = img.get_data()
        affine = img.affine

        if len(data.shape) > 3:
            den = np.zeros(data.shape)

            for i in range(data.shape[-1]):
                print('direction # ' + str(i))
                sigma = estimate_sigma(data[..., i], N=4)
                den[..., i] = nlmeans(data[..., i], sigma=sigma, patch_radius=1, block_radius=5, rician=True)

            nib.save(nib.Nifti1Image(den.astype(np.float32), img.affine), 'denoised.nii.gz')
        else:
            sigma = estimate_sigma(data, N=4)

            den = nlmeans(data, sigma=sigma, patch_radius=1, block_radius=5, rician=True)

        nib.save(nib.Nifti1Image(den, affine), 'denoised.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('denoised.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None