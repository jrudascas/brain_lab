from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class ReslicingInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    vox_sz = traits.Float(mandatory=True)


class ReslicingOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class Reslicing(BaseInterface):
    input_spec = ReslicingInputSpec
    output_spec = ReslicingOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        from dipy.align.reslice import reslice

        img = nib.load(self.inputs.in_file)
        data = img.get_data()
        affine = img.affine

        old_vox_sz = img.header.get_zooms()[:3]

        new_vox_sz = (self.inputs.vox_sz, self.inputs.vox_sz, self.inputs.vox_sz)

        data, affine = reslice(data, affine, old_vox_sz, new_vox_sz, num_processes=3)

        nib.save(nib.Nifti1Image(data, affine), 'resliced.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('resliced.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None
