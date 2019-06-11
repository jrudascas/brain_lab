from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class RegistrationInputSpec(BaseInterfaceInputSpec):
    image_to_align = File(exists=True, mandatory=True)
    image_to_apply = File(exists=True, mandatory=True)
    reference = File(exists=True, mandatory=True)


class RegistrationOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class Registration(BaseInterface):
    input_spec = RegistrationInputSpec
    output_spec = RegistrationOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        from ..tools.utils import affine_registration, syn_registration

        img_to_align = nib.load(self.inputs.image_to_align)
        data_to_align = img_to_align.get_data()
        affine_to_align = img_to_align.affine

        img_reference = nib.load(self.inputs.reference)
        data_reference = img_reference.get_data()
        affine_reference = img_reference.affine

        img_to_apply = nib.load(self.inputs.image_to_apply)
        data_to_apply = img_to_apply.get_data()
        affine_to_apply = img_to_apply.affine

        transformed, starting_affine = affine_registration(moving=data_to_align,
                                                           static=data_reference,
                                                           moving_grid2world=affine_to_align,
                                                           static_grid2world=affine_reference)

        transformed, mapping = syn_registration(moving=data_to_align,
                                                static=data_reference,
                                                moving_grid2world=affine_to_align,
                                                static_grid2world=affine_reference,
                                                metric='CC',
                                                dim=3, level_iters=[10, 10, 5],
                                                prealign=starting_affine)

        if len(data_to_apply.shape) > 3:
            transformed = np.zeros((data_reference.shape + (data_to_apply.shape[-1],)))
            for i in range(data_to_apply.shape[-1]):
                transformed[..., i] = mapping.transform(data_to_apply[..., i], interpolation='nearest')

        nib.save(nib.Nifti1Image(transformed, affine_reference), 'registered.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('registered.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None
