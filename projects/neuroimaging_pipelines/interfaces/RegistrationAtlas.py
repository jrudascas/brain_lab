from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class RegistrationAtlasInputSpec(BaseInterfaceInputSpec):
    image_to_align = File(exists=True, mandatory=True)
    atlas_to_apply = File(exists=True, mandatory=True)
    reference = File(exists=True, mandatory=True)


class RegistrationAtlasOutputSpec(TraitedSpec):
    out_file = File(genfile=True)


class RegistrationAtlas(BaseInterface):
    input_spec = RegistrationAtlasInputSpec
    output_spec = RegistrationAtlasOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np
        from ..tools.utils import affine_registration, syn_registration
        import scipy.ndimage as ndim

        img_atlas = nib.load(self.inputs.atlas_to_apply)
        atlas_data = np.squeeze(img_atlas.get_data())

        img_to_align = nib.load(self.inputs.image_to_align)
        data_to_align = img_to_align.get_data()
        affine_to_align = img_to_align.affine

        img_reference = nib.load(self.inputs.reference)
        data_reference = img_reference.get_data()
        affine_reference = img_reference.affine

        transformed, starting_affine = affine_registration(moving=data_to_align,
                                                           static=data_reference,
                                                           moving_grid2world=affine_to_align,
                                                           static_grid2world=affine_reference)

        transformed, mapping = syn_registration(moving=data_to_align,
                                                static=data_reference,
                                                moving_grid2world=affine_to_align,
                                                static_grid2world=affine_reference,
                                                prealign=starting_affine)

        list_path_roi = []

        cont = 1
        for index in np.unique(atlas_data):
            if index != 0: #Backgroud
                roi = np.zeros(atlas_data.shape)
                roi[np.where(atlas_data == index)] = 1

                warped_roi = mapping.transform_inverse(ndim.binary_dilation(roi).astype(int), interpolation='nearest')

                warped_roi = ndim.binary_dilation(warped_roi)
                warped_roi = ndim.binary_erosion(warped_roi)

                bin_warped_roi = np.ceil(warped_roi)

                filled_warped_roi = ndim.binary_fill_holes(bin_warped_roi.astype(int)).astype(int)

                if not ('registered_atlas' in locals()):
                    registered_atlas = np.zeros(filled_warped_roi.shape)

                registered_atlas[np.where(filled_warped_roi != 0)] = cont
                cont += 1

        nib.save(nib.Nifti1Image(registered_atlas.astype(np.float32), affine_to_align), 'atlas_registered.nii.gz')

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath('atlas_registered.nii.gz')}

    def _gen_filename(self, name):
        if name == 'out_file':
            return os.path.abspath(self.inputs.out_file)
        return None
