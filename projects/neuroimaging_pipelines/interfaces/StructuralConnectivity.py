from nipype.interfaces.base import  BaseInterfaceInputSpec, File, TraitedSpec, traits, BaseInterface
import os


class StructuralConnectivityInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)

class StructuralConnectivityOutputSpec(TraitedSpec):
    out_file_fa = File(genfile=True)
    out_file_fa_color = File(genfile=True)
    out_file_md = File(genfile=True)
    out_file_rd = File(genfile=True)
    out_file_ad = File(genfile=True)
    out_file_trk = File(genfile=True)


class StructuralConnectivity(BaseInterface):

    input_spec = StructuralConnectivityInputSpec
    output_spec = StructuralConnectivityOutputSpec

    def _run_interface(self, runtime):
        from dipy.tracking.streamline import Streamlines

        s_affine = streamlines_generator.affine
        strem = Streamlines(streamlines_generator, buffer_size=512)

        labels = nib.load(new_path_atlas).get_data().astype(int)

        index = np.unique(labels)
        M, grouping = connectivity_matrix(strem, labels, affine=s_affine,
                                          return_mapping=True,
                                          mapping_as_streamlines=True)
        M = M[index, :]
        M = M[:, index]
        M = M[1:, 1:]
        M = np.insert(M, 0, index[1:], axis=1)
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