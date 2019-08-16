from nipype.interfaces.base import BaseInterfaceInputSpec, File, TraitedSpec, BaseInterface
import os

class ArtifacRemotionInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True)
    out_file = File(mandatory=True)
    outlier_files = File(mandatory=True)

class ArtifacRemotionOutputSpec(TraitedSpec):
    out_file = File(genfile=True)

class ArtifacRemotion(BaseInterface):
    input_spec = ArtifacRemotionInputSpec
    output_spec = ArtifacRemotionOutputSpec

    def _run_interface(self, runtime):
        import nibabel as nib
        import numpy as np

        img = nib.load(self.inputs.in_file)
        affine = img.affine
        data = img.get_data()

        artifact_volumens = np.loadtxt(self.inputs.outlier_files, dtype=int)
        if len(artifact_volumens) != 0:
            data_artifact_removed = np.delete(data, artifact_volumens, axis=-1)
        else:
            data_artifact_removed = data

        nib.save(nib.Nifti1Image(data_artifact_removed, affine), self.inputs.out_file)

        return runtime

    def _list_outputs(self):
        return {'out_file': os.path.abspath(self.inputs.out_file)}