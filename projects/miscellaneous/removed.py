import os

folder_path = '/home/brainlab/Desktop/Rudas/Data/hcp_100/output/workingdir/preproc'


cont = 1
for (root, dirs, files) in os.walk(folder_path):
    for file in sorted(files):
        if file == 'tractography2.trk':
            try:
                os.remove(root + '/' + file)
            except Exception:
                continue
'''

s = ['899885_3T_Diffusion_preproc',
                '857263_3T_Diffusion_preproc',
                '856766_3T_Diffusion_preproc',
                '792564_3T_Diffusion_preproc',
                '756055_3T_Diffusion_preproc',
                '751348_3T_Diffusion_preproc',
                '672756_3T_Diffusion_preproc',
                '654754_3T_Diffusion_preproc',
                '499566_3T_Diffusion_preproc',
                '414229_3T_Diffusion_preproc',
                '397760_3T_Diffusion_preproc',
                '366446_3T_Diffusion_preproc',
                '298051_3T_Diffusion_preproc',
                '280739_3T_Diffusion_preproc',
                '245333_3T_Diffusion_preproc',
                '239944_3T_Diffusion_preproc',
                '221319_3T_Diffusion_preproc',
                '214423_3T_Diffusion_preproc',
                '212318_3T_Diffusion_preproc',
                '211720_3T_Diffusion_preproc',
                '211417_3T_Diffusion_preproc',
                '208226_3T_Diffusion_preproc',
                '201111_3T_Diffusion_preproc',
                '199655_3T_Diffusion_preproc',
                '198451_3T_Diffusion_preproc',
                '196750_3T_Diffusion_preproc',
                '192540_3T_Diffusion_preproc',
                '190031_3T_Diffusion_preproc',
                '189450_3T_Diffusion_preproc',
                '188347_3T_Diffusion_preproc',
                '178950_3T_Diffusion_preproc',
                '176542_3T_Diffusion_preproc',
                '163129_3T_Diffusion_preproc',
                '162733_3T_Diffusion_preproc',
                '161731_3T_Diffusion_preproc',
                '160123_3T_Diffusion_preproc',
                '159340_3T_Diffusion_preproc',
                '156637_3T_Diffusion_preproc',
                '154734_3T_Diffusion_preproc',
                '153025_3T_Diffusion_preproc',
                '151627_3T_Diffusion_preproc',
                '151526_3T_Diffusion_preproc',
                '151223_3T_Diffusion_preproc',
                '149741_3T_Diffusion_preproc',
                '149539_3T_Diffusion_preproc',
                '149337_3T_Diffusion_preproc',
                '148840_3T_Diffusion_preproc',
                '148335_3T_Diffusion_preproc',
                '147737_3T_Diffusion_preproc',
                '146432_3T_Diffusion_preproc',
                '144832_3T_Diffusion_preproc',
                '140925_3T_Diffusion_preproc',
                '139637_3T_Diffusion_preproc',
                '138534_3T_Diffusion_preproc',
                '136833_3T_Diffusion_preproc',
                '135932_3T_Diffusion_preproc',
                '135225_3T_Diffusion_preproc',
                '133928_3T_Diffusion_preproc',
                '133019_3T_Diffusion_preproc',
                '131722_3T_Diffusion_preproc',
                '131217_3T_Diffusion_preproc',
                '130316_3T_Diffusion_preproc',
                '130013_3T_Diffusion_preproc',
                '129028_3T_Diffusion_preproc',
                '128632_3T_Diffusion_preproc',
                '128127_3T_Diffusion_preproc',
                '127933_3T_Diffusion_preproc']
'''

'''
cont = 1
for sub in s:
    path = folder_path + '/' + sub
    for file in os.listdir(path):
        os.rename(path + '/' + file, path + '/data')
        
'''