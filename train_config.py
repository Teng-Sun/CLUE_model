from functools import reduce

"""
mosei -> split_dataset_2 (mosei 2分类偏差数据集)
mosei -> split_dataset_7classes_1 (mosei 7分类偏差数据集)

mosi -> split_dataset_1 (mosi 2分类偏差数据集)
mosi -> split_dataset_2 (新 mosi 2分类偏差数据集)
mosi -> split_dataset_7classes_1 (mosi 7分类偏差数据集)
"""

model_name = 'magbert_model'
data = 'mosei'
data_name = 'split_dataset_2'
if '7classes' in data_name:
    output_size = 7
else: output_size = 2

run_name = 'log_magbert_lr'

normal_string = 'nohup python train.py --data {} --dataset_name {} --base_model {} --output_size {} --gpu_id 2'.format(data, data_name, model_name, output_size)

clip =                      [('clip', i) for i in                       [2.0, 1.5, 1.0]]
klloss =                    [('klloss', i) for i in                     [True, False]]
fusion_mode =               [('fusion_mode', i) for i in                ['sum', 'hm']]
tmodel_learning_rate =      [('tmodel_learning_rate', i) for i in       [1e-5]]
tmodel_weight_decay =       [('tmodel_weight_decay', i) for i in        [0, 1e-5]]

misa_learning_rate =        [('misa_learning_rate', i) for i in         [1e-5, 3e-5, 6e-5]]
misa_weight_decay =         [('misa_weight_decay', i) for i in          [1e-5, 3e-5]]
misa_dropout =              [('misa_dropout', i) for i in               [0.1, 0.3, 0.5]]
misa_diff_weight =          [('misa_diff_weight', i) for i in           [0.1, 0.3, 0.5]]
misa_recon_weight =         [('misa_recon_weight', i) for i in          [1.0, 0.6, 0.3]]
batch_size =                [('batch_size', i) for i in                 [16, 24]]

magbert_dropout =           [('magbert_dropout', i) for i in            [0.1, 0.3, 0.5]]
magbert_learning_rate =     [('magbert_learning_rate', i) for i in      [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]]
magbert_weight_decay =      [('magbert_weight_decay', i) for i in       [0, 1e-5]]
magbert_beta_shift =        [('magbert_beta_shift', i) for i in         [0.5, 1.5, 2]]

selfmm_learning_rate =      [('selfmm_learning_rate', i) for i in       [1e-5]]
selfmm_weight_decay =       [('selfmm_weight_decay', i) for i in        [1e-5, 0]]
selfmm_H =                  [('selfmm_H', i) for i in                   [2.0, 3.0]]

min_number =                [('min_number', i) for i in                 [200, 300, 400]]
split_rate =                [('split_rate', i) for i in                 [0.5, 0.3, 0.1]]
random_rate =               [('random_rate', i) for i in                [0, 0.5, 0.8]]

variance =                  [('variance', i) for i in                   [0.1]]


tmodel_learning_rate =      [('tmodel_learning_rate', i) for i in       [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]]
tmodel_weight_decay =       [('tmodel_weight_decay', i) for i in        [0, 1e-5]]
misa_learning_rate =        [('misa_learning_rate', i) for i in         [1e-5, 3e-5, 6e-5]]
misa_weight_decay =         [('misa_weight_decay', i) for i in          [1e-5, 3e-5]]
fusion_mode =               [('fusion_mode', i) for i in                ['sum', 'hm']]

new_list = [[[]], magbert_learning_rate, tmodel_learning_rate]

function = lambda all_list: reduce(lambda x, y: [i + [j] for i in x for j in y], all_list)
all_config_list = function(new_list)

with open('{}.bash'.format(run_name), 'w+') as f:
    for index, config_list in enumerate(all_config_list):
        f.write(normal_string)
        for config in config_list:
            f.write(' --{} {}'.format(config[0], config[1]))
        # f.write(' --model_savepath /home/causal_model/checkpoints/{}'.format(run_name))
        # f.write(' --model_index {}'.format(index))
        f.write(' >> /home/sunteng/new_CLUE_model/{}/{}.log \n'.format(run_name, index))