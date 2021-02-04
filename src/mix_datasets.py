"""Scriping for mixing dataset into a new one."""


import numpy as np
from sklearn.utils import shuffle

from src.utils import picklify, unpicklify


def main(datset1_path, datset2_path, mix_dataset_path,
         n_class=10,
         label_separation: ('Random state', 'flag', 's')=False,
         random_state: ('Random state', 'option', 'r')=42):
    
    if random_state is not None:
        np.random.seed(random_state)
    
    dataset1 = unpicklify(datset1_path)
    dataset2 = unpicklify(datset2_path)
    assert dataset1.keys() == dataset2.keys()
    print('Equal shapes:', all(dataset1[key].shape == dataset2[key].shape
                               for key in dataset1))
    
    if label_separation:
        assert n_class % 2 == 0
        middle_label = n_class // 2
        
        print('Label Separation on:', middle_label)

        for sign, dataset in enumerate((dataset1, dataset2)):
            
            train_mask = dataset['y_train'] < middle_label
            test_mask = dataset['y_test'] < middle_label

            if sign:
                train_mask = ~train_mask
                test_mask = ~test_mask

            dataset['X_train'] = dataset['X_train'][train_mask]
            dataset['y_train'] = dataset['y_train'][train_mask]
            
            dataset['X_test'] = dataset['X_test'][test_mask]
            dataset['y_test'] = dataset['y_test'][test_mask]


    mix_dataset = {'X_train': np.vstack([dataset1['X_train'], dataset2['X_train']]),
                   'X_test': np.vstack([dataset1['X_test'], dataset2['X_test']]),
                   'y_train': np.concatenate([dataset1['y_train'], dataset2['y_train']]),
                   'y_test': np.concatenate([dataset1['y_test'], dataset2['y_test']])
    }
   
    print(mix_dataset['X_train'].shape, mix_dataset['y_train'].shape)

    perm_train = np.random.permutation(len(mix_dataset['y_train']))
    mix_dataset['X_train'] = mix_dataset['X_train'][perm_train]
    mix_dataset['y_train'] = mix_dataset['y_train'][perm_train]
    mix_dataset['train_split'] = (perm_train >= len(dataset1['y_train']))

    perm_test = np.random.permutation(len(mix_dataset['y_test']))
    mix_dataset['X_test'] = mix_dataset['X_test'][perm_test]
    mix_dataset['y_test'] = mix_dataset['y_test'][perm_test]
    mix_dataset['test_split'] = (perm_test >= len(dataset1['y_test']))
    
    picklify(mix_dataset_path, mix_dataset)


if __name__ == '__main__':
    import plac; plac.call(main)