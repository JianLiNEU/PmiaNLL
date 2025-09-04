import os



def save_info(save_dir, data_set, file_title):
    path = save_dir + f'/{file_title}_info.txt'
    print('save path: {}'.format(path))
    if os.path.isfile(path):
        os.remove(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(path, 'w', encoding='utf-8') as fout:
        fout.writelines(data_set)