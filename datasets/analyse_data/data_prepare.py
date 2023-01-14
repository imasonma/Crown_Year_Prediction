import os
import sys
import json
import traceback

import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

from Common import DataBaseUtils, read_txt, load_ct_info, change_axes_of_image


def run_prepare_data(cfg):
    data_prepare = DataPrepare(cfg)

    pool = Pool(int(cpu_count() * 0.7))
    for data in data_prepare.data_info:
        # data_prepare.process(data)
        # try:
            pool.apply_async(data_prepare.process, (data,))
        # except Exception as err:
            # traceback.print_exc()
            # print('Create coarse/fine image/mask throws exception %s, with series_id %s!' % (err, data.series_id))

    pool.close()
    pool.join()

    # data_prepare._split_train_val()
    # pass


class MaskData(object):
    def __init__(self, series_id, image_path, mask_path,
                 smooth_mask_path=None, coarse_image_path=None,
                 coarse_mask_path=None, fine_image_path=None, fine_mask_path=None):
        super(MaskData, self).__init__()

        self.series_id = series_id
        self.image_path = image_path
        self.mask_path = mask_path
        self.smooth_mask_path = smooth_mask_path
        self.coarse_image_path = coarse_image_path
        self.coarse_mask_path = coarse_mask_path
        self.fine_image_path = fine_image_path
        self.fine_mask_path = fine_mask_path


class DataPrepare(object):
    def __init__(self, cfg):
        super(DataPrepare, self).__init__()
        self.cfg = cfg

        self.out_dir = cfg.DATA_PREPARE.OUT_DIR
        self.db_dir = self.out_dir + '/db/'
        self.train_db_file = self.db_dir + 'seg_raw_train'
        self.test_db_file = self.db_dir + 'seg_raw_test'
        self.out_db_file = self.db_dir + 'seg_pre-process_database'
        self.out_train_db_file = self.db_dir + 'seg_train_fold_1'
        self.out_val_db_file = self.db_dir + 'seg_val_fold_1'

        self.image_dir = cfg.DATA_PREPARE.TRAIN_IMAGE_DIR
        self.mask_dir = cfg.DATA_PREPARE.TRAIN_MASK_DIR
        self.mask_label = cfg.DATA_PREPARE.MASK_LABEL
        self.extend_size = cfg.DATA_PREPARE.EXTEND_SIZE

        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
        self._create_db_file(phase='train')
        self._create_db_file(phase='test')
        self.data_info = self._read_db()

    def _read_db(self):
        local_data = []
        env = lmdb.open(self.train_db_file, map_size=int(1e9))
        txn = env.begin()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            label_info['image_path'] = {f'{i}_path':label_info[f'{i}_path']  for i in self.cfg.DATA_PREPARE.IMAGE_MODAL}

            tmp_data = MaskData(key,
                                label_info['image_path'],
                                label_info['mask_path'])
            local_data.append(tmp_data)
        env.close()

        print('Num of ct is %d.' % len(local_data))
        return local_data

    def _create_modal_info(self, series_id, parse, data_dict = dict()):
        for i in self.cfg.DATA_PREPARE.IMAGE_MODAL:
            image_path = os.path.join(self.cfg.ENVIRONMENT.DATA_BASE_DIR, series_id, i + parse)
            if os.path.exists(image_path):
                data_dict[f'{i}_path'] = image_path
            else:
                print(f'{series_id} has invalid {i}.')
        return data_dict

    def _create_db_file(self, phase='train', parse = '.nii.gz'):
        db_file_path = self.train_db_file if phase == 'train' else self.test_db_file
        DataBaseUtils.creat_db(db_file_path)

        series_ids = read_txt(self.cfg.DATA_PREPARE.TRAIN_SERIES_IDS_TXT) \
            if phase == 'train' else read_txt(self.cfg.DATA_PREPARE.TEST_SERIES_IDS_TXT)

        for series_id in series_ids:
            if phase == 'train':
                mask_path = os.path.join(self.cfg.ENVIRONMENT.DATA_BASE_DIR, series_id, self.cfg.DATA_PREPARE.MASK_MODAL + parse)
                if os.path.exists(mask_path):
                    data_dict = {
                        'mask_path': mask_path
                    }
                    data_dict = self._create_modal_info(series_id, parse, data_dict)
                    DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)
                else:
                    print('%s has invalid image/mask.' % series_id)
            else:
                image_path = os.path.join(self.cfg.DATA_PREPARE.TEST_IMAGE_DIR, series_id + '_0000' + parse)
                mask_path = os.path.join(self.cfg.DATA_PREPARE.TEST_MASK_DIR, series_id + parse) \
                    if self.cfg.DATA_PREPARE.TEST_MASK_DIR is not None else None
                data_dict = self._create_modal_info(series_id, parse)
                data_dict['mask_path'] = mask_path
                DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)

    def _split_train_val(self):
        default_train_db = self.cfg.DATA_PREPARE.DEFAULT_TRAIN_DB
        default_val_db = self.cfg.DATA_PREPARE.DEFAULT_VAL_DB

        if default_train_db is not None and default_val_db is not None:
            env = lmdb.open(default_train_db, map_size=int(1e9))
            txn = env.begin()
            series_ids_train = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids_train.append(key)

            env = lmdb.open(default_val_db, map_size=int(1e9))
            txn = env.begin()
            series_ids_val = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids_val.append(key)

            env = lmdb.open(self.out_db_file, map_size=int(1e9))
            txn = env.begin()
        else:
            env = lmdb.open(self.out_db_file, map_size=int(1e9))
            txn = env.begin()
            series_ids = []
            for key, value in txn.cursor():
                key = str(key, encoding='utf-8')
                series_ids.append(key)

            series_ids_train, series_ids_val = train_test_split(series_ids, test_size=self.cfg.DATA_PREPARE.VAL_RATIO,
                                                                random_state=0)

        print('Num of train series is: %d, num of val series is: %d.' % (len(series_ids_train), len(series_ids_val)))

        # create train db
        env_train = lmdb.open(self.out_train_db_file, map_size=int(1e9))
        txn_train = env_train.begin(write=True)
        for series_id in series_ids_train:
            value = str(txn.get(str(series_id).encode()), encoding='utf-8')
            txn_train.put(key=str(series_id).encode(), value=str(value).encode())
        txn_train.commit()
        env_train.close()

        # create val db
        env_val = lmdb.open(self.out_val_db_file, map_size=int(1e9))
        txn_val = env_val.begin(write=True)
        for series_id in series_ids_val:
            value = str(txn.get(str(series_id).encode()), encoding='utf-8')
            txn_val.put(key=str(series_id).encode(), value=str(value).encode())
        txn_val.commit()
        env_val.close()

        env.close()
        if self.cfg.DATA_PREPARE.IS_SPLIT_5FOLD:
            self._split_5fold_train_val()

    def _split_5fold_train_val(self):
        raw_train_db = self.out_train_db_file
        raw_val_db = self.out_val_db_file
        new_train_db = raw_train_db.split('_1')[0]
        new_val_db = raw_val_db.split('_1')[0]

        env = lmdb.open(raw_train_db, map_size=int(1e9))
        txn = env.begin()
        default_train_series_uid = []
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            default_train_series_uid.append(key)
        num_train = len(default_train_series_uid)
        new_train_series_uid = [default_train_series_uid[:int(num_train * 0.25)],
                                default_train_series_uid[int(num_train * 0.25):int(num_train * 0.5)],
                                default_train_series_uid[int(num_train * 0.5):int(num_train * 0.75)],
                                default_train_series_uid[int(num_train * 0.75):]]

        env = lmdb.open(raw_val_db, map_size=int(1e9))
        txn = env.begin()
        default_val_series_uid = []
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            default_val_series_uid.append(key)

        env = lmdb.open(self.out_db_file, map_size=int(1e9))
        txn = env.begin()

        for i in range(4):
            out_train_db = new_train_db + '_' + str(i + 2)
            out_val_db = new_val_db + '_' + str(i + 2)
            out_5fold_train = []
            for j in range(4):
                if i != j:
                    out_5fold_train.extend(new_train_series_uid[j])
            out_5fold_train.extend(default_val_series_uid)
            out_5fold_val = new_train_series_uid[i]

            # create train db
            env_train = lmdb.open(out_train_db, map_size=int(1e9))
            txn_train = env_train.begin(write=True)
            for series_id in out_5fold_train:
                value = str(txn.get(str(series_id).encode()), encoding='utf-8')
                txn_train.put(key=str(series_id).encode(), value=str(value).encode())
            txn_train.commit()
            env_train.close()

            # create val db
            env_val = lmdb.open(out_val_db, map_size=int(1e9))
            txn_val = env_val.begin(write=True)
            for series_id in out_5fold_val:
                value = str(txn.get(str(series_id).encode()), encoding='utf-8')
                txn_val.put(key=str(series_id).encode(), value=str(value).encode())
            txn_val.commit()
            env_val.close()
        env.close()

    def process(self, data):
        series_id = data.series_id
        image_info = [load_ct_info(i) for i in data.image_path.values()]
        mask_info = load_ct_info(data.mask_path)
        npy_mask = mask_info['npy_image']
        mask_direction = mask_info['direction']
        mask_spacing = mask_info['spacing']
        print('Start processing %s.' % series_id)


        spacing_list, direction_list, npy_image_list = [], [], []
        for i in image_info:
            spacing_list.append(i['spacing'])
            direction_list.append(i['direction'])
            npy_image_list.append(i['npy_image'])

        assert spacing_list.count(spacing_list[0]) == len(spacing_list), f'{series_id} image spacing is different.'
        image_spacing = spacing_list[0]

        assert direction_list.count(direction_list[0]) == len(direction_list), f'{series_id} image direction is different.'
        image_direction = direction_list[0]

        if self.cfg.DATA_PREPARE.IS_NORMALIZATION_DIRECTION:
            npy_image__list = [change_axes_of_image(i, image_direction) for i in npy_image_list]
            npy_mask = change_axes_of_image(npy_mask, mask_direction)

        image_shape_list = [i['npy_image'].shape for i in image_info]
        assert image_shape_list.count(image_shape_list[0]) == len(image_shape_list), f'Shape of image is not equal in series_id: {series_id}'
        print(image_shape_list)

        if image_shape_list[0] != npy_mask.shape:
            print('Shape of image/mask are not equal in series_id: {}'.format(data.series_id))
            return

        num_label = np.max(np.array(self.mask_label))
        print('End processing %s.' % series_id)

        if self.out_coarse_size is not None:
            coarse_image, _ = ScipyResample.resample_to_size(npy_image, self.out_coarse_size)
            coarse_mask, _ = ScipyResample.resample_mask_to_size(npy_mask, self.out_coarse_size, num_label=num_label)
        else:
            coarse_image, _ = ScipyResample.resample_to_spacing(npy_image, image_spacing, self.out_coarse_spacing)
            coarse_mask, _ = ScipyResample.resample_mask_to_spacing(npy_mask, mask_spacing, self.out_coarse_spacing,
                                                                    num_label=num_label)

        data.coarse_image_path = os.path.join(self.coarse_image_save_dir, series_id)
        data.coarse_mask_path = os.path.join(self.coarse_mask_save_dir, series_id)
        np.save(data.coarse_image_path, coarse_image)
        np.save(data.coarse_mask_path, coarse_mask)
