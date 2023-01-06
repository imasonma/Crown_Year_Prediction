import os
import sys
import json
import traceback

import lmdb
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split

from Common import DataBaseUtils


def run_prepare_data(cfg):
    data_prepare = DataPrepare(cfg)
    pass


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
            tmp_data = MaskData(key,
                                label_info['image_path'],
                                label_info['mask_path'])
            local_data.append(tmp_data)
        env.close()

        print('Num of ct is %d.' % len(local_data))
        return local_data

    def _create_modal_info(self, image_dir, parse, data_dict = dict()):
        for i in self.cfg.DATA_PREPARE.IMAGE_MODAL:
            image_path = os.path.join(self.cfg.ENVIRONMENT.DATA_BASE_DIR, image_dir, i, series_id + '_0000' + parse)
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
                mask_path = os.path.join(self.cfg.DATA_PREPARE.TRAIN_MASK_DIR, series_id + parse)
                if os.path.exists(mask_path):
                    data_dict = {
                        'mask_path': mask_path
                    }
                    data_dict = self._create_modal_info(self.cfg.DATA_PREPARE.TRAIN_IMAGE_DIR, parse, data_dict)
                    DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)
                else:
                    print('%s has invalid image/mask.' % series_id)
            else:
                image_path = os.path.join(self.cfg.DATA_PREPARE.TEST_IMAGE_DIR, series_id + '_0000' + parse)
                mask_path = os.path.join(self.cfg.DATA_PREPARE.TEST_MASK_DIR, series_id + parse) \
                    if self.cfg.DATA_PREPARE.TEST_MASK_DIR is not None else None
                data_dict = self._create_modal_info(self.cfg.DATA_PREPARE.TEST_IMAGE_DIR, parse)
                data_dict['mask_path'] = mask_path
                DataBaseUtils.update_record_in_db(db_file_path, series_id, data_dict)
