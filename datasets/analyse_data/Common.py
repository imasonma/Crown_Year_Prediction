import lmdb
import json
import os

import numpy as np
import SimpleITK as sitk


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.__str__()
        else:
            return super(MyEncoder, self).default(obj)


def read_txt(txt_file):
    txt_lines = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            txt_lines.append(line)
    return txt_lines


class DataBaseUtils(object):
    def __init__(self):
        super(DataBaseUtils, self).__init__()

    @staticmethod
    def creat_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.commit()
        env.close()

    @staticmethod
    def update_record_in_db(db_dir, idx, data_dict):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.put(str(idx).encode(), value=json.dumps(data_dict, cls=MyEncoder).encode())
        txn.commit()
        env.close()

    @staticmethod
    def delete_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.delete(str(idx).encode())
        txn.commit()
        env.close()

    @staticmethod
    def get_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        value = txn.get(str(idx).encode())
        env.close()
        if value is None:
            return None
        value = str(value, encoding='utf-8')
        data_dict = json.loads(value)

        return data_dict

    @staticmethod
    def read_records_in_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        out_records = dict()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            out_records[key] = label_info
        env.close()

        return out_records

    @staticmethod
    def write_records_in_db(db_dir, in_records):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        for key, value in in_records.items():
            txn.put(str(key).encode(), value=json.dumps(value, cls=MyEncoder).encode())
        txn.commit()
        env.close()


def load_ct_info(file_path, sort_by_distance=True):
    sitk_image = None
    try:
        if os.path.isdir(file_path):
            reader = sitk.ImageSeriesReader()
            series_ids = reader.GetGDCMSeriesIDs(file_path)
            dcm_series = reader.GetGDCMSeriesFileNames(file_path, series_ids[0])
            reader.SetFileNames(dcm_series)
            sitk_image = load_ct_from_dicom(file_path, sort_by_distance)
        else:
            sitk_image = sitk.ReadImage(file_path)
    except Exception as err:
        print('load ct throws exception %s, with file %s!' % (err, file_path))

    if sitk_image is None:
        res = {}
    else:
        origin = list(reversed(sitk_image.GetOrigin()))
        spacing = list(reversed(sitk_image.GetSpacing()))
        direction = sitk_image.GetDirection()
        direction = [direction[8], direction[4], direction[0]]
        res = {"sitk_image": sitk_image,
               "npy_image": sitk.GetArrayFromImage(sitk_image),
               "origin": origin,
               "spacing": spacing,
               "direction": direction}
    return res


def load_ct_from_dicom(dcm_path, sort_by_distance=True):
    class DcmInfo(object):
        def __init__(self, dcm_path, series_instance_uid, acquisition_number, sop_instance_uid, instance_number,
                     image_orientation_patient, image_position_patient):
            super(DcmInfo, self).__init__()

            self.dcm_path = dcm_path
            self.series_instance_uid = series_instance_uid
            self.acquisition_number = acquisition_number
            self.sop_instance_uid = sop_instance_uid
            self.instance_number = instance_number
            self.image_orientation_patient = image_orientation_patient
            self.image_position_patient = image_position_patient

            self.slice_distance = self._cal_distance()

        def _cal_distance(self):
            normal = [self.image_orientation_patient[1] * self.image_orientation_patient[5] -
                      self.image_orientation_patient[2] * self.image_orientation_patient[4],
                      self.image_orientation_patient[2] * self.image_orientation_patient[3] -
                      self.image_orientation_patient[0] * self.image_orientation_patient[5],
                      self.image_orientation_patient[0] * self.image_orientation_patient[4] -
                      self.image_orientation_patient[1] * self.image_orientation_patient[3]]

            distance = 0
            for i in range(3):
                distance += normal[i] * self.image_position_patient[i]
            return distance

    def is_sop_instance_uid_exist(dcm_info, dcm_infos):
        for item in dcm_infos:
            if dcm_info.sop_instance_uid == item.sop_instance_uid:
                return True
        return False

    def get_dcm_path(dcm_info):
        return dcm_info.dcm_path

    reader = sitk.ImageSeriesReader()
    if sort_by_distance:
        dcm_infos = []

        files = os.listdir(dcm_path)
        for file in files:
            file_path = os.path.join(dcm_path, file)

            dcm = dicomio.read_file(file_path, force=True)
            _series_instance_uid = dcm.SeriesInstanceUID
            _acquisition_number = dcm.AcquisitionNumber
            _sop_instance_uid = dcm.SOPInstanceUID
            _instance_number = dcm.InstanceNumber
            _image_orientation_patient = dcm.ImageOrientationPatient
            _image_position_patient = dcm.ImagePositionPatient

            dcm_info = DcmInfo(file_path, _series_instance_uid, _acquisition_number, _sop_instance_uid,
                               _instance_number, _image_orientation_patient, _image_position_patient)

            if is_sop_instance_uid_exist(dcm_info, dcm_infos):
                continue

            dcm_infos.append(dcm_info)

        dcm_infos.sort(key=lambda x: x.slice_distance)
        dcm_series = list(map(get_dcm_path, dcm_infos))
    else:
        dcm_series = reader.GetGDCMSeriesFileNames(dcm_path)

    reader.SetFileNames(dcm_series)
    sitk_image = reader.Execute()
    return sitk_image


def save_ct_from_sitk(sitk_image, save_path, sitk_type=None, use_compression=False):
    if sitk_type is not None:
        sitk_image = sitk.Cast(sitk_image, sitk_type)
    sitk.WriteImage(sitk_image, save_path, use_compression)


def save_ct_from_npy(npy_image, save_path, origin=None, spacing=None,
                     direction=None, sitk_type=None, use_compression=False):
    sitk_image = sitk.GetImageFromArray(npy_image)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    save_ct_from_sitk(sitk_image, save_path, sitk_type, use_compression)


def dcm_2_mha(dcm_path, mha_path, use_compress):
    res = load_ct_info(dcm_path)
    sitk.WriteImage(res['sitk_image'], mha_path, use_compress)


def convert_npy_image_2_sitk(npy_image, origin=None, spacing=None,
                             direction=None, sitk_type=sitk.sitkFloat32):
    sitk_image = sitk.GetImageFromArray(npy_image)
    sitk_image = sitk.Cast(sitk_image, sitk_type)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    return sitk_image

def convert_npy_mask_2_sitk(npy_mask, label=None, origin=None, spacing=None,
                            direction=None, sitk_type=sitk.sitkUInt8):
    if label is not None:
        npy_mask[npy_mask != 0] = label
    sitk_mask = sitk.GetImageFromArray(npy_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk_type)
    if origin is not None:
        sitk_mask.SetOrigin(origin)
    if spacing is not None:
        sitk_mask.SetSpacing(spacing)
    if direction is not None:
        sitk_mask.SetDirection(direction)
    return sitk_mask

def change_axes_of_image(npy_image, orientation):
    '''default orientation=[1, -1, -1]'''
    if orientation[0] < 0:
        npy_image = np.flip(npy_image, axis=0)
    if orientation[1] > 0:
        npy_image = np.flip(npy_image, axis=1)
    if orientation[2] > 0:
        npy_image = np.flip(npy_image, axis=2)
    return npy_image
