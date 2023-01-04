import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor, logger


def image_padding(img, lower_bound, upper_bound, constant=0):
    filter = sitk.ConstantPadImageFilter()
    filter.SetPadLowerBound(lower_bound)
    filter.SetPadUpperBound(upper_bound)
    filter.SetConstant(constant)
    padded_img = filter.Execute(img)
    return padded_img


def radiomics2D(data: np.ndarray, kernel_size: int = 3, params: str, padding: int = 1, stride: int = 1) -> np.ndarray:
    """radiomics2D.

    Args:
        data (np.ndarray): data
        kernel_size (int): kernel_size
        params (str): params
        padding (int): padding
        stride (int): stride

    Returns:
        np.ndarray:
    """
    if padding != 0:
        image_padded = image_padding(image2d, (padding, padding), (padding, padding), 0)
    else:
        image_padded = image2d

    x_img_shape = image2d.GetSize()[0]
    y_img_shape = image2d.GetSize()[1]

    mask2d = np.zeros((image_padded.GetSize()[1], image_padded.GetSize()[0]))
    mask2d = sitk.GetImageFromArray(mask2d)
    mask2d.SetSpacing(image_padded.GetSpacing())
    mask2d.SetOrigin(image_padded.GetOrigin())
    mask2d.SetDirection(image_padded.GetDirection())
    mask2d = sitk.Cast(mask2d, sitk.sitkInt32)

    output = np.zeros((x_img_shape, y_img_shape))

    extractor = featureextractor.RadiomicsFeatureExtractor(param)

    for y in range(0, y_img_shape - kernel_size, stride):
        for x in range(0, x_img_shape - kernel_size, stride):
            mask_patch = deepcopy(mask2d)
            mask_patch[x : x + kernel_size, y : y + kernel_size] = 1
            try:
                result = extractor.execute(image_padded, mask_patch)
                feature = {key: val for key, val in six.iteritems(result) if key.find('diagnostics') == -1}
                feat_keys = list(feature.keys())
                if not feat_names:
                    feat_names = feat_keys
                    output = np.repeat(output[:, :, np.newaxis], len(feat_names), axis=2)  # broadcasting
                else:
                    assert feat_names == list(feature.keys())  # ensure feature order
                output[x, y] = np.array(list(feature.values()))
            except:
                pass

def build_radimics(data):
    return radiomicsConv2D(data, kernel_size=3, param=param_path, padding=1, stride=1)
