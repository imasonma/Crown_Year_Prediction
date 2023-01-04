from monai.networks import nets

def build_model(config):
    name = config.MODEL.TYPE

    dimension = config.DATA.DIMENSION
    in_channels = config.DATA.IN_CHANNELS
    num_classes = config.DATA.NUM_CLASSES

    if name == 'unet':
        model = nets.BasicUNet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, norm='batch', dropout=0.2)
    if name == 'attunet':
        model = nets.AttentionUnet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, channels = (32, 64, 128, 256, 32), strides=(2,2,2,2), dropout=0.1)
    if name == 'flexunet':
        model = nets.FlexibleUNet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, backbone='efficientnet-b8', dropout=0.2, pretrained=True)
    if name == 'crossunet':
        from net import CrossUNet
        model = CrossUNet.BasicUNet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, norm='batch', dropout=0.2)
    if name == 'unetplus':
        from net import unetplus
        model = unetplus.BasicUNetPlusPlus(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, norm='batch', dropout=0.2)
    if name == 'swin':
        model = nets.SwinUNETR(img_size=(512, 512), spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, norm_name='batch', drop_rate=0.2)
    if name == 'nnunet':
        model = nets.DynUNet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, kernel_size=[3, 3, 3, 1], strides=[1, 1, 1, 1], upsample_kernel_size=[1, 1, 1], norm_name="instance")
    if name == 'rminet':
        from net import rminet
        model = rminet.BasicUNet(spatial_dims=dimension, in_channels=in_channels, out_channels=num_classes, norm='batch', dropout=0.2)
    if name == 'vbnet':
        from net import vbnet
        model = vbnet.SegmentationNet(in_channels, num_classes)
    if name == 'fuunet':
        from net import FuseUNet_share_only_CT
        model = FuseUNet_share_only_CT.BasicUNet(2, in_channels, num_classes)

    return model
