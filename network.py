'''
Defines classes and methods related to network training
NOTE: Settings are exported to the config file via the Project() class
'''
from enum import Enum
import deeplabcut


class NetworkMode(Enum):
    '''Declares network modes'''
    SINGLE_NETWORK = 'single'
    PER_CAM = 'each'
    RGB = 'rgb'


class SingleNetworkConfig():
    '''Settings for single network xrommtools project'''

    @property
    def network_arch(self):
        return self._network_arch

    def __init__(self,
                 dlc_config_path: str,
                 max_iters: int):

        self._network_arch = 'single'
        self.dlc_config_path = dlc_config_path
        self.max_iters = max_iters
    pass

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'dlc_config_path': self.dlc_config_path,
             'max_iters': self.max_iters,
             'network_arch': 'single'}
        return d

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.dlc_config_path)


class PerCamNetworkConfig():
    '''Settings for per-cam network xrommtools project'''

    def __init__(self,
                 dlc_config_path: str,
                 dlc_config_path_cam2: str,
                 max_iters: int):

        self.dlc_config_path = dlc_config_path
        self.dlc_config_path_cam2 = dlc_config_path_cam2
        self.max_iters = max_iters

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'dlc_config_path': self.dlc_config_path,
             'dlc_config_path_cam2': self.dlc_config_path_cam2,
             'max_iters': self.max_iters,
             'network_arch': 'per_cam'}
        return d

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.dlc_config_path)
        deeplabcut.create_training_dataset(self.dlc_config_path_cam2)


class RGBNetworkConfig():
    '''Settings for RGB network xrommtools project'''

    def __init__(self,
                 dlc_config_path: str,
                 max_iters: int,
                 swapped_markers: bool,
                 crossed_markers: bool):

        self.dlc_config_path = dlc_config_path
        self.max_iters = max_iters
        self.swapped_markers = swapped_markers
        self.crossed_markers = crossed_markers
    pass

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'dlc_config_path': self.dlc_config_path,
             'max_iters': self.max_iters,
             'network_arch': 'rgb',
             'swapped_markers': self.swapped_markers,
             'crossed_markers': self.crossed_markers}
        return d
    pass
