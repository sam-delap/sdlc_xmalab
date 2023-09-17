'''
Defines classes and methods related to network training
NOTE: Settings are exported to the config file via the Project() class
'''
import os
from enum import Enum
import abc
from abc import ABC, abstractmethod
import deeplabcut


class NetworkMode(Enum):
    '''Declares network modes'''
    SINGLE_NETWORK = 'single'
    PER_CAM = 'per cam'
    RGB = 'rgb'

class NetworkConfig(ABC):
    '''
    NetworkConfig type objects interact with DeepLabCut
    and provide a simple way to implement DLC functionality with
    different input layers
    '''
    @abc.abstractproperty
    def network_arch(self):
        '''Name for the network architecture. Should be a NetworkMode'''
        pass

    @abc.abstractproperty
    def dlc_config_path(self):
        '''Path to DLC config'''
        pass

    @abc.abstractproperty
    def dlc_project_path(self):
        '''Path to DLC project'''
        pass

    @abc.abstractproperty
    def max_iters(self):
        '''Max iterations for this type of network'''
        pass

    @abstractmethod
    def create_training_dataset(self):
        '''Create a DeepLabCut training dataset'''
        pass

    @abstractmethod
    def train_network(self):
        '''Train a DeepLabCut network'''
        pass

    @abstractmethod
    def analyze_videos(self):
        '''Analyze novel videos'''
        pass

    @abstractmethod
    def create_labeled_videos(self):
        '''Create labeled videos (for presentations)'''
        pass

    @abstractmethod
    def update_config_refs(self):
        '''Update references to the DLC config'''
        pass

    @abstractmethod
    def to_yaml(self):
        '''Convert object to YAML for saving in an XROMM_DLCTools project'''
        pass

class SingleNetworkConfig(NetworkConfig):
    '''
    Settings for single network xrommtools project
    Credit to J.D. Laurence-Chasen for initial implementation
    '''

    @property
    def network_arch(self) -> NetworkMode:
        return self._network_arch

    @property
    def dlc_config_path(self) -> str:
        return self._dlc_config_path

    @dlc_config_path.setter
    def dlc_config_path(self, new_path: str):
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Couldn't find config at {new_path}")
        self._dlc_config_path = new_path
        self.update_config_refs()

    @property
    def dlc_project_path(self) -> str:
        return self._dlc_project_path

    @property
    def max_iters(self) -> int:
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value: int):
        self._max_iters = value

    def __init__(self,
                 dlc_config_path: str,
                 max_iters: int):

        self._network_arch = NetworkMode.SINGLE_NETWORK
        self._dlc_config_path = dlc_config_path
        self._dlc_project_path = dlc_config_path[:dlc_config_path.find('config')]
        self._max_iters = max_iters

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'network_arch': self._network_arch.value,
             'dlc_config_path': self.dlc_config_path,
             'max_iters': self.max_iters}
             
        return d

    def update_config_refs(self) -> None:
        raise NotImplementedError('Need to update all other config refs')

    def create_training_dataset(self) -> None:
        deeplabcut.create_training_dataset(self.dlc_config_path)

    def train_network(self) -> None:
        deeplabcut.train_network(self.dlc_config_path)


class PerCamNetworkConfig(NetworkConfig):
    '''Settings for per-cam network xrommtools project'''

    @property
    def network_arch(self) -> NetworkMode:
        return self._network_arch

    @property
    def dlc_config_path(self) -> str:
        return self._dlc_config_path

    @dlc_config_path.setter
    def dlc_config_path(self, new_path: str) -> None:
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Couldn't find config at {new_path}")
        self._dlc_config_path = new_path
        self._dlc_project_path = new_path[:new_path.find('config')]
        self.update_config_refs(self._dlc_config_path)

    @property
    def dlc_config_path_cam2(self) -> str:
        return self._dlc_config_path_cam2

    @dlc_config_path_cam2.setter
    def dlc_config_path_cam2(self, new_path: str) -> None:
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Couldn't find config at {new_path}")
        self._dlc_config_path_cam2 = new_path
        self._dlc_project_path_cam2 = new_path[:new_path.find('config')]
        self.update_config_refs(self._dlc_config_path_cam2)

    @property
    def dlc_project_path(self) -> str:
        return self._dlc_project_path

    @property
    def max_iters(self) -> int:
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value: int):
        self._max_iters = value

    def __init__(self,
                 dlc_config_path: str,
                 dlc_config_path_cam2: str,
                 max_iters: int):

        self._network_arch = NetworkMode.PER_CAM
        self._dlc_config_path = dlc_config_path
        self._dlc_project_path = dlc_config_path[:dlc_config_path.find('config')]
        self._dlc_config_path_cam2 = dlc_config_path_cam2
        self._dlc_project_path_cam2 = dlc_config_path_cam2[:dlc_config_path_cam2.find('config')]
        self._max_iters = max_iters

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'network_arch': self.network_arch.value,
             'dlc_config_path': self.dlc_config_path,
             'dlc_config_path_cam2': self.dlc_config_path_cam2,
             'max_iters': self.max_iters}
        return d

    def create_training_dataset(self):
        deeplabcut.create_training_dataset(self.dlc_config_path)
        deeplabcut.create_training_dataset(self.dlc_config_path_cam2)

    def train_network(self):
        deeplabcut.train_network(self.dlc_config_path, maxiters=self.max_iters)
        deeplabcut.train_network(self.dlc_config_path_cam2, maxiters=self.max_iters)

    @staticmethod
    def update_config_refs(new_config_path: str):
        raise NotImplementedError


class RGBNetworkConfig(NetworkConfig):
    '''Settings for RGB network xrommtools project'''

    @property
    def network_arch(self) -> NetworkMode:
        return self._network_arch

    @property
    def dlc_config_path(self) -> str:
        return self._dlc_config_path

    @dlc_config_path.setter
    def dlc_config_path(self, new_path: str):
        if not os.path.exists(new_path):
            raise FileNotFoundError(f"Couldn't find config at {new_path}")
        self._dlc_config_path = new_path
        self.update_config_refs()

    @property
    def dlc_project_path(self) -> str:
        return self._dlc_project_path

    @property
    def max_iters(self) -> int:
        return self._max_iters

    @max_iters.setter
    def max_iters(self, value: int):
        self._max_iters = value

    @property
    def swapped_markers(self) -> bool:
        return self._swapped_markers

    @property
    def crossed_markers(self) -> bool:
        return self._crossed_markers

    def __init__(self,
                 dlc_config_path: str,
                 max_iters: int,
                 swapped_markers: bool,
                 crossed_markers: bool):

        self._network_arch = NetworkMode.RGB
        self._dlc_config_path = dlc_config_path
        self._dlc_project_path = dlc_config_path[:dlc_config_path.find('config')]
        self._max_iters = max_iters
        self._swapped_markers = swapped_markers
        self._crossed_markers = crossed_markers

    def to_yaml(self):
        '''Structure network settings data so it can be sent to yaml'''
        d = {'network_arch': self.network_arch.value,
             'dlc_config_path': self.dlc_config_path,
             'max_iters': self.max_iters,
             'swapped_markers': self.swapped_markers,
             'crossed_markers': self.crossed_markers}
        return d
