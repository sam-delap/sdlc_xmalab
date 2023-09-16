'''Create, read, and modify xrommtools projects'''
import os
from ruamel.yaml import YAML
from network import SingleNetworkConfig, PerCamNetworkConfig, RGBNetworkConfig
from network import NetworkMode
from data import AutocorrectSettings


class Project():
    '''Contains information about an xrommtools project'''

    def __init__(self,
                 task: str,
                 config_path: str,
                 experimenter: str,
                 network: SingleNetworkConfig | PerCamNetworkConfig | RGBNetworkConfig,
                 autocorrect_settings=AutocorrectSettings()):
        '''Create and initialize a new xrommtools project'''
        self._task = task
        self._config_path = config_path
        self.experimenter = experimenter
        self._training_data_path = os.path.join(self.config_path,
                                                'trainingdata')
        self._novel_data_path = os.path.join(self.config_path,
                                             'trials')
        self._network = network
        self._autocorrect_settings = autocorrect_settings

    @property
    def task(self):
        return self._task

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        if not os.path.exists(value):
            raise FileNotFoundError(f"Couldn't find config at {value}")
        raise NotImplementedError('Need to update all other config refs')
        self.update_config_refs()

    @property
    def training_data_path(self):
        return self._training_data_path

    @property
    def novel_data_path(self):
        return self._novel_data_path

    @property
    def network(self):
        return self._network

    @property
    def autocorrect_settings(self):
        return self._autocorrect_settings

    def update_config_refs(self):
        self._training_data_path = os.path.join(self.config_path,
                                                'trainingdata')
        self._novel_data_path = os.path.join(self.config_path,
                                             'trials')

    @classmethod
    def from_yaml(cls, config_path: str):
        '''Load an existing project using the YAML config'''
        yaml = YAML()
        with open(config_path, "r") as f:
            d = yaml.load(f)

        match(d['network']['network_arch']):
            case NetworkMode.SINGLE_NETWORK:
                network = SingleNetworkConfig(d['network']['dlc_config_path'],
                                              d['network']['max_iters'])
            case NetworkMode.PER_CAM:
                network = PerCamNetworkConfig(d['network']['dlc_config_path'],
                                              d['network']['dlc_config_path_cam2'],
                                              d['network']['max_iters'])
            case NetworkMode.RGB:
                network = RGBNetworkConfig(d['network']['dlc_config_path'],
                                           d['network']['max_iters'],
                                           d['network']['swapped_markers'],
                                           d['network']['crossed_markers'])
            case _:
                raise SyntaxError('Invalid value for network arch',
                                  'Network arch must be one of',
                                  [arch.value for arch in NetworkMode])

        return cls(d['project']['task'],
                   d['project']['config_path'],
                   d['project']['experimenter'],
                   network)

    def to_yaml(self):
        '''Save config to a file'''
        yaml_loader = YAML()
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                d = yaml_loader.load(f)
        else:
            d = {}
        d['project'] = {'task': self.task,
                        'config_path': self.config_path,
                        'experimenter': self.experimenter}
        d['network'] = self.network.to_yaml()
        d['autocorrect'] = self.autocorrect_settings.to_yaml()

        yaml_dump = YAML()
        with open(self.config_path, "w+") as f:
            yaml_dump.dump(d, f)
