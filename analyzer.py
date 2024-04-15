"""Primary interface for analyzing the XROMM data using trained network"""

import os
import math

from itertools import combinations
import cv2
import deeplabcut
import imagehash
import pandas as pd
from PIL import Image
from ruamel.yaml import YAML

import xrommtools
from xma_data_processor import XMADataProcessor


class Analyzer:
    """Analyzes XROMM videos using trained network."""

    def __init__(self, config):
        self._trials_path = os.path.join(config["working_dir"], "trials")
        self._data_processor = XMADataProcessor(config)
        self._config = config
        self._dlc_config = config['path_config_file']

    def analyze_videos(self):
        '''Analyze videos with a pre-existing network'''
        trials = self._data_processor.list_trials()

        # Establish project vars
        yaml = YAML()
        with open(self._dlc_config) as dlc_config:
            dlc = yaml.load(dlc_config)
        iteration = dlc['iteration']

        mode = self._config['tracking_mode']
        if mode == '2D':
            xrommtools.analyze_xromm_videos(self._dlc_config, self._trials_path, iteration)
        elif mode == 'per_cam':
            xrommtools.analyze_xromm_videos(path_config_file=self._dlc_config,
                                            path_config_file_cam2=self._config['path_config_file_2'],
                                            path_data_to_analyze=self._trials_path,
                                            iteration=iteration,
                                            nnetworks=2)
        else:
            for trial in trials:
                trial_path =  os.path.join(self._trials_path, trial)
                video_path = f'{trial_path}/{trial}_rgb.avi'
                if not os.path.exists(video_path):
                    self._data_processor.make_rgb_video(self._trials_path)
                destfolder = f'{trial_path}/it{iteration}/'
                deeplabcut.analyze_videos(self._dlc_config, video_path, destfolder=destfolder, save_as_csv=True)
                self._split_dlc_to_xma(trial)

    def analyze_video_similarity_project(self):
        '''Analyze all videos in a project and take their average similar. This is dangerous, as it will assume that all cam1/cam2 pairs match
        or don't match!'''
        similarity_score = {}
        yaml = YAML()
        trial_perms = combinations(self._data_processor.list_trials(), 2)
        for trial1, trial2 in trial_perms:
            self._config['trial_1_name'] = trial1
            self._config['trial_2_name'] = trial2
            with open(os.path.join(self._config['working_dir'], 'project_config.yaml'), 'w') as file:
                yaml.dump(self._config, file)
            # TODO: pass the trial names directly instead of writing into config
            similarity_score[(trial1, trial2)] = self.analyze_video_similarity_trial()
        return similarity_score

    def analyze_video_similarity_trial(self):
        '''Analyze the average similarity between trials using image hashing'''
        trials = [self._config['trial_1_name'], self._config['trial_2_name']]
        cameras = ['cam1', 'cam2']
        videos = {
            (trial, cam): cv2.VideoCapture(
                self._data_processor.find_cam_file(os.path.join(self._trials_path, trial), cam))
            for trial in trials for cam in cameras
        }

        # Compare hashes based on the camera views configuration
        if self._config['cam1s_are_the_same_view']:
            cam1_diff, noc1 = self._compare_two_videos(videos[(trials[0], 'cam1')], videos[(trials[1], 'cam1')])
            cam2_diff, noc2 = self._compare_two_videos(videos[(trials[0], 'cam2')], videos[(trials[1], 'cam2')])
        else:
            cam1_diff, noc1 = self._compare_two_videos(videos[(trials[0], 'cam1')], videos[(trials[1], 'cam2')])
            cam2_diff, noc2 = self._compare_two_videos(videos[(trials[0], 'cam2')], videos[(trials[1], 'cam1')])

        # Calculate the average difference for each camera view
        cam1_avg_diff = cam1_diff / noc1 if noc1 > 0 else 0
        cam2_avg_diff = cam2_diff / noc2 if noc2 > 0 else 0

        # Calculate the overall trial average difference
        trial_avg_diff = (cam1_avg_diff + cam2_avg_diff) / 2

        # Note: The number of comparisons (noc) grows with video size, which could potentially affect the similarity measure,
        # making larger videos appear more similar than they actually are. This aspect may need further consideration.
        return trial_avg_diff

    def analyze_marker_similarity_project(self):
        '''Analyze all videos in a project and get their average rhythmicity. This assumes that all cam1/2 pairs are either the same or different!'''
        marker_similarity = {}
        yaml = YAML()

        trial_perms = combinations(self._data_processor.list_trials(), 2)
        for trial1, trial2 in trial_perms:
            self._config['trial_1_name'] = trial1
            self._config['trial_2_name'] = trial2
            with open(os.path.join(self._config['working_dir'], 'project_config.yaml'), 'w') as file:
                yaml.dump(self._config, file)
            # TODO: pass the trial names directly instead of writing into config
            marker_similarity[(trial1, trial2)] = abs(self.analyze_marker_similarity_trial())
        return marker_similarity

    def analyze_marker_similarity_trial(self):
        '''Analyze marker similarity for a pair of trials using the distance formula.'''
        # Find CSVs for each trial
        trial1 = self._config['trial_1_name']
        trial2 = self._config['trial_2_name']
        trial1_path = os.path.join(self._trials_path, trial1)
        trial2_path = os.path.join(self._trials_path, trial2)

        # Get a list of markers that each trial have in common
        bodyparts1 = self._data_processor.get_bodyparts_from_xma(trial1_path, mode='rgb')
        bodyparts2 = self._data_processor.get_bodyparts_from_xma(trial2_path, mode='rgb')
        markers_in_common = [marker for marker in bodyparts1 if marker in bodyparts2]

        trial1_csv = pd.read_csv(os.path.join(trial1_path, f'{trial1}.csv'))
        trial2_csv = pd.read_csv(os.path.join(trial2_path, f'{trial2}.csv'))

        avg_distances = []
        for marker in markers_in_common:
            avg_x1, avg_y1 = trial1_csv[f'{marker}_X'].mean(), trial1_csv[f'{marker}_Y'].mean()
            avg_x2, avg_y2 = trial2_csv[f'{marker}_X'].mean(), trial2_csv[f'{marker}_Y'].mean()
  
            # Calculate the distance between the average positions for this marker in the two trials
            distance = math.sqrt((avg_x2 - avg_x1) ** 2 + (avg_y2 - avg_y1) ** 2)
            avg_distances.append(distance)

        # Calculate the mean of the distances to get an overall similarity measure
        marker_similarity = sum(avg_distances) / len(avg_distances) if avg_distances else 0

        return marker_similarity

    def get_max_dissimilarity_for_trial(self, trial_path, window):
        '''Calculate the dissimilarity within the trial given the frame sliding window.'''        
        video1 = cv2.VideoCapture(self._data_processor.find_cam_file(trial_path, 'cam1'))
        video2 = cv2.VideoCapture(self._data_processor.find_cam_file(trial_path, 'cam2'))

        hashes1 = self._hash_trial_video(video1)
        hashes2 = self._hash_trial_video(video2)

        return self._find_dissimilar_regions(hashes1, hashes2, window)

    def _split_dlc_to_xma(self, trial, save_hdf=True):
        '''Takes the output from RGB deeplabcut and splits it into XMAlab-readable output'''
        bodyparts_xy = []
        yaml = YAML()
        with open(self._dlc_config) as dlc_config:
            dlc = yaml.load(dlc_config)
        iteration = dlc['iteration']
        trial_path = os.path.join(self._trials_path, trial)

        rgb_parts = self._data_processor.get_bodyparts_from_xma(trial_path, mode='rgb')
        for part in rgb_parts:
            bodyparts_xy.append(part+'_X')
            bodyparts_xy.append(part+'_Y')

        csv_path = [file for file in os.listdir(f'{trial_path}/it{iteration}') if '.csv' in file and '-2DPoints' not in file]
        if len(csv_path) > 1:
            raise FileExistsError('Found more than 1 data CSV for RGB trial. Please remove CSVs from older analyses from this folder before analyzing.')
        if len(csv_path) < 1:
            raise FileNotFoundError(f'Couldn\'t find data CSV for trial {trial}. Something wrong with DeepLabCut?')

        csv_path = csv_path[0]
        xma_csv_path = f'{trial_path}/it{iteration}/{trial}-Predicted2DPoints.csv'

        df = pd.read_csv(f'{trial_path}/it{iteration}/{csv_path}', skiprows=1)
        df.index = df['bodyparts']
        df = df.drop(columns=df.columns[df.loc['coords'] == 'likelihood'])
        df = df.drop(columns=[column for column in df.columns if column not in rgb_parts and column not in [f'{bodypart}.1' for bodypart in rgb_parts]])
        df.columns = bodyparts_xy
        df = df.drop(index='coords')
        df.to_csv(xma_csv_path, index=False)
        print("Successfully split DLC format to XMALab 2D points; saved "+str(xma_csv_path))
        if save_hdf:
            tracked_hdf = os.path.splitext(csv_path)[0]+'.h5'
            df.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w', nan_rep='NaN')

    def _compare_two_videos(self, video1, video2):
        '''Compare two videos using image hashing'''
        hashes1 = self._hash_trial_video(video1)
        hashes2 = self._hash_trial_video(video2)

        video1_frames = len(hashes1)
        video2_frames = len(hashes2)
        noc = math.perm(video1_frames + video2_frames, 2)  # Might need revision based on actual comparison logic

        print(f'Video 1 frames: {video1_frames}')
        print(f'Video 2 frames: {video2_frames}')

        print('Comparing hashes between videos')
        hash_dif = sum(hash1 - hash2 for hash1 in hashes1 for hash2 in hashes2)

        return hash_dif, noc

    def _hash_trial_video(self, video):
        '''Generate image hashes for a single video'''
        video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video frames: {video_frames}')

        hashes = []
        print('Creating hashes for video')
        for i in range(video_frames):
            print(f'Current frame: {i}')
            ret, frame = video.read()
            if not ret:
                print('Error reading video frame')
                cv2.destroyAllWindows()
                break
            hashes.append(imagehash.phash(Image.fromarray(frame)))

        return hashes

    def _find_dissimilar_regions(self, hashes1, hashes2, window):
        '''Find the region of maximum dissimilarity given 2 lists of hashes and a sliding window (how many frames)'''
        start_frame_vid1 = 0
        start_frame_vid2 = 0
        max_hash_dif_vid1 = 0
        max_hash_dif_vid2 = 0
        hash_dif_vid1 = 0
        hash_dif_vid2 = 0

        for slider in range(0, len(hashes1) // window):
            print(f'Current start frame {slider * window}')
            hash_dif_vid1, hash_dif_vid2 = self._compare_hash_sets(hashes1[slider * window:(slider + 1) * window], hashes2[slider * window:(slider + 1) * window])

            print(f'Current hash diff (vid 1): {hash_dif_vid1}')
            print(f'Current hash diff (vid 2): {hash_dif_vid2}')
            if hash_dif_vid1 > max_hash_dif_vid1:
                max_hash_dif_vid1 = hash_dif_vid1
                start_frame_vid1 = slider * window

            if hash_dif_vid2 > max_hash_dif_vid2:
                max_hash_dif_vid2 = hash_dif_vid2
                start_frame_vid2 = slider * window

            print(f'Max hash diff (vid 1): {max_hash_dif_vid1}')
            print(f'Max hash diff (vid 2): {max_hash_dif_vid2}')

            print(f'Start frame (vid 1): {start_frame_vid1}')
            print(f'Start frame (vid 2): {start_frame_vid2}')

        return start_frame_vid1, start_frame_vid2

    def _compare_hash_sets(self, hashes1, hashes2):
        '''Compares two sets of image hashes to find dissimilarities'''
        hash1_dif = 0
        hash2_dif = 0

        print(f'Hash set 1 {hashes1[0]}')
        print(f'Hash set 2 {hashes2[0]}')
        # Compares all possible combinations of images
        for combination in combinations(hashes1, 2):
            hash1_dif = hash1_dif + (combination[0] - combination[1])

        for combination in combinations(hashes2, 2):
            hash2_dif = hash2_dif + (combination[0] - combination[1])

        return hash1_dif, hash2_dif