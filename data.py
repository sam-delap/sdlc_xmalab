'''Manipulate and clean user data'''
import os
import cv2
import pandas as pd
import numpy as np
import abc
from abc import ABC, abstractmethod
import blend_modes
from enum import Enum
from ruamel.yaml import YAML
from network import NetworkMode


class RGBStrategy(Enum):
    '''Declares what to do with the blue channel of the merged video'''
    EMPTY = 'empty'
    DIFF = 'diff'
    MULTIPLY = 'multiply'

class AutocorrectSettings():
    '''Defines all possible settings for autocorrect'''
    def __init__(self,
                 search_radius=15,
                 threshold=8,
                 krad=17,
                 gsigma=10,
                 img_wt=3.6,
                 blur_wt=-2.9,
                 gamma=0.1,
                 testing=False,
                 trial_name='Your Trial Here',
                 cam='cam1',
                 marker='Your Marker Here',
                 frame_number=1):
        if search_radius < 10:
            raise ValueError('Search radius must be at least 10 pixels')
        if threshold < 0 or threshold > 255:
            raise ValueError('Binary threshold must be a grayscale value'
                             + 'between 0 and 255')
        self.search_radius = search_radius
        self.threshold = threshold
        self.krad = krad
        self.gsigma = gsigma
        self.img_wt = img_wt
        self.blur_wt = blur_wt
        self.gamma = gamma
        self.testing = testing
        self.trial_name = trial_name
        self.cam = cam
        self.marker = marker
        self.frame_number = frame_number

    @classmethod
    def from_yaml(cls, config_path: str):
        yaml = YAML()
        with open(config_path, "r") as f:
            project = yaml.load(f)
        d = project['autocorrect']
        return cls(d['image']['search_radius'],
                   d['image']['threshold'],
                   d['image']['krad'],
                   d['image']['gsigma'],
                   d['image']['img_wt'],
                   d['image']['blur_wt'],
                   d['image']['gamma'],
                   d['testing']['testing'],
                   d['testing']['trial_name'],
                   d['testing']['cam'],
                   d['testing']['marker'],
                   d['testing']['frame_num'])

    def to_yaml(self):
        return {'image': {'search_radius':self.search_radius,
                          'threshold':self.threshold,
                          'krad':self.krad,
                          'gsigma':self.gsigma,
                          'img_wt':self.img_wt,
                          'blur_wt':self.blur_wt,
                          'gamma':self.gamma},
                'testing': {'testing':self.testing,
                            'trial_name':self.trial_name,
                            'cam':self.cam,
                            'marker':self.marker,
                            'frame_number':self.frame_number}}

        


class Trial(ABC):
    @abc.abstractproperty
    def trial_path(self):
       pass 

    @abc.abstractproperty
    def trial_name(self):
        pass

    @abc.abstractproperty
    def cam1_path(self):
        pass

    @abc.abstractproperty
    def cam2_path(self):
        pass

    @abc.abstractproperty
    def csv_path(self):
        pass

    @abc.abstractproperty
    def csv(self):
        pass

    @abc.abstractproperty
    def bodyparts(self):
        pass

    @abstractmethod
    def vid_to_pngs(self):
        pass

    @abstractmethod
    def xma_to_dlc(self):
        pass

    @abstractmethod
    def dlc_to_xma(self):
        pass

    @abstractmethod
    def update_h5(self):
        pass

class XrommToolsTrial(Trial):
    '''Credit to J.D. Laurence-Chasen for the initial design'''
    @property
    def trial_path(self):
        return self._trial_path

    @property
    def trial_name(self):
        return self._trial_name

    @property
    def cam1_path(self):
        return self._cam1_path

    @property
    def cam2_path(self):
        return self._cam2_path

    @property
    def csv_path(self):
        return self._csv_path

    @property
    def csv(self):
        return self._csv

    @property
    def bodyparts(self):
        return self._bodyparts

    def __init__(self,
                 trial_path: str):
        self._trial_path = trial_path
        self._trial_name = os.path.basename(trial_path)
        contents = os.listdir(trial_path)
        for name in contents:
            if 'cam1' in name:
                self._cam1_path = os.path.join(trial_path, name)
            elif 'cam2' in name:
                self._cam2_path = os.path.join(trial_path, name)
            elif '.csv' in name:
                self._csv_path = os.path.join(trial_path, name)

        if self._cam1_path is None:
            raise FileNotFoundError('Couldn\'t find cam1 data')
        elif self._cam2_path is None:
            raise FileNotFoundError('Couldn\'t find cam2 data')
        elif self._csv_path is None:
            raise FileNotFoundError('Couldn\'t find pointsfile')

        self._csv = pd.read_csv(self.csv_path)
        self._csv = self._csv.dropna(how='all')

        names = self._csv.columns.values
        parts = [name.rsplit('_', 2)[0] for name in names]
        self._bodyparts = []
        for part in parts:
            if part not in self._bodyparts:
                self._bodyparts.append(part)

        # Error handling
        if self.csv.isna().sum().sum() > 0:
            raise AttributeError('Detected',
                                 f'{len(self.csv) - len(self.csv.dropna())}',
                                 'partially tracked frames.')

    def vid_to_pngs(self):
        '''Handle conversion from video to labeled-data PNGs'''

    def dlc_to_xma(self, savepath: str,
                   cam1_h5_path: str,
                   cam2_h5_path: str):
        '''Convert DeepLabCut-formatted marker data to XMAlab'''
        h5_save_path = os.path.join(savepath,
                                    self.trial_name
                                    + "-Predicted2DPoints.h5")
        csv_save_path = os.path.join(savepath,
                                     self.trial_name
                                     + "-Predicted2DPoints.csv")

        cam1data = pd.read_hdf(cam1_h5_path)
        cam2data = pd.read_hdf(cam2_h5_path)
        parts = cam1data.columns.get_level_values('bodyparts').unique()

        # make new column names
        nparts = len(list(parts))
        parts = [item for item in parts for repetitions in range(4)]
        post = ["_cam1_X", "_cam1_Y", "_cam2_X", "_cam2_Y"]*nparts
        cols = [m+str(n) for m, n in zip(parts, post)]

        # remove likelihood columns
        cam1data = cam1data.drop(cam1data.columns[2::3], axis=1)
        cam2data = cam2data.drop(cam2data.columns[2::3], axis=1)

        # replace col names with new indices
        c1cols = list(range(0,
                            cam1data.shape[1]*2,
                            4)) + list(range(1, cam1data.shape[1]*2, 4))
        c2cols = list(range(2,
                            cam2data.shape[1]*2,
                            4)) + list(range(3, cam1data.shape[1]*2, 4))
        c1cols.sort()
        c2cols.sort()
        cam1data.columns = c1cols
        cam2data.columns = c2cols

        df = pd.concat([cam1data, cam2data],
                       axis=1).sort_index(axis=1)
        df.columns = cols
        df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
        df.to_csv(csv_save_path, na_rep='NaN', index=False)
        print("Saved h5 (DLC) and human-readable (CSV) data versions.",
              "If you need to update values in the CSV, run update_h5")

    def update_h5(self):
        raise NotImplementedError


class SingleNetworkTrial(XrommToolsTrial):
    '''J.D. Single Network (both views) workflow'''

    def xma_to_dlc(self,
                   dlc_project_path: str,
                   dataset_name: str,
                   experimenter: str):
        '''Formats XMAlab pointsfile for DLC, extracts corresponding frames'''
        relnames = []
        print(self.csv.index)
        new_path = os.path.join(dlc_project_path,
                                "labeled-data",
                                dataset_name)
        h5_save_path = os.path.join(new_path,
                                    f"CollectedData_{experimenter}.h5")
        csv_save_path = os.path.join(new_path,
                                     f"CollectedData_{experimenter}.csv")
        try:
            contents = os.listdir(new_path)
        except FileNotFoundError:
            os.makedirs(new_path)
            contents = None

        if contents is not None:
            raise FileExistsError(f'Dataset folder {dataset_name} exists')

        for i, video_path in enumerate([self.cam1_path, self.cam2_path], 1):
            print(f"Extracting camera {i} trial images...")

            # extract 2D points data
            xpos = self.csv.iloc[:, 0 + (i-1)*2::4]
            ypos = self.csv.iloc[:, 1 + (i-1)*2::4]
            data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)

            relpath = os.path.join("labeled-data", dataset_name)
            max_index = int(self.csv.index.max())
            # if video file is actually folder of frames
            if os.path.isdir(video_path):
                imgs = os.listdir(video_path)

                # Assumes images are in order within folder
                for count, img in enumerate(imgs):
                    if count > max_index:
                        break
                    elif count not in self.csv.index:
                        continue

                    frame_num = str(count + 1).zfill(4)
                    image = cv2.imread(os.path.join(video_path, img))
                    new_img_name = self.trial_name + f"_cam{i}_{frame_num}.png"

                    relname = os.path.join(relpath, new_img_name)
                    relnames.append(relname)

                    image_path = os.path.join(new_path, new_img_name)
                    cv2.imwrite(image_path, image)

            # extract frames from video and convert to png
            else:
                cap = cv2.VideoCapture(video_path)
                success, image = cap.read()
                count = 0
                while success:
                    if count > max_index:
                        break
                    elif count not in self.csv.index:
                        continue

                    frame_num = str(count + 1).zfill(4)
                    new_img_name = self.trial_name + f"_cam{i}_{frame_num}.png"

                    relname = os.path.join(relpath, new_img_name)
                    relnames.append(relname)

                    image_path = os.path.join(new_path, new_img_name)
                    cv2.imwrite(image_path, image)

                    success, image = cap.read()
                    count += 1
                cap.release()

            # Format data
            print('Extracting 2D Points...')
            print(relnames)
            temp = np.empty((data.shape[0], 2,))
            temp[:] = np.nan
            for i, bodypart in enumerate(self.bodyparts):
                index = pd.MultiIndex.from_product([[experimenter],
                                                    [bodypart],
                                                    ['x', 'y']],
                                                   names=['scorer',
                                                          'bodyparts',
                                                          'coords'])
                frame = pd.DataFrame(temp, columns=index, index=relnames)
                frame.iloc[:, 0:2] = data.iloc[:,
                                               2*i:2*i+2].values.astype(float)
                dataFrame = pd.concat([dataFrame, frame], axis=1)
            dataFrame.replace('', np.nan, inplace=True)
            dataFrame.replace(' NaN', np.nan, inplace=True)
            dataFrame.replace(' NaN ', np.nan, inplace=True)
            dataFrame.replace('NaN ', np.nan, inplace=True)
            dataFrame.apply(pd.to_numeric)
            dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
            dataFrame.to_csv(csv_save_path, na_rep='NaN')
            print("Saved h5 (DLC) and human-readable (CSV) data versions. ",
                  "If you need to update values in the CSV, run update_h5")


class PerCamNetworkTrial(XrommToolsTrial):
    '''J.D. separate network workflow'''

    def xma_to_dlc(self,
                   cam1_dlc_project_path: str,
                   cam2_dlc_project_path: str,
                   dataset_name: str,
                   experimenter: str):
        '''Formats XMAlab pointsfile for DLC, extracts corresponding frames'''
        for i, dlc_project_path in enumerate([cam1_dlc_project_path,
                                              cam2_dlc_project_path]):
            print(f"Extracting camera {i} trial images",
                  "and 2D points from", self.trial_name, "...")
            relnames = []
            new_path = os.path.join(dlc_project_path,
                                    "labeled-data",
                                    f"{dataset_name}_cam{i}")
            try:
                contents = os.listdir(new_path)
            except FileNotFoundError:
                os.makedirs(new_path)
            if contents:
                raise FileExistsError(f'Camera {i} dataset folder exists')

            h5_save_path = os.path.join(new_path,
                                        f"/CollectedData_{experimenter}.h5")
            csv_save_path = os.path.join(new_path,
                                         f"/CollectedData_{experimenter}.csv")

            # extract 2D points data
            xpos = self.csv.iloc[:, 0 + (i-1)*2::4]
            ypos = self.csv.iloc[:, 1 + (i-1)*2::4]
            data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)

            for i, video_path in enumerate([self.cam1_path,
                                            self.cam2_path]):
                # if video file is actually folder of frames
                if os.path.isdir(video_path):
                    imgs = os.listdir(video_path)
                    abspath = os.path.join("labeled-data",
                                           dataset_name + f"_cam{i}")

                    for count, img_name in enumerate(imgs):
                        raise NotImplementedError('Check against CSV here')
                        frame_num = (count + 1).zfill(4)
                        image = cv2.imread(os.path.join(video_path, img_name))
                        relname = os.path.join(abspath,
                                               self.trial_name
                                               + f"_{frame_num}.png")
                        relnames = relnames.append(relname)
                        cv2.imwrite(os.path.join(new_path, relname), image)

                # otherwise, extract frames from video and convert to png
                else:
                    abspath = os.path.join("labeled-data", dataset_name)
                    cap = cv2.VideoCapture(video_path)
                    success, image = cap.read()
                    count = 1
                    while success:
                        frame_num = count.zfill(4)
                        relname = os.path.join(abspath,
                                               self.trial_name
                                               + f"_{frame_num}.png")
                        relnames.append(relname)
                        cv2.imwrite(os.path.join(new_path, relname), image)
                        success, image = cap.read()
                        count += 1
                    cap.release()

            # Part 3: Complete final structure of datafiles
            dataFrame = pd.DataFrame()
            temp = np.empty((data.shape[0], 2,))
            temp[:] = np.nan
            for i, bodypart in enumerate(self.bodyparts):
                index = pd.MultiIndex.from_product([[experimenter],
                                                    [bodypart],
                                                    ['x', 'y']],
                                                   names=['scorer',
                                                          'bodyparts',
                                                          'coords'])
                frame = pd.DataFrame(temp, columns=index, index=relnames)
                frame.iloc[:, 0:2] = data.iloc[:,
                                               2*i:2*i+2].values.astype(float)
                dataFrame = pd.concat([dataFrame, frame], axis=1)
            dataFrame.replace('', np.nan, inplace=True)
            dataFrame.replace(' NaN', np.nan, inplace=True)
            dataFrame.replace(' NaN ', np.nan, inplace=True)
            dataFrame.replace('NaN ', np.nan, inplace=True)
            dataFrame.apply(pd.to_numeric)
            dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
            dataFrame.to_csv(csv_save_path, na_rep='NaN')
            print("Saved h5 (DLC) and human-readable (CSV) data versions. ",
                  "If you need to update values in the CSV, run update_h5")


class RGBNetworkTrial(XrommToolsTrial):
    '''Credit to Dr. L Fahn-Lai for original design'''

    @property
    def rgb_path(self):
        return self._rgb_path

    # Change default strategy to whichever one performs the best
    def __init__(self,
                 trial_path: str,
                 codec='avc1',
                 strategy=RGBStrategy.EMPTY.value):
        super().__init__(trial_path)
        self._rgb_path = os.path.join(self.trial_path,
                                      self.trial_name + "_rgb.avi")
        self.merge_rgb(codec, strategy)

    def merge_rgb(self,
                  codec,
                  strategy):
        '''Creates a merged RGB video from cam1/2 videos'''
        print('Creating merged video for trial', self.trial_name)
        if os.path.exists(os.path.join(self.rgb_path)):
            raise FileExistsError('Merged RGB video already exists for',
                                  self.trial_name)
        try:
            cam1_video = cv2.VideoCapture(self.cam1_path)
        except FileNotFoundError:
            print('Couldn\'t find cam 1 video file for', self.trial_name)
            raise
        try:
            cam2_video = cv2.VideoCapture(self.cam2_path)
        except FileNotFoundError:
            print('Couldn\'t find cam 2 video file for', self.trial_name)
            raise

        frame_width = int(cam1_video.get(3))
        frame_height = int(cam1_video.get(4))
        frame_rate = round(cam1_video.get(5), 2)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(self.rgb_path,
                              fourcc,
                              frame_rate,
                              (frame_width, frame_height))
        i = 1
        while cam1_video.isOpened():
            print(f'Current Frame: {i}')
            ret_cam1, frame_cam1 = cam1_video.read()
            _, frame_cam2 = cam2_video.read()
            if ret_cam1:
                frame_cam1 = cv2.cvtColor(frame_cam1,
                                          cv2.COLOR_BGR2BGRA,
                                          4).astype(np.float32)
                frame_cam2 = cv2.cvtColor(frame_cam2,
                                          cv2.COLOR_BGR2BGRA,
                                          4).astype(np.float32)
                frame_cam1 = cv2.normalize(frame_cam1, None, 0, 255,
                                           norm_type=cv2.NORM_MINMAX)
                frame_cam2 = cv2.normalize(frame_cam2, None, 0, 255,
                                           norm_type=cv2.NORM_MINMAX)
                if strategy == "difference":
                    frame_blue = blend_modes.difference(frame_cam1,
                                                        frame_cam2,
                                                        1)
                elif strategy == "multiply":
                    frame_blue = blend_modes.multiply(frame_cam1,
                                                      frame_cam2,
                                                      1)
                else:
                    frame_blue = np.zeros((frame_width,
                                           frame_height,
                                           3),
                                          np.uint8)
                    frame_blue = cv2.cvtColor(frame_blue,
                                              cv2.COLOR_BGR2BGRA,
                                              4).astype(np.float32)
                frame_cam1 = cv2.cvtColor(frame_cam1,
                                          cv2.COLOR_BGRA2BGR).astype(np.uint8)
                frame_cam2 = cv2.cvtColor(frame_cam2,
                                          cv2.COLOR_BGRA2BGR).astype(np.uint8)
                frame_blue = cv2.cvtColor(frame_blue,
                                          cv2.COLOR_BGRA2BGR).astype(np.uint8)
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2GRAY)
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2GRAY)
                frame_blue = cv2.cvtColor(frame_blue, cv2.COLOR_BGR2GRAY)
                merged = cv2.merge((frame_blue, frame_cam2, frame_cam1))
                out.write(merged)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

            i = i + 1
        cam1_video.release()
        cam2_video.release()
        out.release()
        cv2.destroyAllWindows()
        print("Merged RGB video created at", self.rgb_path)

    def xma_to_dlc(self,
                   dlc_project_path: str,
                   dataset_name: str,
                   compression: str,
                   swapped: bool,
                   crossed: bool):
        '''Convert XMAlab data to DeepLabCut format'''
        new_path = os.path.join(dlc_project_path,
                                'labeled-data',
                                dataset_name)
        try:
            os.listdir(new_path)
        except FileNotFoundError:
            os.makedirs(new_path)
        # Extract frames from video to labeled-data
        frame_index = 1
        frame_num = frame_index.zfill(4)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        cap = cv2.VideoCapture(self.rgb_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            if frame_index not in self.csv:
                raise NotImplementedError('Process CSV here')
                frame_index += 1
                continue

            print(f'Extracting frame {frame_index}')
            png_name = self.trial_name + f'_{frame_num}.png'
            png_path = os.path.join(new_path, png_name)
            cv2.imwrite(png_path,
                        frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, compression])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_index += 1

            cap.release()
            cv2.destroyAllWindows()

        raise NotImplementedError('Should be part of init/bodypart retrieval')
        if swapped:
            # Need to load h5 before this
            print("Creating cam1Y-cam2Y-swapped synthetic markers")
            swaps = []
            df_sw = pd.DataFrame()
            for marker in self.bodyparts:
                name_x1 = marker+'_cam1_X'
                name_x2 = marker+'_cam2_X'
                name_y1 = marker+'_cam1_Y'
                name_y2 = marker+'_cam2_Y'
                swap_name_x1 = 'sw_'+name_x1
                swap_name_x2 = 'sw_'+name_x2
                swap_name_y1 = 'sw_'+name_y1
                swap_name_y2 = 'sw_'+name_y2
                df_sw[swap_name_x1] = self.csv[name_x1]
                df_sw[swap_name_y1] = self.csv[name_y2]
                df_sw[swap_name_x2] = self.csv[name_x2]
                df_sw[swap_name_y2] = self.csv[name_y1]
                swaps.extend([swap_name_x1,
                              swap_name_y1,
                              swap_name_x2,
                              swap_name_y2])
            df = self.csv.join(df_sw)
            print(swaps)
        if crossed:
            print("Creating cam1-cam2-crossed synthetic markers")
            crosses = []
            df_cx = pd.DataFrame()
            for marker in self.bodyparts:
                name_x1 = marker+'_cam1_X'
                name_x2 = marker+'_cam2_X'
                name_y1 = marker+'_cam1_Y'
                name_y2 = marker+'_cam2_Y'
                cross_name_x = 'cx_'+marker+'_cam1x2_X'
                cross_name_y = 'cx_'+marker+'_cam1x2_Y'
                df_cx[cross_name_x] = self.csv[name_x1]*df[name_x2]
                df_cx[cross_name_y] = df[name_y1]*df[name_y2]
                crosses.extend([cross_name_x, cross_name_y])
            df = df.join(df_cx)
            print(crosses)
        names_final = df.columns.values
        parts_final = [name.rsplit('_', 1)[0] for name in names_final]
        parts_unique_final = []
        for part in parts_final:
            if part not in parts_unique_final:
                parts_unique_final.append(part)
        print("Importing markers: ")
        print(parts_unique_final)
        with open(dlc_config_path, 'r') as dlc_config:
            yaml = YAML()
            dlc_proj = yaml.load(dlc_config)

        dlc_proj['bodyparts'] = parts_unique_final

        with open(project['path_config_file'], 'w') as dlc_config:
            yaml.dump(dlc_proj, dlc_config)

        df = df.dropna(how='all')
        unique_frames_set = {}
        unique_frames_set = {index for index in range(1, project['nframes'] + 1) if index not in unique_frames_set}
        unique_frames = sorted(unique_frames_set)
        print("Importing frames: ")
        print(unique_frames)
        df['frame_index']=[substitute_data_abspath + f'/{trial_name}_rgb_'+str(index).zfill(4)+'.png' for index in unique_frames]
        df['scorer']=project['experimenter']
        df = df.melt(id_vars=['frame_index','scorer'])
        new = df['variable'].str.rsplit("_",n=1,expand=True)
        df['variable'],df['coords'] = new[0], new[1]
        df=df.rename(columns={'variable':'bodyparts'})
        df['coords']=df['coords'].str.rstrip(" ").str.lower()
        cat_type = pd.api.types.CategoricalDtype(categories=parts_unique_final,ordered=True)
        df['bodyparts']=df['bodyparts'].str.lstrip(" ").astype(cat_type)
        newdf = df.pivot_table(columns=['scorer', 'bodyparts', 'coords'],index='frame_index',values='value',aggfunc='first',dropna=False)
        newdf.index.name=None
        if not os.path.exists(substitute_data_abspath):
            os.makedirs(substitute_data_abspath)
        if outlier_mode:
            data_name = os.path.join(substitute_data_abspath,"MachineLabelsRefine.h5")
            tracked_hdf = os.path.join(substitute_data_abspath,("MachineLabelsRefine_"+".h5"))
        else:
            data_name = os.path.join(substitute_data_abspath,("CollectedData_"+project['experimenter']+".h5"))
            tracked_hdf = os.path.join(substitute_data_abspath,("CollectedData_"+project['experimenter']+".h5"))
        newdf.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
        newdf.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w')
        tracked_csv = data_name.split('.h5')[0]+'.csv'
        newdf.to_csv(tracked_csv, na_rep='NaN')
        print("Successfully spliced XMALab 2D points to DLC format", "saved "+str(data_name), "saved "+str(tracked_hdf), "saved "+str(tracked_csv), sep='\n')


class NovelTrial():
    pass
