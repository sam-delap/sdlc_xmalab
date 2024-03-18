'''A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking'''
# Import packages
import math
import os
import warnings
from itertools import combinations
from subprocess import PIPE, Popen

import cv2
import deeplabcut
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ruamel.yaml import YAML

import xrommtools
from network import Network
from xma_data_processor import XMADataProcessor


def create_new_project(working_dir=os.getcwd(), experimenter='NA', mode='2D'):
    '''Create a new xrommtools project'''
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    dirs = ["trainingdata", "trials"]
    for folder in dirs:
        if not os.path.exists(f'{working_dir}/{folder}'):
            os.mkdir(f'{working_dir}/{folder}')

    # Create a fake video to pass into the deeplabcut workflow
    frame = np.zeros((480, 480, 3), np.uint8)
    out = cv2.VideoWriter(f'{working_dir}/dummy.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          15,
                          (480,480))
    out.write(frame)
    out.release()

    # Create a new project
    yaml = YAML()
    task = os.path.basename(working_dir)
    path_config_file = deeplabcut.create_new_project(task, experimenter,
                                                    [os.path.join(working_dir,
                                                                  "dummy.avi")],
                                                    working_dir + os.path.sep,
                                                     copy_videos=True)

    if isinstance(path_config_file, str):
        template = f"""
    task: {task}
    experimenter: {experimenter}
    working_dir: {working_dir}
    path_config_file: {path_config_file}
    dataset_name: MyData
    nframes: 0
    maxiters: 150000
    tracking_threshold: 0.1 # Fraction of total frames included in training sample
    tracking_mode: 2D
    swapped_markers: false
    crossed_markers: false

# Image Processing Vars
    search_area: 15
    threshold: 8
    krad: 17
    gsigma: 10
    img_wt: 3.6
    blur_wt: -2.9
    gamma: 0.1

# Autocorrect() Testing Vars

    trial_name: your_trial_here
    cam: cam1
    frame_num: 1
    marker: your_marker_here
    test_autocorrect: false # Set to true if you want to see autocorrect's output in Jupyter

# Video Similarity Analysis Vars
    cam1s_are_the_same_view: true    
        """

        tmp = yaml.load(template)

        if mode == 'per_cam':
            task_2 = f'{task}_cam2'
            path_config_file_2 = deeplabcut.create_new_project(task_2,
                                                               experimenter,
                                                               [os.path.join(working_dir,
                                                                            "dummy.avi")],
                                                               working_dir
                                                               + os.path.sep,
                                                               copy_videos=True)
            tmp['path_config_file_2'] = path_config_file_2

        with open(f"{working_dir}/project_config.yaml", 'w') as config:
            yaml.dump(tmp, config)

        try:
            os.rmdir(path_config_file[:path_config_file.find("config")] + os.path.join("labeled-data","dummy"))
        except FileNotFoundError:
            pass

        try:
            os.remove(os.path.join(path_config_file[:path_config_file.find("config")], "videos", "dummy.avi"))
        except FileNotFoundError:
            pass
    
    os.remove(f'{working_dir}/dummy.avi')


def load_project(working_dir=os.getcwd()):
    '''Load an existing project (only used internally/in testing)'''
    # Open the config
    with open(os.path.join(working_dir, "project_config.yaml"), 'r') as config_file:
        yaml = YAML()
        project = yaml.load(config_file)

    experimenter = str(project['experimenter'])
    project['experimenter'] = experimenter
    if project['dataset_name'] == 'MyData':
        warnings.warn('Default project name in use', SyntaxWarning)

    training_data_path = os.path.join(project['working_dir'], "trainingdata")
    trial = [folder for folder in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, folder)) and not folder.startswith('.')][0] 
    trial_path = os.path.join(training_data_path, trial)
    # Load trial CSV
    try:     
        trial_csv = pd.read_csv(os.path.join(trial_path, f'{trial}.csv'))
    except FileNotFoundError as e:
        print(f'Please make sure that your trainingdata 2DPoints csv file is named {trial}.csv')
        raise e

    # Give search_area a minimum of 10
    project['search_area'] = int(project['search_area'] + 0.5) if project['search_area'] >= 10 else 10

    # Drop untracked frames (all NaNs)
    trial_csv = trial_csv.dropna(how='all')

    # Make sure there aren't any partially tracked frames
    if trial_csv.isna().sum().sum() > 0:
        raise AttributeError(f'Detected {len(trial_csv) - len(trial_csv.dropna())} partially tracked frames. \
    Please ensure that all frames are completely tracked')

    # Check/set the default value for tracked frames
    if project['nframes'] <= 0:
        project['nframes'] = len(trial_csv)

    elif project['nframes'] != len(trial_csv):
        warnings.warn('Project nframes tracked does not match 2D Points file. \
        If this is intentional, ignore this message')

    data_processor = XMADataProcessor(config=project)
    # Check the current nframes against the threshold value * the number of frames in the cam1 video
    trial_path = os.path.join(training_data_path, trial)
    cam1_video_path = data_processor.find_cam_file(trial_path, "cam1")
    video = cv2.VideoCapture(cam1_video_path)

    if project['nframes'] < int(video.get(cv2.CAP_PROP_FRAME_COUNT)) * project['tracking_threshold']:
        tracking_threshold = project['tracking_threshold']
        warnings.warn(f'Project nframes is less than the recommended {tracking_threshold * 100}% of the total frames')

    # Check DLC bodyparts (marker names)
    default_bodyparts = ['bodypart1', 'bodypart2', 'bodypart3', 'objectA']

    path_to_trial = os.path.join(working_dir, 'trainingdata', trial)
    bodyparts = data_processor.get_bodyparts_from_xma(
        path_to_trial,
        mode=project['tracking_mode'])
    
    dlc_config_loader = YAML()
    with open(project['path_config_file'], 'r') as dlc_config:
        dlc_yaml = dlc_config_loader.load(dlc_config)

    if dlc_yaml['bodyparts'] == default_bodyparts:
        dlc_yaml['bodyparts'] = bodyparts
    elif dlc_yaml['bodyparts'] != bodyparts:
        raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

    with open(project['path_config_file'], 'w') as dlc_config:
        yaml.dump(dlc_yaml, dlc_config)

    # Check DLC bodyparts (marker names) for config 2 if needed
    if project['tracking_mode'] == 'per_cam':
        dlc_config_loader = YAML()
        with open(project['path_config_file_2'], 'r') as dlc_config:
            dlc_yaml = dlc_config_loader.load(dlc_config)
        # Better conditional logic could definitely be had to reduce function calls here
        if dlc_yaml['bodyparts'] == default_bodyparts:
            dlc_yaml['bodyparts'] = data_processor.get_bodyparts_from_xma(path_to_trial,
                                                                          mode=project['tracking_mode'])

        elif dlc_yaml['bodyparts'] != data_processor.get_bodyparts_from_xma(path_to_trial,
                                                                            mode=project['tracking_mode']):
            raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

        with open(project['path_config_file_2'], 'w') as dlc_config:
            yaml.dump(dlc_yaml, dlc_config)

    # Check test_autocorrect params for defaults
    if project['test_autocorrect']:
        if project['trial_name'] == 'your_trial_here':
            raise SyntaxError('Please specify a trial to test autocorrect() with')
        if project['marker'] == 'your_marker_here':
            raise SyntaxError('Please specify a marker to test autocorrect() with')

    # Update changed attributes to match in the file
    with open(os.path.join(working_dir, 'project_config.yaml'), 'w') as file:
        yaml.dump(project, file)

    return project

def train_network(working_dir=os.getcwd()):
    '''Starts training the network using xrommtools-compatible data in the working directory.'''
    project = load_project(working_dir)
    network = Network(config=project)
    network.train()

def analyze_videos(working_dir=os.getcwd()):
    '''Analyze videos with a pre-existing network'''
    # Open the config
    project = load_project(working_dir)

    # Error if trials directory is empty
    new_data_path = os.path.join(working_dir, 'trials')
    trials = [folder for folder in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, folder)) and not folder.startswith('.')]
    if len(trials) <= 0:
        raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {working_dir}/trials folder')

    # Establish project vars
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)
    iteration = dlc['iteration']

    if project['tracking_mode'] == '2D':
        xrommtools.analyze_xromm_videos(project['path_config_file'], new_data_path, iteration)
    elif project['tracking_mode'] == 'per_cam':
        xrommtools.analyze_xromm_videos(path_config_file=project['path_config_file'],
                                        path_config_file_cam2=project['path_config_file_2'],
                                        path_data_to_analyze=new_data_path,
                                        iteration=iteration,
                                        nnetworks=2)
    else:
        for trial in trials:
            data_processor = XMADataProcessor(config=project)
            video_path = f'{working_dir}/trials/{trial}/{trial}_rgb.avi'
            if not os.path.exists(video_path):
                data_processor.make_rgb_video(os.path.join(working_dir, 'trials'))
            destfolder = f'{working_dir}/trials/{trial}/it{iteration}/'
            deeplabcut.analyze_videos(project['path_config_file'], video_path, destfolder=destfolder, save_as_csv=True)
            split_dlc_to_xma(project, trial)

def autocorrect_trial(working_dir=os.getcwd()): #try 0.05 also
    '''Do XMAlab-style autocorrect on the tracked beads'''
    # Open the config
    project = load_project(working_dir)

    # Error if trials directory is empty
    new_data_path = os.path.join(working_dir, 'trials')
    trials = [folder for folder in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, folder)) and not folder.startswith('.')]
    if len(trials) <= 0:
        raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {working_dir}/trials folder')

    # Establish project vars
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)

    iteration = dlc['iteration']

    # For each trial
    for trial in trials:
        # Find the appropriate pointsfile
        try:
            csv = pd.read_csv(new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-Predicted2DPoints.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find predicted 2D points file. Please check the it{iteration} folder for trial {trial}') from None
        out_name = new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-AutoCorrected2DPoints.csv'

        if project['test_autocorrect']:
            cams = [project['cam']]
        else:
            cams = ['cam1', 'cam2']

        # For each camera
        for cam in cams:
            csv = autocorrect_video(cam, trial, csv, project, new_data_path)

        # Print when autocorrect finishes
        if not project['test_autocorrect']:
            print(f'Done! Saving to {out_name}')
            csv.to_csv(out_name, index=False)

def autocorrect_video(cam, trial, csv, project, new_data_path):
    '''Run the autocorrect function on a single video within a single trial'''
    # Find the raw video
    video_path = new_data_path + '/' + trial + '/' + trial + '_' + cam + '.avi'
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f'Couldn\'t find a video at file path: {video_path}') from None

    if project['test_autocorrect']:
        video.set(1, project['frame_num'] - 1)
        ret, frame = video.read()
        if ret is False:
            raise IOError('Error reading video frame')
        path_to_trial = os.path.join(new_data_path, project['trial_name'])
        autocorrect_frame(path_to_trial, frame, project['cam'], project['frame_num'], csv, project)
        return csv

    # For each frame of video
    print(f'Total frames in video: {video.get(cv2.CAP_PROP_FRAME_COUNT)}')

    for frame_index in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Load frame
        print(f'Current Frame: {frame_index + 1}')
        ret, frame = video.read()
        if ret is False:
            raise IOError('Error reading video frame')
        csv = autocorrect_frame(f'{new_data_path}/{trial}', frame, cam, frame_index, csv, project)
    return csv

def autocorrect_frame(path_to_trial, frame, cam, frame_index, csv, project):
    '''Run the autocorrect function for a single frame (no output)'''
    # For each marker in the frame
    parts_unique = get_bodyparts_from_xma(project, path_to_trial, mode='2D')
    if project['test_autocorrect']:
        parts_unique = [project['marker']]
    else:
        parts_unique = get_bodyparts_from_xma(project, path_to_trial, mode='2D')
    for part in parts_unique:
        # Find point and offsets
        x_float = csv.loc[frame_index, part + '_' + cam + '_X']
        y_float = csv.loc[frame_index, part + '_' + cam + '_Y']
        x_start = int(x_float-project['search_area']+0.5)
        y_start = int(y_float-project['search_area']+0.5)
        x_end = int(x_float+project['search_area']+0.5)
        y_end = int(y_float+project['search_area']+0.5)

        subimage = frame[y_start:y_end, x_start:x_end]

        subimage_filtered = filter_image(subimage, project['krad'], project['gsigma'], project['img_wt'], project['blur_wt'], project['gamma'])

        subimage_float = subimage_filtered.astype(np.float32)
        radius = int(1.5 * 5 + 0.5) #5 might be too high
        sigma = radius * math.sqrt(2 * math.log(255)) - 1
        subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, 2 * radius + 1), sigma)

        subimage_diff = subimage_float-subimage_blurred
        subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

        # Median
        subimage_median = cv2.medianBlur(subimage_diff, 3)

        # LUT
        subimage_median = filter_image(subimage_median, krad=3)

        # Thresholding
        subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
        min_val, _, _, _ = cv2.minMaxLoc(subimage_median)
        thres = 0.5 * min_val + 0.5 * np.mean(subimage_median) + project['threshold'] * 0.01 * 255
        _, subimage_threshold =  cv2.threshold(subimage_median, thres, 255, cv2.THRESH_BINARY_INV)

        # Gaussian blur
        subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3,3), 1.3)

        # Find contours
        contours, _ = cv2.findContours(subimage_gaussthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x_start,y_start))
        contours_im = [contour-[x_start, y_start] for contour in contours]

        # Find closest contour
        dist = 1000
        best_index = -1
        detected_centers = {}
        for i, cnt in enumerate(contours):
            detected_center, _ = cv2.minEnclosingCircle(cnt)
            dist_tmp = math.sqrt((x_float - detected_center[0])**2 + (y_float - detected_center[1])**2)
            detected_centers[round(dist_tmp, 4)] = detected_center
            if dist_tmp < dist:
                best_index = i
                dist = dist_tmp

        if project['test_autocorrect']:
            print('Raw')
            show_crop(subimage, 15)

            print('Filtered')
            show_crop(subimage_filtered, 15)

            print(f'Blurred: {sigma}')
            show_crop(subimage_blurred, 15)

            print('Diff (Float - blurred)')
            show_crop(subimage_diff, 15)

            print('Median')
            show_crop(subimage_median, 15)

            print('Median filtered')
            show_crop(subimage_median, 15)

            print('Threshold')
            show_crop(subimage_threshold, 15)

            print('Gaussian')
            show_crop(subimage_threshold, 15)

            print('Best Contour')
            detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
            show_crop(subimage, 15, contours = [contours_im[best_index]], detected_marker = detected_center_im)

        # Save center of closest contour to CSV
        if best_index >= 0:
            detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
            csv.loc[frame_index, part + '_' + cam + '_X']  = detected_center[0]
            csv.loc[frame_index, part + '_' + cam + '_Y']  = detected_center[1]
    return csv

def filter_image(image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.10):
    '''Filter the image to make it easier to see the bead'''
    krad = krad*2+1
    # Gaussian blur
    image_blur = cv2.GaussianBlur(image, (krad, krad), gsigma)
    # Add to original
    image_blend = cv2.addWeighted(image, img_wt, image_blur, blur_wt, 0)
    lut = np.array([((i/255.0)**gamma)*255.0 for i in range(256)])
    image_gamma = image_blend.copy()
    im_type = len(image_gamma.shape)
    if im_type == 2:
        image_gamma = lut[image_gamma]
    elif im_type == 3:
        image_gamma[:,:,0] = lut[image_gamma[:,:,0]]
        image_gamma[:,:,1] = lut[image_gamma[:,:,1]]
        image_gamma[:,:,2] = lut[image_gamma[:,:,2]]
    return image_gamma

def show_crop(src, center, scale=5, contours=None, detected_marker=None):
    '''Display a visual of the marker and Python's projected center'''
    if len(src.shape) < 3:
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    image = src.copy().astype(np.uint8)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    if contours:
        overlay = image.copy()
        scaled_contours = [contour*scale for contour in contours]
        cv2.drawContours(overlay, scaled_contours, -1, (255,0,0),2)
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    cv2.drawMarker(image, (center*scale, center*scale), color = (0,255,255), markerType = cv2.MARKER_CROSS, markerSize = 10, thickness = 1)
    if detected_marker:
        cv2.drawMarker(image,
        (int(detected_marker[0]*scale),
        int(detected_marker[1]*scale)),
        color = (255,0,0),
        markerType = cv2.MARKER_CROSS,
        markerSize = 10,
        thickness = 1)
    plt.imshow(image)
    plt.show()

def get_bodyparts_from_xma(project, path_to_trial, mode):
    '''Pull the names of the XMAlab markers from the 2Dpoints file'''
    data_processor = XMADataProcessor(config=project)
    return data_processor.get_bodyparts_from_xma(path_to_trial, mode)

def split_rgb(trial_path, codec='avc1'):
    '''Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos.'''
    trial_name = os.path.basename(os.path.normpath(trial_path))
    out_name = trial_name+'_split_'

    try:
        rgb_video = cv2.VideoCapture(f'{trial_path}/{trial_name}_rgb.avi')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Couldn\'t find video at {trial_path}/{trial_name}_rgb.avi') from e
    frame_width = int(rgb_video.get(3))
    frame_height = int(rgb_video.get(4))
    frame_rate = round(rgb_video.get(5),2)
    if codec == 'uncompressed':
        pix_format = 'gray'   ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
        cam1_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
        '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'cam1.avi'], stdin=PIPE)
        cam2_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
        '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'cam2.avi'], stdin=PIPE)
        blue_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
        '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'blue.avi'], stdin=PIPE)
    else:
        if codec == 0:
            fourcc = 0
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        cam1 = cv2.VideoWriter(f'{trial_path}/{out_name}'+'cam1.avi',
                                fourcc,
                                frame_rate,(frame_width, frame_height))
        cam2 = cv2.VideoWriter(f'{trial_path}/{out_name}'+'cam2.avi',
                                fourcc,
                                frame_rate,(frame_width, frame_height))
        blue_channel = cv2.VideoWriter(f'{trial_path}/{out_name}'+'blue.avi',
                                fourcc,
                                frame_rate,(frame_width, frame_height))

    i = 1
    while rgb_video.isOpened():
        ret, frame = rgb_video.read()
        print(f'Current Frame: {i}')
        i = i + 1
        if ret:
            B, G, R = cv2.split(frame)
            if codec == 'uncompressed':
                im_r = Image.fromarray(R)
                im_g = Image.fromarray(G)
                im_b = Image.fromarray(B)
                im_r.save(cam1_split_ffmpeg.stdin, 'PNG')
                im_g.save(cam2_split_ffmpeg.stdin, 'PNG')
                im_b.save(blue_split_ffmpeg.stdin, 'PNG')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cam1.write(R)
                cam2.write(G)
                blue_channel.write(B)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    if codec == 'uncompressed':
        cam1_split_ffmpeg.stdin.close()
        cam1_split_ffmpeg.wait()
        cam2_split_ffmpeg.stdin.close()
        cam2_split_ffmpeg.wait()
        blue_split_ffmpeg.stdin.close()
        blue_split_ffmpeg.wait()
    else:
        cam1.release()
        cam2.release()
        blue_channel.release()
    rgb_video.release()
    cv2.destroyAllWindows()
    print(f"Cam1 grayscale video created at {trial_path}/{out_name}cam1.avi!")
    print(f"Cam2 grayscale video created at {trial_path}/{out_name}cam2.avi!")
    print(f"Blue channel grayscale video created at {trial_path}/{out_name}blue.avi!")

def split_dlc_to_xma(project, trial, save_hdf=True):
    '''Takes the output from RGB deeplabcut and splits it into XMAlab-readable output'''
    bodyparts_xy = []
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)
    iteration = dlc['iteration']
    trial_path = project['working_dir'] + f'/trials/{trial}'

    rgb_parts = get_bodyparts_from_xma(project, trial_path, mode='rgb')
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

def analyze_video_similarity_project(working_dir):
    '''Analyze all videos in a project and take their average similar. This is dangerous, as it will assume that all cam1/cam2 pairs match
    or don't match!'''
    project = load_project(working_dir)
    similarity_score = {}
    new_data_path = os.path.join(working_dir, 'trials')
    list_of_trials = [folder for folder in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, folder)) and not folder.startswith('.')]
    yaml = YAML()

    trial_perms = combinations(list_of_trials, 2)
    for trial1, trial2 in trial_perms:
        project['trial_1_name'] = trial1
        project['trial_2_name'] = trial2
        with open(os.path.join(working_dir, 'project_config.yaml'), 'w') as file:
            yaml.dump(project, file)
        similarity_score[(trial1, trial2)] = analyze_video_similarity_trial(working_dir)
    
    return similarity_score

def analyze_video_similarity_trial(working_dir):
    '''Analyze the average similarity between trials using image hashing'''
    project = load_project(working_dir)

    # Find videos for each trial
    trial1_cam1 = cv2.VideoCapture(os.path.join(f'{working_dir}/trials', project['trial_1_name'], project['trial_1_name'] + '_cam1.avi'))
    trial2_cam1 = cv2.VideoCapture(os.path.join(f'{working_dir}/trials', project['trial_2_name'], project['trial_2_name'] + '_cam1.avi'))
    trial1_cam2 = cv2.VideoCapture(os.path.join(f'{working_dir}/trials', project['trial_1_name'], project['trial_1_name'] + '_cam2.avi'))
    trial2_cam2 = cv2.VideoCapture(os.path.join(f'{working_dir}/trials', project['trial_2_name'], project['trial_2_name'] + '_cam2.avi'))
    
    # Compare hashes
    if project['cam1s_are_the_same_view']:
        cam1_dif, noc1 = compare_two_videos(trial1_cam1, trial2_cam1)
        cam2_dif, noc2 = compare_two_videos(trial1_cam2, trial2_cam2)
    else:
        cam1_dif, noc1 = compare_two_videos(trial1_cam1, trial2_cam2)
        cam2_dif, noc2 = compare_two_videos(trial1_cam2, trial2_cam1)

    return (cam1_dif + cam2_dif) / (noc1 + noc2)

def compare_two_videos(video1, video2):
    '''Do an image hashing between two videos'''
    video1_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    video2_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    noc = math.perm(video1_frames + video2_frames, 2)
    print(f'Video 1 frames: {video1_frames}')
    print(f'Video 2 frames: {video2_frames}')
    hash_dif = 0

    hashes1 = []
    print ('Creating hashes for video 1')
    for i in range(video1_frames):
        print(f'Current frame (video 1): {i}')
        ret, frame1 = video1.read()
        if not ret:
            print('Error reading video 1 frame')
            cv2.destroyAllWindows()
            break
        hashes1.append(imagehash.phash(Image.fromarray(frame1)))

    print('Creating hashes for video 2')
    hashes2 = []
    for j in range(video2_frames):
        print(f'Current frame (video 2): {j}')
        ret, frame2 = video2.read()
        if not ret:
            print('Error reading video 2 frame')
            cv2.destroyAllWindows()
            break
        hashes2.append(imagehash.phash(Image.fromarray(frame2)))
    
    print('Comparing hashes between videos')
    for hash1 in hashes1:
        for hash2 in hashes2:
            hash_dif = hash_dif + (hash1 - hash2)
    
    return hash_dif, noc

def get_max_dissimilarity_for_trial(trial_path, window):
    trial_name = os.path.basename(trial_path)
    video1 = cv2.VideoCapture(os.path.join(trial_path, f'{trial_name}_cam1.avi'))
    video2 = cv2.VideoCapture(os.path.join(trial_path, f'{trial_name}_cam2.avi'))

    hashes1, hashes2 = hash_trial_videos(video1, video2)
    return find_dissimilar_regions(hashes1, hashes2, window)

def hash_trial_videos(video1, video2):
    '''Do an image hashing between two videos'''
    video1_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
    video2_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Video 1 frames: {video1_frames}')
    print(f'Video 2 frames: {video2_frames}')

    hashes1 = []
    print ('Creating hashes for video 1')
    for i in range(video1_frames):
        print(f'Current frame (video 1): {i}')
        ret, frame1 = video1.read()
        if not ret:
            print('Error reading video 1 frame')
            cv2.destroyAllWindows()
            break
        hashes1.append(imagehash.phash(Image.fromarray(frame1)))

    print('Creating hashes for video 2')
    hashes2 = []
    for j in range(video2_frames):
        print(f'Current frame (video 2): {j}')
        ret, frame2 = video2.read()
        if not ret:
            print('Error reading video 2 frame')
            cv2.destroyAllWindows()
            break
        hashes2.append(imagehash.phash(Image.fromarray(frame2)))

    return hashes1, hashes2

def find_dissimilar_regions(hashes1, hashes2, window):
    '''Find the region of maximum dissimilarity given 2 lists of hashes and a sliding window (how many frames)'''
    start_frame_vid1 = 0
    start_frame_vid2 = 0
    max_hash_dif_vid1 = 0
    max_hash_dif_vid2 = 0
    hash_dif_vid1 = 0
    hash_dif_vid2 = 0

    for slider in range(0, len(hashes1) // window):
        print(f'Current start frame {slider * window}')
        hash_dif_vid1, hash_dif_vid2 = compare_hash_sets(hashes1[slider * window:(slider + 1) * window], hashes2[slider * window:(slider + 1) * window])

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

def compare_hash_sets(hashes1, hashes2):
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

def analyze_marker_similarity_project(working_dir):
    '''Analyze all videos in a project and get their average rhythmicity. This assumes that all cam1/2 pairs are either the same or different!'''
    project = load_project(working_dir)
    marker_similarity = {}
    new_data_path = os.path.join(working_dir, 'trials')
    list_of_trials = [folder for folder in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, folder)) and not folder.startswith('.')]
    yaml = YAML()

    trial_perms = combinations(list_of_trials, 2)
    for trial1, trial2 in trial_perms:
        project['trial_1_name'] = trial1
        project['trial_2_name'] = trial2
        with open(os.path.join(working_dir, 'project_config.yaml'), 'w') as file:
            yaml.dump(project, file)
        marker_similarity[(trial1, trial2)] = abs(analyze_marker_similarity_trial(working_dir))
    
    return marker_similarity

def analyze_marker_similarity_trial(working_dir):
    '''Analyze marker similarity for a pair of trials. Returns the mean difference for paired marker positions (X - X, Y - Y for each marker)'''
    project = load_project(working_dir)

    # Find CSVs for each trial
    trial1_path = os.path.join(f'{working_dir}/trials', project['trial_1_name'])
    trial2_path = os.path.join(f'{working_dir}/trials', project['trial_2_name'])
    
    # Get a list of markers that each trial have in commmon
    # Marker similarity is always in rgb mode.
    bodyparts1 = get_bodyparts_from_xma(project, trial1_path, mode='rgb')
    bodyparts2 = get_bodyparts_from_xma(project, trial2_path, mode='rgb')
    markers_in_common = [marker for marker in bodyparts1 if marker in bodyparts2]
    bodyparts_xy = [f'{marker}_X' for marker in markers_in_common] + [f'{marker}_Y' for marker in markers_in_common]
    trial1_csv = pd.read_csv(os.path.join(trial1_path, project['trial_1_name'] + '.csv'))
    trial2_csv = pd.read_csv(os.path.join(trial2_path, project['trial_2_name'] + '.csv'))

    marker_similarity = sum([(trial1_csv[marker] - trial2_csv[marker]).sum() / (len(trial1_csv[marker]) + len(trial2_csv[marker])) for marker in bodyparts_xy]) / len(bodyparts_xy)

    return marker_similarity

def train_many_projects(parent_dir):
    '''Train and analyze multiple SDLC_XMALAB projects given a parent folder'''
    for folder in os.listdir(parent_dir):
        project_path = os.path.join(parent_dir, folder)
        if os.path.isdir(project_path):
            train_network(project_path)
            analyze_videos(project_path)
