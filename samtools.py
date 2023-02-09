'''A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking'''
# Import packages
import os
import math
import warnings
from subprocess import Popen, PIPE
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import deeplabcut
from deeplabcut.utils import xrommtools
from ruamel.yaml import YAML
import blend_modes

def create_new_project(working_dir=os.getcwd(), experimenter='NA'):
    '''Create a new xrommtools project'''
    saved_dir=os.getcwd()
    try:
        os.chdir(working_dir)
    except FileNotFoundError:
        os.mkdir(working_dir)
        os.chdir(working_dir)
    dirs = ["trainingdata", "trials", "XMA_files"]
    for folder in dirs:
        try:
            os.mkdir(folder)
        except FileExistsError:
            continue

    # Create a fake video to pass into the deeplabcut workflow
    frame = np.zeros((480, 480, 3), np.uint8)
    out = cv2.VideoWriter('dummy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (480,480))
    out.write(frame)
    out.release()

    # Create a new project
    yaml = YAML()
    if '\\' in working_dir:
        task = working_dir.split("\\")[len(working_dir.split("\\")) - 1]
    else:
        task = working_dir.split('/')[len(working_dir.split('/')) - 1]

    path_config_file = deeplabcut.create_new_project(task, experimenter,
        [working_dir + "\\dummy.avi"], working_dir + "\\", copy_videos=True)

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

# Image Processing Vars
        search_area: 15
        threshold: 8
        krad: 17
        gsigma: 10
        img_wt: 3.6
        blur_wt: -2.9
        gamma: 0.1
        """

        tmp = yaml.load(template)

        with open("project_config.yaml", 'w') as config:
            yaml.dump(tmp, config)

        try:
            os.rmdir(path_config_file[:path_config_file.find("config")] + "labeled-data\\dummy")
        except FileNotFoundError:
            pass

        try:
            os.remove(path_config_file[:path_config_file.find("config")] + "\\videos\\dummy.avi")
        except FileNotFoundError:
            pass

    try:
        os.remove("dummy.avi")
    except FileNotFoundError:
        pass
    os.chdir(saved_dir)

def load_project(working_dir=os.getcwd()):
    '''Load an existing project (only used internally/in testing)'''
    # Open the config
    with open(working_dir + "\\project_config.yaml", 'r') as config_file:
        yaml = YAML()
        project = yaml.load(config_file)

    experimenter = str(project['experimenter'])
    project['experimenter'] = experimenter
    if project['dataset_name'] == 'MyData':
        warnings.warn('Default project name in use', SyntaxWarning)

    # Load trial CSV
    try:
        training_data_path = os.path.join(project['working_dir'], "trainingdata")
        trial = os.listdir(training_data_path)[0]
        trial_csv = pd.read_csv(training_data_path + '/' + trial + '/' + trial + '.csv')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Please make sure that your trainingdata 2DPoints csv file is named {trial}.csv') from e

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

    # Check the current nframes against the threshold value * the number of frames in the cam1 video
    cam1_video_path = f'{training_data_path}/{trial}/{trial}_cam1.avi'
    try:
        video = cv2.VideoCapture(cam1_video_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Please make sure that your cam 1 video file is named {trial}_cam1.avi') from None

    if project['nframes'] < int(video.get(cv2.CAP_PROP_FRAME_COUNT)) * project['tracking_threshold']:
        tracking_threshold = project['tracking_threshold']
        warnings.warn(f'Project nframes is less than the recommended {tracking_threshold * 100}% of the total frames')

    # Check DLC bodyparts (marker names)
    with open(project['path_config_file'], 'r') as dlc_config:
        default_bodyparts = ['bodypart1', 'bodypart2', 'bodypart3', 'objectA']

        dlc_config_loader = YAML()

        dlc_yaml = dlc_config_loader.load(dlc_config)
        trial_name = os.listdir(working_dir + '/trainingdata')[0]

        if dlc_yaml['bodyparts'] == default_bodyparts:
            dlc_yaml['bodyparts'] = get_bodyparts_from_xma(os.path.join(working_dir, 'trainingdata', trial_name))

        elif dlc_yaml['bodyparts'] != get_bodyparts_from_xma(os.path.join(working_dir, 'trainingdata', trial_name)):
            raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

    with open(project['path_config_file'], 'w') as dlc_config:
        yaml.dump(dlc_yaml, dlc_config)

    # Update changed attributes to match in the file
    with open(os.path.join(working_dir, 'project_config.yaml'), 'w') as file:
        yaml.dump(project, file)

    return project

def train_network(working_dir=os.getcwd()):
    '''Start training xrommtools-compatible data'''
    project = load_project(working_dir=working_dir)
    data_path = working_dir + "/trainingdata"

    try:
        xrommtools.xma_to_dlc(project['path_config_file'],
        data_path,
        project['dataset_name'],
        project['experimenter'],
        project['nframes'])
    except UnboundLocalError:
        pass
    deeplabcut.create_training_dataset(project['path_config_file'])
    deeplabcut.train_network(project['path_config_file'], maxiters=project['maxiters'])

def analyze_videos(working_dir=os.getcwd()):
    '''Analyze videos with a pre-existing network'''
    # Open the config
    project = load_project(working_dir)

    # Error if trials directory is empty
    if len(os.listdir(f'{working_dir}/trials')) <= 0:
        raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {working_dir}/trials folder')

    # Establish project vars
    yaml = YAML()
    new_data_path = working_dir + "/trials"
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)
    iteration = dlc['iteration']

    xrommtools.analyze_xromm_videos(project['path_config_file'], new_data_path, iteration)

def autocorrect_trial(working_dir=os.getcwd()): #try 0.05 also
    '''Do XMAlab-style autocorrect on the tracked beads'''
    # Open the config
    project = load_project(working_dir)

    # Error if trials directory is empty
    if len(os.listdir(f'{working_dir}/trials')) <= 0:
        raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {working_dir}/trials folder')

    # Establish project vars
    new_data_path = working_dir + "/trials"
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)

    iteration = dlc['iteration']

    # For each trial
    for trial in os.listdir(new_data_path):
        # Find the appropriate pointsfile
        try:
            csv = pd.read_csv(new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-Predicted2DPoints.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find predicted 2D points file. Please check the it{iteration} folder for trial {trial}') from None
        out_name = new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-AutoCorrected2DPoints.csv'

        # For each camera
        for cam in ['cam1','cam2']:
            csv = autocorrect_video(cam, trial, csv, project, new_data_path)

        # Print when autocorrect finishes
        print(f'Done! Saving to {out_name}')
        csv.to_csv(out_name, index=False)

def autocorrect_video(cam, trial, csv, project, new_data_path):
    '''Run the autocorrect function on a single video within a single trial'''
    # Find the raw video
    video_path = new_data_path + '/' + trial + '/' + trial + '_' + cam + '.avi'
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f'Couldn\'t find a video at file path: {video_path}') from None

    # For each frame of video
    print(f'Total frames in video: {video.get(cv2.CAP_PROP_FRAME_COUNT)}')

    for frame_index in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Load frame
        print(f'Current Frame: {frame_index + 1}')
        ret, frame = video.read()
        if ret is False:
            raise IOError('Error reading video frame')
        csv = autocorrect_frame(new_data_path, trial, frame, cam, frame_index, csv, project)
    return csv

def autocorrect_frame(new_data_path, trial, frame, cam, frame_index, csv, project):
    '''Run the autocorrect function for a single frame (no output)'''
    # For each marker in the frame
    parts_unique = get_bodyparts_from_xma(f'{new_data_path}/{trial}')
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

def get_bodyparts_from_xma(path_to_trial):
    '''Pull the names of the XMAlab markers from the 2Dpoints file'''

    csv_path = [file for file in os.listdir(path_to_trial) if file[-4:] == '.csv']
    if len(csv_path) > 1:
        raise FileExistsError('Found more than 1 CSV file for trial: ' + path_to_trial)
    if len(csv_path) <= 0:
        raise FileNotFoundError('Couldn\'t find a CSV file for trial: ' + path_to_trial)
    trial_csv = pd.read_csv(path_to_trial + '/' + csv_path[0], sep=',',header=0, dtype='float',na_values='NaN')
    names = trial_csv.columns.values
    parts = [name.rsplit('_',2)[0] for name in names]
    parts_unique = []
    for part in parts:
        if part not in parts_unique:
            parts_unique.append(part)
    return parts_unique

def jupyter_test_autocorrect(working_dir=os.getcwd(), cam='cam1', marker_name=None, frame_num=1, csv_path=None):
    '''Test the filtering parameters for autocorrect_frame() from a jupyter notebook'''
    project = load_project(working_dir)
    new_data_path = working_dir + "/trials"
    trial_name = os.listdir(new_data_path)[0]
    predicted_vid_path = new_data_path + '/' + trial_name + '/' + trial_name + '_' + cam + '.avi'
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)

    iteration = dlc['iteration']
    print(f'Analyzing video at: {predicted_vid_path}')
    # Find the raw video
    try:
        video = cv2.VideoCapture(predicted_vid_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'Please make sure that your {cam} video file is named {trial_name}_{cam}.avi') from None
    # For each frame of video
    print(f'Loading {cam} video for trial {trial_name}')
    print(f'Total frames in video: {int(video.get(cv2.CAP_PROP_FRAME_COUNT))}')
    if csv_path is None:
        csv_path = f'{new_data_path}/{trial_name}/it{iteration}/{trial_name}-Predicted2DPoints.csv'

    try:
        csv = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f'Please point to a 2DPoints csv file or put a CSV file at: {csv_path}')

    # Load frame
    video.set(1, frame_num - 1)
    ret, sample_frame = video.read()
    if ret is False:
        raise IOError('Error reading video frame')

    # Grab the name of the first marker from the CSV if the user didn't specify
    if marker_name is None:
        marker_name = csv.columns.values[0].rsplit('_',2)[0]
    x_float = csv.loc[frame_num, marker_name + '_' + cam + '_X']
    y_float = csv.loc[frame_num, marker_name + '_' + cam + '_Y']
    x_start = int(x_float-15+0.5)
    y_start = int(y_float-15+0.5)
    x_end = int(x_float+15+0.5)
    y_end = int(y_float+15+0.5)

    subimage = sample_frame[y_start:y_end, x_start:x_end]

    print('Raw')
    show_crop(subimage, 15)
    subimage_filtered = filter_image(subimage, project['krad'], project['gsigma'], project['img_wt'], project['blur_wt'], project['gamma'])
    print('Filtered')
    show_crop(subimage_filtered, 15)

    subimage_float = subimage_filtered.astype(np.float32)
    radius = int(1.5 * 5 + 0.5) #5 might be too high
    sigma = radius * math.sqrt(2 * math.log(255)) - 1
    subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, 2 * radius + 1), sigma)
    print(f'Blurred: {sigma}')
    show_crop(subimage_blurred, 15)

    subimage_diff = subimage_float-subimage_blurred
    subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
    print('Diff (Float - blurred)')
    show_crop(subimage_diff, 15)

    # Median
    subimage_median = cv2.medianBlur(subimage_diff, 3)
    print('Median')
    show_crop(subimage_median, 15)

    # LUT
    subimage_median = filter_image(subimage_median, krad=3)
    print('Median filtered')
    show_crop(subimage_median, 15)

    # Thresholding
    subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
    min_val, _, _, _ = cv2.minMaxLoc(subimage_median)
    thres = 0.5 * min_val + 0.5 * np.mean(subimage_median) + project['threshold'] * 0.01 * 255
    ret, subimage_threshold =  cv2.threshold(subimage_median, thres, 255, cv2.THRESH_BINARY_INV)
    print('Threshold')
    show_crop(subimage_threshold, 15)

    # Gaussian blur
    subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3,3), 1.3)
    print('Gaussian')
    show_crop(subimage_threshold, 15)

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

    # Display contour on raw image
    if best_index >= 0:
        detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
        detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
        show_crop(subimage, 15, contours = [contours_im[best_index]], detected_marker = detected_center_im)


def merge_rgb(trial_path, codec='avc1', mode=None):
    '''Takes the path to a trial subfolder and exports a single new video with cam1 video written to the red channel and cam2 video written to the green channel.
    The blue channel is, depending on the value passed as "mode", either the difference blend between A and B, the multiply blend, or just a black frame.'''
    trial_name = trial_path.split('/')[-1]
    try:
        cam1_video = cv2.VideoCapture(f'{trial_path}/{trial_name}_cam1.avi')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Make sure your cam1 video for trial {trial_name} is named {trial_name}_cam1.avi') from e
    try:
        cam2_video = cv2.VideoCapture(f'{trial_path}/{trial_name}_cam2.avi')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Make sure your cam2 video for trial {trial_name} is named {trial_name}_cam2.avi') from e

    frame_width = int(cam1_video.get(3))
    frame_height = int(cam1_video.get(4))
    frame_rate = round(cam1_video.get(5),2)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(f'{trial_path}/{trial_name}_rgb.avi',
                            fourcc,
                            frame_rate,(frame_width, frame_height))
    i = 1
    while cam1_video.isOpened():
        print(f'Current Frame: {i}')
        ret_cam1, frame_cam1 = cam1_video.read()
        _, frame_cam2 = cam2_video.read()
        if ret_cam1:
            frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
            frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
            frame_cam1 = cv2.normalize(frame_cam1, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            frame_cam2 = cv2.normalize(frame_cam2, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            if mode == "difference":
                extra_channel = blend_modes.difference(frame_cam1,frame_cam2,1)
            elif mode == "multiply":
                extra_channel = blend_modes.multiply(frame_cam1,frame_cam2,1)
            else:
                extra_channel = np.zeros((frame_width, frame_height,3),np.uint8)
                extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGR2BGRA,4).astype(np.float32)
            frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGRA2BGR).astype(np.uint8)
            frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGRA2BGR).astype(np.uint8)
            extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGRA2BGR).astype(np.uint8)
            frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2GRAY)
            frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2GRAY)
            extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGR2GRAY)
            merged = cv2.merge((extra_channel, frame_cam2, frame_cam1))
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
    print(f"Merged RGB video created at {trial_path}/{trial_name}_rgb.avi!")


def split_rgb(trial_path, codec='avc1'):
    '''Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos.'''
    trial_name = trial_path.split('/')[-1]
    out_name = trial_name+'_split_'
    cap = cv2.VideoCapture(f'{trial_path}/{trial_name}_rgb.avi')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_rate = round(cap.get(5),2)
    if codec == 'uncompressed':
        pix_format = 'gray'   ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
        cam1_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
        '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), out_name+'_cam1.avi'], stdin=PIPE)
        cam2_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
        '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), out_name+'_cam2.avi'], stdin=PIPE)
    else:
        if codec == 0:
            fourcc = 0
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
        out1 = cv2.VideoWriter(out_name+'cam1.mp4',
                                fourcc,
                                frame_rate,(frame_width, frame_height))
        out2 = cv2.VideoWriter(out_name+'cam2.mp4',
                                fourcc,
                                frame_rate,(frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            _, G, R = cv2.split(frame)
            if codec == 'uncompressed':
                im_r = Image.fromarray(R)
                im_g = Image.fromarray(G)
                im_r.save(cam1_split_ffmpeg.stdin, 'PNG')
                im_g.save(cam2_split_ffmpeg.stdin, 'PNG')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                out1.write(R)
                out2.write(G)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    if codec == 'uncompressed':
        cam1_split_ffmpeg.stdin.close()
        cam1_split_ffmpeg.wait()
        cam2_split_ffmpeg.stdin.close()
        cam2_split_ffmpeg.wait()
    cap.release()
    if codec != 'uncompressed':
        out1.release()
        out2.release()
    cv2.destroyAllWindows()
    print("done!")
    return [out_name+'c1.mp4', out_name+'c2.mp4']
