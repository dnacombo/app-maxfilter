#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil
import pandas as pd


def convert_parameters_to_None(config):
    # Convert all "" to None when the App runs on BL
    tmp = dict((k, None) for k, v in config.items() if v == "")
    config.update(tmp)

    return config


def read_optional_files(config, out_dir_name):
    # Get the optional files

    ## Read the optional files ##

    # From meg/fif datatype # 

    # Read the crosstalk file 
    if 'crosstalk' in config.keys():
        cross_talk_file = config.pop('crosstalk')
        if cross_talk_file is not None:
            if os.path.exists(cross_talk_file) is False:
                cross_talk_file = None
            else: 
                shutil.copy2(cross_talk_file, os.path.join(out_dir_name, 'crosstalk_meg.fif'))  # required to run a pipeline on BL
    else:
    	cross_talk_file = None

    # Read the calibration file
    if 'calibration' in config.keys():
        calibration_file = config.pop('calibration')
        if calibration_file is not None:
            if os.path.exists(calibration_file) is False:
                calibration_file = None
            else:
                shutil.copy2(calibration_file, os.path.join(out_dir_name, 'calibration_meg.dat'))  # required to run a pipeline on BL
    else:
    	calibration_file = None
    
    # Read the events file
    # We don't copy this file in outdir yet because this file can be given in fif-override and we take this one by default
    if 'events' in config.keys():
        events_file = config.pop('events')
        if events_file is not None:
            if os.path.exists(events_file) is False:
                events_file = None
    else:
    	events_file = None

    # Read head pos file
    # We don't copy this file in outdir because this file can be given in fif-override and we take this one by default
    if 'headshape' in config.keys():
        head_pos_file = config.pop('headshape')
        if head_pos_file is not None:
            if os.path.exists(head_pos_file) is False:
                head_pos_file = None
    else:
    	head_pos_file = None

    # Read channels file
    # We don't copy this file in outdir because this file can be given in fif-override and we take this one by default
    if 'channels' in config.keys():
        channels_file = config.pop('channels')
        if channels_file is not None: 
            if os.path.exists(channels_file) is False:
            	channels_file = None  
    else:
    	channels_file = None 

    # Read destination file
    # We don't copy this file in outdir because this file can be given in fif-override and we take this one by default
    if 'destination' in config.keys():
        destination = config.pop('destination')
        if destination is not None:
            if os.path.exists(destination) is False:
                destination = None
    else:
    	destination = None

    # From meg/fif-override datatype #

    # Read the destination file
    if 'destination_override' in config.keys():
        destination_override = config.pop('destination_override')
        # No need to test if destination_override is None, this key is only present when the app runs on BL
        if os.path.exists(destination_override) is False:
            if destination is not None:
                shutil.copy2(destination, os.path.join(out_dir_name, 'destination.fif'))  # required to run a pipeline on BL
        else:
            shutil.copy2(destination_override, os.path.join(out_dir_name, 'destination.fif'))  # required to run a pipeline on BL 
            destination = destination_override

    # Read head pos file
    if 'headshape_override' in config.keys():
        head_pos_file_override = config.pop('headshape_override')
        # No need to test if headshape_override is None, this key is only present when the app runs on BL
        if os.path.exists(head_pos_file_override) is False:
            if head_pos_file is not None:
                shutil.copy2(head_pos_file, os.path.join(out_dir_name, 'headshape.pos'))  # required to run a pipeline on BL 
        else:
            head_pos_file = mne.chpi.read_head_pos(head_pos_file_override)
            shutil.copy2(head_pos_file, os.path.join(out_dir_name, 'headshape.pos'))  # required to run a pipeline on BL 

    # Read channels file
    if 'channels_override' in config.keys():
        channels_file_override = config.pop('channels_override')
        # No need to test if channels_override is None, this key is only present when the app runs on BL    
        if os.path.exists(channels_file_override) is False:
            if channels_file is not None:
                shutil.copy2(channels_file, os.path.join(out_dir_name, 'channels.tsv'))
        else:
            shutil.copy2(channels_file_override, os.path.join(out_dir_name, 'channels.tsv'))
            channels_file = channels_file_override
        
    # Read the events file
    if "events_override" in config.keys():
        events_file_override = config.pop('events_override')
        # No need to test if events_override is None, this key is only present when the app runs on BL
        if os.path.exists(events_file_override) is False:
            if events_file is not None:
                shutil.copy2(events_file, os.path.join(out_dir_name, 'events.tsv'))
        else:
            shutil.copy2(events_file_override, os.path.join(out_dir_name, 'events.tsv'))  # required to run a pipeline on BL
            events_file = events_file_override
    
    return config, cross_talk_file, calibration_file, events_file, head_pos_file, channels_file, destination


def update_data_info_bads(data, channels_file):  

    df_channels = pd.read_csv(channels_file, sep='\t')
    # Select bad channels' name
    bad_channels = df_channels[df_channels["status"] == "bad"]['name']
    bad_channels = list(bad_channels.values)
    # Put channels.tsv bad channels in data.info['bads']
    data.info['bads'].sort() 
    bad_channels.sort()
    # Warning message
    if data.info['bads'] != bad_channels:
        user_warning_message_channels = f'Bad channels from the info of your MEG file are different from ' \
                                        f'those in the channels.tsv file. By default, only bad channels from channels.tsv ' \
                                        f'are considered as bad: the info of your MEG file is updated with those channels.'
        data.info['bads'] = bad_channels 
    else: 
    	user_warning_message_channels = None

    return data, user_warning_message_channels


def message_optional_files_in_reports(calibration_file, cross_talk_file, head_pos_file, destination):

    # Calibration file
    if calibration_file is None:
        report_calibration_file = 'No calibration file provided'
    else:
        report_calibration_file = 'Calibration file provided'   

    # Cross talk file
    if cross_talk_file is None:
        report_cross_talk_file = 'No cross-talk file provided'
    else:
        report_cross_talk_file = 'Cross-talk file provided'  

    # Head pos file
    if head_pos_file is None:
        report_head_pos_file = 'No headshape file provided'
    else:
        report_head_pos_file = 'Headshape file provided'

    # Destination
    if destination is None:
        report_destination_file = 'No destination file provided'
    else:
        report_destination_file = 'Destination file provided'

    return report_calibration_file, report_cross_talk_file, report_head_pos_file, report_destination_file 


def define_kwargs(config):

    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 

    return config






 #    # Read the destination file
 #    # We suppose that this file is obtained only with the BL App 
 #    if 'destination_override' in config.keys():
 #        destination = config.pop('destination_override')
 #        if destination is None or os.path.exists(destination) is False:
 #            # Use the destination parameter if it's not None
 #            if config['param_destination'] is not None:
 #                destination = config['param_destination']
 #                report_param_destination = destination
 #                # Convert destination parameter into array when the app is run on BL
 #                if isinstance(destination, str):
 #                    destination = list(map(float, destination.split(', ')))
 #                    destination = np.array(destination)
 #            else:
 #                destination = None
 #        else:
 #            report_destination_file = 'Destination file provided'
 #            # Raise a value error if the user provides both the destination file and the destination parameter
 #            if config['param_destination'] is not None:
 #                value_error_message = f"You can't provide both a destination file and a " \
 #                                      f"destination parameter. One of them must be None."
 #                raise ValueError(value_error_message)
 #            else:
 #                report_param_destination = None
 #    else:
 #        # Use the destination parameter if it's not None
 #        if config['param_destination'] is not None:
 #            destination = config['param_destination']
 #            report_param_destination = destination
 #            # Convert destination parameter into array when the app is run on BL
 #            if isinstance(destination, str):
 #                destination = list(map(float, destination.split(', ')))
 #                destination = np.array(destination)
 #        else:
 #            destination = None
 #            report_param_destination = destination
 #            report_destination_file = 'No destination file provided'