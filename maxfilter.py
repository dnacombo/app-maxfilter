#!/usr/local/bin/python3

import json
import mne
import warnings

# Print mne version
print(mne.__version__)

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# Read the files
data_file = str(config.pop('input_raw'))
raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

# Read the calibration files
cross_talk_file = config.pop('input_cross_talk')
if cross_talk_file is not None:
    cross_talk_file = str(cross_talk_file)

calibration_file = config.pop('input_calibration')
if calibration_file is not None:
    calibration_file = str(calibration_file)

# Head pos file
head_pos_file = config.pop('head_pos')
if head_pos_file is not None:
    head_pos_file = mne.chpi.read_head_pos(str(head_pos_file))

# Apply MaxFilter

# Warning if bad channels are empty
if raw.info['bads'] is None:
    warnings.warn('No channels are marked as bad.'
                  'Make sure to check (automatically or visually) for bad channels before running MaxFilter.')

# Check if MaxFilter was already applied on the data
sss_info = raw.info['proc_history'][0]['max_info']['sss_info']
tsss_info = raw.info['proc_history'][0]['max_info']['max_st']
if bool(sss_info) or bool(tsss_info) is True:
    raise ValueError('You cannot apply MaxFilter if data have already '
                     'processed with Maxwell-filter.')

# MaxFilter
raw_maxfilter = mne.preprocessing.maxwell_filter(raw, calibration=calibration_file, cross_talk=cross_talk_file,
                                                 head_pos=head_pos_file, **config['params_maxwell_filter'])

# Save file
type_maxfilter = config.pop('params_maxwell_filter')
if type_maxfilter['st_duration'] is None:
    raw_maxfilter.save(raw_maxfilter.filenames[0].replace('.fif', '_%s.fif' % config['output_tag_sss']),
                       **config['params_save'])
else:
    raw_maxfilter.save(raw_maxfilter.filenames[0].replace('.fif', '_%s.fif' % config['output_tag_tsss']),
                       **config['params_save'])

