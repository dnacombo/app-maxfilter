import json
import mne


# Test version
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
raw_sss = mne.preprocessing.maxwell_filter(raw, calibration=calibration_file, cross_talk=cross_talk_file,
                                           head_pos=head_pos_file, **config['params'])

# Save file
raw_sss.save(raw_sss.filenames[0].replace('.fif', '_%s.fif' % config['output_tag']), overwrite=True)
