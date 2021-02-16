#!/usr/local/bin/python3

import json
import mne
import warnings

# Generate a json.product to display messages on Brainlife UI
dict_json_product = {'brainlife': []}

# Load inputs from config.json
with open('config.json') as config_json:
    config = json.load(config_json)

# Read the files
data_file = config.pop('fif')
raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

# Read the calibration files
if 'cross_talk_correction' in config.keys():
    cross_talk_file = config.pop('cross_talk_correction')
else:
    cross_talk_file = None

if 'calibration' in config.keys():
    calibration_file = config.pop('calibration')
else:
    calibration_file = None

# Read the run to realign all runs
if 'destination' in config.keys():
    destination_file = config.pop('destination')
else:
    destination_file = None

# Head pos file
if 'head_position' in config.keys():
    head_pos_file = config.pop('head_position')
    if head_pos_file is not None:  # when App is run locally and "head_position": null in config.json
        head_pos_file = mne.chpi.read_head_pos(head_pos_file)
else:
    head_pos_file = None

# Warning if bad channels are empty
if not raw.info['bads']:
    UserWarning_message = f'No channels are marked as bad. ' \
                          f'Make sure to check (automatically or visually) for bad channels before ' \
                          f'running MaxFilter.'
    warnings.warn(UserWarning_message)
    dict_json_product['brainlife'].append({'type': 'warning', 'msg': UserWarning_message})

# Check if MaxFilter was already applied on the data
if raw.info['proc_history']:
    sss_info = raw.info['proc_history'][0]['max_info']['sss_info']
    tsss_info = raw.info['proc_history'][0]['max_info']['max_st']
    if bool(sss_info) or bool(tsss_info) is True:
        ValueError_message = f'You cannot apply MaxFilter if data have already ' \
                             f'processed with Maxwell filtering.'
        # Raise exception
        raise ValueError(ValueError_message)

# Apply MaxFilter
raw_maxfilter = mne.preprocessing.maxwell_filter(raw, calibration=calibration_file, cross_talk=cross_talk_file,
                                                 head_pos=head_pos_file, destination=destination_file,
                                                 st_duration=config['param_st_duration'],
                                                 st_correlation=config['param_st_correlation'])

# Save file
if config['param_st_duration'] is not None:
    raw_maxfilter.save("out_dir_maxfilter/test-raw_tsss.fif", overwrite=True)
else:
    raw_maxfilter.save("out_dir_maxfilter/test-raw_sss.fif", overwrite=True)

# Generate a report
report = mne.Report(title='Results Maxfilter', verbose=True)

# Plot MEG signals in temporal domain
fig_raw = raw.pick(['meg']).plot(duration=10, butterfly=False, show_scrollbars=False)
fig_raw_maxfilter = raw_maxfilter.pick(['meg']).plot(duration=10, butterfly=False, show_scrollbars=False)

# Plot power spectral density
fig_raw_psd = raw.plot_psd()
fig_raw_maxfilter_psd = raw_maxfilter.plot_psd()

# Add figures to report
report.add_figs_to_section(fig_raw, captions='MEG signals before MaxFilter')
report.add_figs_to_section(fig_raw_maxfilter, captions='MEG signals after MaxFilter')
report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before MaxFilter')
report.add_figs_to_section(fig_raw_maxfilter_psd, captions='Power spectral density after MaxFilter')

# Save report
report.save('out_dir_report/report_maxfilter.html', overwrite=True)

# Success message in product.json
dict_json_product['brainlife'].append({'type': 'success', 'msg': 'MaxFilter was applied successfully.'})

# Save the dict_json_product in a json file
with open('product.json', 'w') as outfile:
    json.dump(dict_json_product, outfile)
