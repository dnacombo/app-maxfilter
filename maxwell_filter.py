#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np
import os
import shutil

def apply_maxwell_filter(raw, calibration_file, cross_talk_file, head_pos_file, destination_file, param_st_duration,
                         param_st_correlation, param_origin, param_int_order, param_ext_order, param_coord_frame, param_regularize,
                         param_ignore_ref, param_bad_condition, param_st_fixed, param_st_only, param_skip_by_annotation,
                         param_mag_scale):
    """Perform Maxwell filtering using MNE Python and save the file once filtered.

    Parameters
    ----------
    raw: instance of mne.io.Raw
        Data to be filtered.
    calibration_file: str or None
        Path to the '.dat' file with fine calibration coefficients. This file is machine/site-specific.
    cross_talk_file: str or None
        Path to the FIF file with cross-talk correction information.
    head_pos_file: str or None
        Path to the '.pos' file containing the info to perform movement compensation.
    destination_file: str or None
        The destination location for the head. Can be None, which will not change the head position, or a string path
        to a FIF file containing a MEG device<->head transformation.
    param_st_duration: float or None
        If not None, apply spatiotemporal SSS with specified buffer duration (in seconds).
    param_st_correlation: float
        Correlation limit between inner and outer subspaces used to reject ovwrlapping intersecting inner/outer signals
        during spatiotemporal SSS.
    param_origin: str
        Origin of internal and external multipolar moment space in meters. The default is 'auto', which means 
        (0., 0., 0.)when coord_frame='meg', and a head-digitization-based origin fit using fit_sphere_to_headshape()
        when coord_frame='head'.
    param_int_order: int
        Order of internal component of spherical expansion.
    param_ext_order: int
        Order of external component of spherical expansion.
    param_coord_frame: str
        The coordinate frame that the origin is specified in, either 'meg' or 'head' (default).
    param_regularize: str or None
        Basis regularization type, must be “in” (default) or None.
    param_ignore_ref: bool
        If True, do not include reference channels in compensation.
    param_bad_condition: str
        How to deal with ill-conditioned SSS matrices. Can be “error” (default), “warning”, “info”, or “ignore”.
    param_st_fixed: bool
        If True (default), do spatiotemporal SSS using the median head position during the param_st_duration window.
    param_st_only: bool
        If True, only tSSS (temporal) projection of MEG data will be performed on the output data.
    param_skip_by_annotation: str or list of str
        If a string (or list of str), any annotation segment that begins with the given string will not be included in
        filtering, and segments on either side of the given excluded annotated segment will be filtered separately.
    param_mag_scale: float
        The magenetometer scale-factor used to bring the magnetometers to approximately the same order of magnitude as
        the gradiometers (default 100.), as they have different units (T vs T/m).

    Returns
    -------
    raw_maxfilter: instance of mne.io.Raw
        The raw data with Maxwell filtering applied.
    """

    # Check if MaxFilter was already applied on the data
    if raw.info['proc_history']:
        sss_info = raw.info['proc_history'][0]['max_info']['sss_info']
        tsss_info = raw.info['proc_history'][0]['max_info']['max_st']
        if bool(sss_info) or bool(tsss_info) is True:
            value_error_message = f'You cannot apply MaxFilter if data have been already ' \
                                  f'processed with Maxwell filtering.'
            # Raise exception
            raise ValueError(value_error_message)

    # Apply MaxFilter
    raw_maxfilter = mne.preprocessing.maxwell_filter(raw, calibration=calibration_file, cross_talk=cross_talk_file,
                                                     head_pos=head_pos_file, destination=destination_file,
                                                     st_duration=param_st_duration, st_correlation=param_st_correlation,
                                                     origin=param_origin, int_order=param_int_order, 
                                                     ext_order=param_ext_order, coord_frame=param_coord_frame, 
                                                     regularize=param_regularize, ignore_ref=param_ignore_ref, 
                                                     bad_condition=param_bad_condition, st_fixed=param_st_fixed, 
                                                     st_only=param_st_only, skip_by_annotation=param_skip_by_annotation,
                                                     mag_scale=param_mag_scale)

    # Save file
    raw_maxfilter.save("out_dir_maxwell_filter/meg.fif", overwrite=True)

    return raw_maxfilter


def _compute_snr(meg_file):
    # Compute the SNR

    # Select only MEG channels and exclude the bad channels
    meg_file = meg_file.pick_types(meg=True, exclude='bads')

    # Create fixed length events
    array_events = mne.make_fixed_length_events(meg_file, duration=10)

    # Create epochs
    epochs = mne.Epochs(meg_file, array_events)

    # Compute mean signal amplitude on each epoch
    epochs_data = epochs.get_data()
    mean_signal_amplitude_per_epoch = epochs_data.mean(axis=(1, 2))  # mean on channels and times

    # Compute mean across all epochs and its std error
    mean_final = mean_signal_amplitude_per_epoch.mean()
    std_error_final = np.std(mean_signal_amplitude_per_epoch, ddof=1) / np.sqrt(
        np.size(mean_signal_amplitude_per_epoch))

    # Compute SNR
    snr = mean_final / std_error_final

    return snr


def _generate_report(data_file_before, raw_before_preprocessing, raw_after_preprocessing, bad_channels,
                     snr_before, snr_after):
    # Generate a report

    # Create instance of mne.Report
    report = mne.Report(title='Results Maxfilter', verbose=True)

    # Plot MEG signals in temporal domain
    fig_raw = raw_before_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, scalings='auto', butterfly=False,
                                                                          show_scrollbars=False, proj=False)
    fig_raw_maxfilter = raw_after_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, scalings='auto',
                                                                                   butterfly=False,
                                                                                   show_scrollbars=False, proj=False)
    # Plot power spectral density
    fig_raw_psd = raw_before_preprocessing.plot_psd()
    fig_raw_maxfilter_psd = raw_after_preprocessing.plot_psd()

    # Add figures to report
    report.add_figs_to_section(fig_raw, captions='MEG signals before MaxFilter', section='Temporal domain')
    report.add_figs_to_section(fig_raw_maxfilter, captions='MEG signals after MaxFilter', section='Temporal domain')
    report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before MaxFilter',
                               section='Frequency domain')
    report.add_figs_to_section(fig_raw_maxfilter_psd, captions='Power spectral density after MaxFilter',
                               section='Frequency domain')

    # Give some info about the file before preprocessing
    sampling_frequency = raw_before_preprocessing.info['sfreq']
    highpass = raw_before_preprocessing.info['highpass']
    lowpass = raw_before_preprocessing.info['lowpass']

    # Put this info in html format
    # Info on data
    html_text_info = f"""<html>
    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>Input file: {data_file_before}</td>
            </tr>
            <tr>
                <td>Bad channels: {bad_channels}</td>
            </tr>
            <tr>
                <td>Sampling frequency: {sampling_frequency}Hz</td>
            </tr>
            <tr>
                <td>Highpass: {highpass}Hz</td>
            </tr>
            <tr>
                <td>Lowpass: {lowpass}Hz</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Info on SNR
    html_text_snr = f"""<html>
    <head>
        <style type="text/css">
            table {{ border-collapse: collapse;}}
            td {{ text-align: center; border: 1px solid #000000; border-style: dashed; font-size: 15px; }}
        </style>
    </head>

    <body>
        <table width="50%" height="80%" border="2px">
            <tr>
                <td>SNR before MaxFilter: {snr_before}</td>
            </tr>
            <tr>
                <td>SNR after MaxFilter: {snr_after}</td>
            </tr>
        </table>
    </body>

    </html>"""

    # Add html to reports
    report.add_htmls_to_section(html_text_info, captions='MEG recording features', section='Info', replace=False)
    report.add_htmls_to_section(html_text_snr, captions='Signal to noise ratio', section='Signal to noise ratio',
                                replace=False)

    # Save report
    report.save('out_dir_report/report_maxfilter.html', overwrite=True)


def main():

    # Generate a json.product to display messages on Brainlife UI
    dict_json_product = {'brainlife': []}

    # Load inputs from config.json
    with open('config.json') as config_json:
        config = json.load(config_json)

    # Read the files
    data_file = config.pop('fif')
    raw = mne.io.read_raw_fif(data_file, allow_maxshield=True)

    # Read the crosstalk file
    cross_talk_file = config.pop('crosstalk')
    if os.path.exists(cross_talk_file) is False:
        cross_talk_file = None
    else: 
        shutil.copy2(cross_talk_file, 'out_dir_maxwell_filter/crosstalk_meg.fif')  # required to run a pipeline on BL

    # Read the calibration file
    calibration_file = config.pop('calibration')
    if os.path.exists(calibration_file) is False:
        calibration_file = None
    else:
        shutil.copy2(calibration_file, 'out_dir_maxwell_filter/calibration_meg.dat') # required to run a pipeline on BL

    # Read the destination file
    destination_file = config.pop('destination')
    if os.path.exists(destination_file) is False:
        destination_file = None
    else:
        shutil.copy2(destination_file, 'out_dir_maxwell_filter/destination.fif') # required to run a pipeline on BL

    # Read head pos file
    head_pos = config.pop('headshape')
    if os.path.exists(head_pos) is True:
        head_pos_file = mne.chpi.read_head_pos(head_pos)
        shutil.copy2(head_pos_file, 'out_dir_maxwell_filter/headshape.pos') # required to run a pipeline on BL 
    else:
        head_pos_file = None

    # Read events file 
    events_file = config.pop('events')
    if os.path.exists(events_file) is True:
        shutil.copy2(events_file, 'out_dir_maxwell_filter/events.tsv')  # required to run a pipeline on BL

    # Check if param_st_duration is not None
    if config['param_st_duration'] == "":
        config['param_st_duration'] = None  # when App is run on Bl, no value for this parameter corresponds to ''

    # Display a warning if bad channels are empty
    if not raw.info['bads']:
        user_warning_message = f'No channels are marked as bad. ' \
                               f'Make sure to check (automatically or visually) for bad channels before ' \
                               f'running MaxFilter.'
        warnings.warn(user_warning_message)
        dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message})

    # Keep bad channels in memory before they are interpolated by MaxFilter
    bad_channels = raw.info['bads']

    # Define kwargs
    # Delete keys values in config.json when this app is executed on Brainlife
    if '_app' and '_tid' and '_inputs' and '_outputs' in config.keys():
        del config['_app'], config['_tid'], config['_inputs'], config['_outputs'] 
    kwargs = config  

    # Apply MaxFilter
    raw_maxfilter = apply_maxwell_filter(raw, calibration_file, cross_talk_file, head_pos_file, destination_file,
                              **kwargs)

    # Write a success message in product.json
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'MaxFilter was applied successfully.'})

    # Compute SNR
    # snr_before = _compute_snr(raw)
    # snr_after = _compute_snr(raw_maxfilter)

    # Generate a report
    # _generate_report(data_file, raw, raw_maxfilter, bad_channels, snr_before, snr_after)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
