#!/usr/local/bin/python3

import json
import mne
import warnings
import numpy as np


def maxfilter(raw, calibration_file, cross_talk_file, head_pos_file, destination_file, param_st_duration,
              param_st_correlation, param_int_order, param_ext_order, param_coord_frame, param_regularize,
              param_ignore_ref, param_bad_condition, param_st_fixed, param_st_only, param_skip_by_annotation,
              param_mag_scale):
    """Performs Maxwell filtering using MNE Python and saves the file once filtered.

    Parameters
    ----------
    raw: instance of mne.io.Raw
        Data to be filtered.
    calibration_file: str or None
        Path to the '.dat' file with fine calibration coefficients. This file is machine/site-specific.
    cross_talk_file: str or None
        Path to the FIF file with cross-talk correction information.
    head_pos_file: array or None
        If array, movement compensation will be performed.
    destination_file: str or None
        The destination location for the head. Can be None, which will not change the head position, or a string path
        to a FIF file containing a MEG device<->head transformation.
    param_st_duration: float or None
        If not None, apply spatiotemporal SSS with specified buffer duration (in seconds).
    param_st_correlation: float
        Correlation limit between inner and outer subspaces used to reject ovwrlapping intersecting inner/outer signals
        during spatiotemporal SSS.
    param_int_order: int
        Order of internal component of spherical expansion.
    param_ext_order: int
        Order of external component of spherical expansion.
    param_coord_frame: str
        The coordinate frame that the origin is specified in, either 'meg' or 'head'.
    param_regularize: str or None
        Basis regularization type, must be “in” or None.
    param_ignore_ref: bool
        If True, do not include reference channels in compensation.
    param_bad_condition: str
        How to deal with ill-conditioned SSS matrices. Can be “error” (default), “warning”, “info”, or “ignore”.
    param_st_fixed: bool
        If True (default), do tSSS using the median head position during the param_st_duration window.
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
                                                     int_order=param_int_order, ext_order=param_ext_order,
                                                     coord_frame=param_coord_frame, regularize=param_regularize,
                                                     ignore_ref=param_ignore_ref, bad_condition=param_bad_condition,
                                                     st_fixed=param_st_fixed, st_only=param_st_only,
                                                     skip_by_annotation=param_skip_by_annotation,
                                                     mag_scale=param_mag_scale)

    # Save file
    if param_st_duration is not None:
        raw_maxfilter.save("out_dir_maxfilter/raw_tsss.fif", overwrite=True)
    else:
        raw_maxfilter.save("out_dir_maxfilter/raw_sss.fif", overwrite=True)

    return raw_maxfilter


def _compute_snr(meg_file):
    # Compute the SNR

    # select only MEG channels and exclude the bad channels
    meg_file = meg_file.pick_types(meg=True, exclude='bads')

    # create fixed length events
    array_events = mne.make_fixed_length_events(meg_file, duration=10)

    # create epochs
    epochs = mne.Epochs(meg_file, array_events)

    # mean signal amplitude on each epoch
    epochs_data = epochs.get_data()
    mean_signal_amplitude_per_epoch = epochs_data.mean(axis=(1, 2))  # mean on channels and times

    # mean across all epochs and its std error
    mean_final = mean_signal_amplitude_per_epoch.mean()
    std_error_final = np.std(mean_signal_amplitude_per_epoch, ddof=1) / np.sqrt(
        np.size(mean_signal_amplitude_per_epoch))

    # compute SNR
    snr = mean_final / std_error_final

    return snr


def _generate_report(data_file_before, raw_before_preprocessing, raw_after_preprocessing, snr_before, snr_after):
    # Generate a report

    # Instance of mne.Report
    report = mne.Report(title='Results Maxfilter', verbose=True)

    # Plot MEG signals in temporal domain
    fig_raw = raw_before_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, butterfly=False,
                                                                          show_scrollbars=False, proj=False)
    fig_raw_maxfilter = raw_after_preprocessing.pick(['meg'], exclude='bads').plot(duration=10, butterfly=False,
                                                                                   show_scrollbars=False, proj=False)
    # Plot power spectral density
    fig_raw_psd = raw_before_preprocessing.plot_psd()
    fig_raw_maxfilter_psd = raw_after_preprocessing.plot_psd()

    # Add figures to report
    # Add figures to report
    report.add_figs_to_section(fig_raw, captions='MEG signals before MaxFilter', section='Temporal domain')
    report.add_figs_to_section(fig_raw_maxfilter, captions='MEG signals after MaxFilter', section='Temporal domain')
    report.add_figs_to_section(fig_raw_psd, captions='Power spectral density before MaxFilter',
                               section='Frequency domain')
    report.add_figs_to_section(fig_raw_maxfilter_psd, captions='Power spectral density after MaxFilter',
                               section='Frequency domain')

    # Put this info in html format
    # Give some info about the file before preprocessing
    bad_channels = raw_before_preprocessing.info['bads']
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

    # Read the calibration files
    if 'crosstalk' in config.keys():
        cross_talk_file = config.pop('crosstalk')
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
    if 'headshape' in config.keys():
        head_pos_file = config.pop('headshape')
        if head_pos_file is not None:  # when App is run locally and "head_position": null in config.json
            head_pos_file = mne.chpi.read_head_pos(head_pos_file)
    else:
        head_pos_file = None

    # Check if param_st_duration is not None
    if config['param_st_duration'] == "":
        param_st_duration = None
    else:
        param_st_duration = config['param_st_duration']

    # Warning if bad channels are empty
    if not raw.info['bads']:
        user_warning_message = f'No channels are marked as bad. ' \
                               f'Make sure to check (automatically or visually) for bad channels before ' \
                               f'running MaxFilter.'
        warnings.warn(user_warning_message)
        dict_json_product['brainlife'].append({'type': 'warning', 'msg': user_warning_message})

    raw_maxfilter = maxfilter(raw, calibration_file, cross_talk_file, head_pos_file, destination_file,
                              param_st_duration, config['param_st_correlation'], config['param_int_order'],
                              config['param_ext_order'], config['param_coord_frame'], config['param_regularize'],
                              config['param_ignore_ref'], config['param_bad_condition'], config['param_st_fixed'],
                              config['param_st_only'], config['param_skip_by_annotation'], config['param_mag_scale'])

    # Success message in product.json
    dict_json_product['brainlife'].append({'type': 'success', 'msg': 'MaxFilter was applied successfully.'})

    # Compute SNR
    snr_before = _compute_snr(raw)
    snr_after = _compute_snr(raw_maxfilter)

    # Generate a report
    _generate_report(data_file, raw, raw_maxfilter, snr_before, snr_after)

    # Save the dict_json_product in a json file
    with open('product.json', 'w') as outfile:
        json.dump(dict_json_product, outfile)


if __name__ == '__main__':
    main()
