#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:22:17 2022

@author: dejan
"""
import os
import io
from datetime import datetime
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import (Button, RadioButtons, SpanSelector,
                                CheckButtons)
from skimage import draw
from distutils.log import warn
import easygui
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_to_dict(list_name):
    return dict(zip(range(len(list_name)), list_name))


DATA_TYPES_LIST = ['Arbitrary',
                   'Spectral',
                   'Intensity',
                   'SpatialX',
                   'SpatialY',
                   'SpatialZ',
                   'SpatialR',
                   'SpatialTheta',
                   'SpatialPhi',
                   'Temperature',
                   'Pressure',
                   'Time',
                   'Derived',
                   'Polarization',
                   'FocusTrack',
                   'RampRate',
                   'Checksum',
                   'Flags',
                   'ElapsedTime',
                   'Frequency',
                   'MpWellSpatialX',
                   'MpWellSpatialY',
                   'MpLocationIndex',
                   'MpWellReference',
                   'PAFZActual',
                   'PAFZError',
                   'PAFSignalUsed',
                   'ExposureTime',
                   'EndMarker']
DATA_TYPES = list_to_dict(DATA_TYPES_LIST)

DATA_UNITS_LIST = ['Arbitrary',
                   'RamanShift',
                   'Wavenumber',
                   'Nanometre',
                   'ElectronVolt',
                   'Micron',
                   'Counts',
                   'Electrons',
                   'Millimetres',
                   'Metres',
                   'Kelvin',
                   'Pascal',
                   'Seconds',
                   'Milliseconds',
                   'Hours',
                   'Days',
                   'Pixels',
                   'Intensity',
                   'RelativeIntensity',
                   'Degrees',
                   'Radians',
                   'Celcius',
                   'Farenheit',
                   'KelvinPerMinute',
                   'FileTime',
                   'Microseconds',
                   'EndMarker']
DATA_UNITS = list_to_dict(DATA_UNITS_LIST)

SCAN_TYPES_LIST = ['Unspecified',
                   'Static',
                   'Continuous',
                   'StepRepeat',
                   'FilterScan',
                   'FilterImage',
                   'StreamLine',
                   'StreamLineHR',
                   'Point',
                   'MultitrackDiscrete',
                   'LineFocusMapping']
SCAN_TYPES = list_to_dict(SCAN_TYPES_LIST)

MAP_TYPES = {0: 'RandomPoints',
             1: 'ColumnMajor',
             2: 'Alternating2',
             3: 'LineFocusMapping',
             4: 'InvertedRows',
             5: 'InvertedColumns',
             6: 'SurfaceProfile',
             7: 'XyLine',
            64: 'LiveTrack',
            66: 'StreamLine',
            68: 'InvertedRows',
           128: 'Slice'}
# Remember to check this 68

MEASUREMENT_TYPES_LIST = ['Unspecified',
                          'Single',
                          'Series',
                          'Map']
MEASUREMENT_TYPES = list_to_dict(MEASUREMENT_TYPES_LIST)

WDF_FLAGS = {0: 'WdfXYXY',
             1: 'WdfChecksum',
             2: 'WdfCosmicRayRemoval',
             3: 'WdfMultitrack',
             4: 'WdfSaturation',
             5: 'WdfFileBackup',
             6: 'WdfTemporary',
             7: 'WdfSlice',
             8: 'WdfPQ',
            16: 'UnknownFlag (check in WiRE?)'}

EXIF_TAGS = {
             # Renishaw's particular tags:
             65184: "FocalPlaneXYOrigins",  # tuple of floats
             65185: "FieldOfViewXY",        # tuple of floats
             65186: "px/µ ?",               # float (1.0, 5.0?)
             # Standard Exif tags:
             34665: "ExifOffset",                # Normally, (114?)
               270: "ImageDescription",          # Normally, "white-light image"
               271: "Make",                      # Normally, "Renishaw"
             41488: "FocalPlaneResolutionUnit",  # (`5` corresponds to microns)
             41486: "FocalPlaneXResolution",     # (27120.6?)
             41487: "FocalPlaneYResolution"}     # (21632.1?)

HEADER_DT = np.dtype([('block_name', '|S4'),
                      ('block_id', np.int32),
                      ('block_size', np.int64)])



def convert_time(t):
    """Convert the Windows 64bit timestamp to the human-readable format.

    Input:
    -------
        t: timestamp in W64 format (default for .wdf files)
    Output:
    -------
        string formatted to suit local settings
    """

    return datetime.fromtimestamp(t / 1e7 - 11644473600)


def hr_filesize(filesize, suffix="B"):
    """Transform the filesize into Human-readable format."""

    for unit in ["", "k", "M", "G", "T", "P", "E", "Z"]:
        if abs(filesize) < 1024.0:
            return f"{filesize:3.1f}{unit}{suffix}"
        filesize /= 1024.0
    return f"{filesize:.1f}Yi{suffix}"


def get_exif(img):
    """Recover exif data from a PIL image"""

    img_exif = dict()
    for tag, value in img._getexif().items():
        decodedTag = EXIF_TAGS.get(tag, tag)
        img_exif[decodedTag] = value
    dunit = img_exif["FocalPlaneResolutionUnit"]
    img_exif["FocalPlaneResolutionUnit"] = DATA_UNITS.get(dunit, dunit)
    return img_exif


def read_WDF(filename, verbose=False, **kwargs):
    """Read the data (and metadata) from the binary .wdf file.

    Example
    -------
    >>> da, img = read_WDF(filename)

    Input
    ------
    filename: str
        The complete (relative or absolute) path to the file
    time_coord: str
        You can set it "seconds_elapsed" to have the time coordinate
        not as datetime value but as an float value counting the seconds
        from the beggining of the measurement.

    Output
    -------
    da: xarray DataArray
        all the recorded spectra with coordinates of each recording,
        along with the selected metadata as attributes.
    img: PIL image
        Returns `None` if no image was recorded
    """

    try:
        f = open(filename, "rb")
        if verbose:
            print(f'Reading the file: \"{filename.split("/")[-1]}\"\n')
    except IOError:
        raise IOError(f"File {filename} does not exist!")
    time_coord = kwargs.pop("time_coord", None)
    filesize = os.path.getsize(filename)
    params = dict()

    def _read(f=f, dtype=np.uint32, count=1):
        """Reads bytes from binary file,
        with the most common values given as default.
        Returns the value itself if one value, or list if count > 1
        Note that you should do ".decode()"
        on strings to avoid getting strings like "b'string'"
        For further information, refer to numpy.fromfile() function
        """

        if count == 1:
            return np.fromfile(f, dtype=dtype, count=count)[0]
        else:
            return np.fromfile(f, dtype=dtype, count=count)[0:count]

    def print_block_header(name, i, verbose=verbose):
        if verbose:
            print(f"\n{' Block : '+ name + ' ':=^80s}\n"
                  f"size: {blocks['BlockSizes'][i]},"
                  f"offset: {blocks['BlockOffsets'][i]}")

    blocks = dict()
    blocks["BlockNames"] = []
    blocks["BlockSizes"] = []
    blocks["BlockOffsets"] = []
    offset = 0
    # Reading all of the block names, offsets and sizes
    while offset < filesize - 1:
        header_dt = np.dtype([('block_name', '|S4'),
                              ('block_id', np.int32),
                              ('block_size', np.int64)])
        f.seek(offset)
        blocks["BlockOffsets"].append(offset)
        block_header = np.fromfile(f, dtype=header_dt, count=1)
        offset += block_header['block_size'][0]
        blocks["BlockNames"].append(block_header['block_name'][0].decode())
        blocks["BlockSizes"].append(block_header['block_size'][0])

    name = 'WDF1'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i]+16)
#        TEST_WDF_FLAG = _read(f, np.uint64)
        params['WdfFlag'] = WDF_FLAGS[_read(f, np.uint64)]
        f.seek(60)
        params['PointsPerSpectrum'] = npoints = _read(f)
        # Number of spectra measured (nspectra):
        params['Capacity'] = nspectra = int(_read(f, np.uint64))
        # Number of spectra recorded (ncollected):
        params['Count'] = ncollected = int(_read(f, np.uint64))
        # Number of accumulations per spectrum:
        params['AccumulationCount'] = _read(f)
        # Number of elements in the y-list (>1 for image):
        params['YlistLength'] = _read(f)
        params['XlistLength'] = _read(f)  # number of elements in the x-list
        params['DataOriginCount'] = _read(f)  # number of data origin lists
        params['ApplicationName'] = _read(f, '|S24').decode()
        version = _read(f, np.uint16, count=4)
        params['ApplicationVersion'] = '.'.join(
            [str(x) for x in version[0:-1]]) +\
            ' build ' + str(version[-1])
        params['ScanType'] = SCAN_TYPES[_read(f)]
        params['MeasurementType'] = MEASUREMENT_TYPES[_read(f)]
        params['StartTime'] = convert_time(_read(f, np.uint64))
        params['EndTime'] = convert_time(_read(f, np.uint64))
        params['SpectralUnits'] = DATA_UNITS[_read(f)]
        laser_wavenumber = _read(f, '<f')
        params['LaserWaveLength'] = np.round(10e6 / laser_wavenumber, 2) \
            if laser_wavenumber else "Unspecified"
        f.seek(240)
        params['Title'] = _read(f, '|S160').decode()
# Printing params:
    if verbose:
        for key, val in params.items():
            print(f'{key:-<40s} : \t{val}')
        if nspectra != ncollected:
            print(f'\nATTENTION:\nNot all spectra were recorded\n'
                  f'Expected nspectra={nspectra},'
                  f'while ncollected={ncollected}'
                  f'\nThe {nspectra-ncollected} missing values'
                  f'will be replaced by zeros\n')

    def pad_if_unfinished(arr, count=params['Count'],
                          capacity=params['Capacity'], replace_value=np.nan):
        """This function should be defined up where the functions are supposed
        to be defined, but then you won't be able to assign the default values,
        since the dict `params` would not have been created yet."""

        if count < capacity:
            arr[count:] = replace_value
        return arr

    name = 'WMAP'
    map_params = {}
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        m_flag = _read(f)
        map_params['MapAreaType'] = MAP_TYPES[m_flag]  # _read(f)]
        _read(f)
        map_params['InitialCoordinates'] = np.round(_read(f, '<f', count=3), 2)
        map_params['StepSizes'] = np.round(_read(f, '<f', count=3), 2)
        map_params['NbSteps'] = n_x, n_y, n_z = _read(f, np.uint32, count=3)
        map_params['LineFocusSize'] = _read(f)
    if verbose:
        for key, val in map_params.items():
            print(f'{key:-<40s} : \t{val}')

    name = 'DATA'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        data_points_count = ncollected * npoints
        spectra = np.empty((nspectra, npoints))  # container
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        spectra[:ncollected] = _read(f, '<f',
                                     count=data_points_count
                                     ).reshape(ncollected, npoints)
        if verbose:
            print(f'{"The number of spectra":-<40s} : \t{ncollected}')
            print(f'{"The number of points in each spectra":-<40s} : \t'
                  f'{npoints}')

    name = 'XLST'
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        params['XlistDataType'] = DATA_TYPES[_read(f)]
        params['XlistDataUnits'] = DATA_UNITS[_read(f)]
        x_values = _read(f, '<f', count=npoints)
    if verbose:
        print(f"{'The shape of the x_values is':-<40s} : \t{x_values.shape} ")
        print(f"*These are the \"{params['XlistDataType']}"
              f"\" recordings in \"{params['XlistDataUnits']}\" units")

    name = 'YLST'  # Not sure what's this about
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        params['YlistDataType'] = DATA_TYPES[_read(f)]
        params['YlistDataUnits'] = DATA_UNITS[_read(f)]
        y_values_count = int((blocks["BlockSizes"][i]-24)/4)
        if y_values_count > 1:
            y_values = _read(f, '<f', count=y_values_count)
            print(y_values)

    name = 'WHTL'  # This is where the image is
    img = None
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        img_bytes = _read(f, count=int((blocks["BlockSizes"][i]-16)/4))
        img = Image.open(io.BytesIO(img_bytes))

    # name = 'WXDB'  # Series of images
    # gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    # if len(gen) > 0:
    #     imgs = []
    #     img_sizes = []
    #     img_psets = dict()
    #     for i in gen:
    #         print_block_header(name, i)
    #         f.seek(blocks["BlockOffsets"][i] + 16)
    #         img_offsets = _read(f, dtype='u8', count=nspectra)
    #         for nn, j in enumerate(img_offsets):
    #             f.seek(int(j+blocks["BlockOffsets"][i]))
    #             size = _read(f)
    #             img_sizes.append(size)
    #             img_type = _read(f, dtype=np.uint8)
    #             img_flag = _read(f, dtype=np.uint8)
    #             img_key = _read(f, dtype=np.uint16)
    #             img_size = _read(f)
    #             img_length = _read(f)
    #             img_psets[nn] = {"img_type": img_type,
    #                              "img_flag": img_flag,
    #                              "img_key": img_key,
    #                              "img_size": img_size,
    #                              "img_length": img_length}

    name = 'ORGN'
    origin_labels = []
    origin_set_dtypes = []
    origin_set_units = []
    # origin_values = np.empty((params['DataOriginCount'], nspectra), dtype='<d')
    gen = [i for i, x in enumerate(blocks["BlockNames"]) if x == name]
    for i in gen:
        coord_dict = dict()
        # coord_names = {3:"x", 4:"y", 5:"z", 6:"r"}
        print_block_header(name, i)
        f.seek(blocks["BlockOffsets"][i] + 16)
        nb_origin_sets = _read(f)
        # The above is the same as params['DataOriginCount']
        for set_n in range(nb_origin_sets):
            data_type_flag = _read(f).astype(np.uint16)
            # not sure why I had to convert to uint16,
            # but if I just read it as uint32, I got rubbish sometimes
            origin_set_dtypes.append(DATA_TYPES[data_type_flag])
            coord_units = DATA_UNITS[_read(f)].lower()
            origin_set_units.append(coord_units)
            label = _read(f, '|S16').decode()
            origin_labels.append(label)

            if data_type_flag == 11:  # special case for reading timestamps
                recording_time = np.array(1e-7 *
                                          _read(f, np.uint64, count=nspectra)
                                          - 11644470000,
                                          dtype='datetime64[s]')
                # I had to add 2 hours to make it compatible with da.StartTime
                # othewise it was: 11644473600
                # Edit2: It seems that 1 hour diff is more appropriate:
                # 11644466400 -> 11644470000
                if time_coord == "seconds_elapsed":
                    recording_time = recording_time -\
                                     np.datetime64(params['StartTime'])
                    print(recording_time[2])
                    recording_time = np.round(
                                         recording_time.astype("float") * 1e-6,
                                         2)
                if recording_time.ndim == 0:  # for single scan measurement
                    recording_time = np.expand_dims(recording_time, 0)
                recording_time = pad_if_unfinished(
                                    recording_time,
                                    replace_value=recording_time[ncollected-1])

                coord_dict = {**coord_dict,
                              label: ("points", recording_time,
                                      {"units": coord_units}
                                      )
                              }
            else:
                coord_values = np.array(
                                  np.round(
                                      _read(f, '<d', count=nspectra),
                                      2))
            if data_type_flag not in [0, 11, 16, 17]:
                # 0:?
                # 11:Time - a special case already dealt with above
                # 16:Checksum - never saw anything useful recorded here
                # 17:Flags - same as 16, probably not used? - check with Renishaw?
                if coord_values.ndim == 0:  # if it's a single scan measurement
                    coord_values = np.expand_dims(coord_values, 0)
                coord_dict = {**coord_dict,
                              label: ("points", coord_values,
                                      {"units": coord_units}
                                      )
                              }
        if verbose:
            print(list(zip(origin_set_dtypes,
                           origin_set_units,
                           origin_labels)))

    if verbose:
        print('\n\n\n')
        print("coordinate", blocks)
    if params["Count"] != params["Capacity"]:
        warn(f"Not all spectra was recorded. \nExpected {nspectra}, "
             f"but only {ncollected} spectra were recorded.\n"
             f"The {nspectra-ncollected} missing spectra will be filled with "
             "zeros."
             "\n\nPlease bear in mind that working with such incomplete"
             " recordings might (and probably will) lead to odd results"
             " further down the pipeline.")

    da = xr.DataArray(spectra,
                      dims=("points", "RamanShifts"),
                      coords={**coord_dict,
                              "shifts": ("RamanShifts", x_values,
                                         {"units": "1/cm"}
                                         )
                              },
                      attrs={**params,
                             **map_params,
                             # **blocks,
                             "FileSize": hr_filesize(filesize)
                             }
                      )
    if len(map_params) > 0:
        if map_params["MapAreaType"] == "Slice":
            if ("R" in da.coords) and ("Z" in da.coords):
                scan_axes = np.array([2, 0])
                _coord_choice = ["R", "R", "Z"]
            else:
                scan_axes = np.array([1, 0])
                _coord_choice = ["X", "Y", "Z"]
        else:
            scan_axes = np.array([1, 0])
            _coord_choice = ["X", "Y", "Z"]
        scan_shape = tuple(da.attrs["NbSteps"][scan_axes])
        # scan_axes = np.argwhere(da.attrs["NbSteps"]>1).squeeze()
        # _coord_choice = ["R", "R", "Z"] if 2 in scan_axes else ["X", "Y", "Z"]
        # if 1 in scan_shape:
        #     da.attrs["MeasurementType"] = "Series like"
        col_coord = _coord_choice[scan_axes[1]]
        row_coord = _coord_choice[scan_axes[0]]
        da.attrs["ScanShape"] = scan_shape
        da.attrs["ColCoord"] = col_coord
        da.attrs["RowCoord"] = row_coord
    else:  # not a map scan
        da.attrs["ScanShape"] = (spectra.shape[0], 1)
        da.attrs["ColCoord"] = ""
        da.attrs["RowCoord"] = da.attrs["MeasurementType"]

    da = da.sortby(["shifts", "Time"])
    # Oddly enough, in case of slice scans
    # it appears that spectra aren't recorded with increasing timestamps (?!)
    if len(map_params) > 0:
        if map_params["MapAreaType"] != "Slice":
            # No matter the type of map scan, we want the same order
            if ("X" in da.coords.keys()) and\
               (0 in np.argwhere(da.attrs["NbSteps"] > 1)):
                da = da.sortby("X")
            if ("Y" in da.coords.keys()) and\
               (1 in np.argwhere(da.attrs["NbSteps"] > 1)):
                da = da.sortby("Y")
            else:
                pass
        else:
            try:
                if len(da.Z) > 1:
                    da = da.sortby("Z")
            except (KeyError, AttributeError):
                pass
    da.attrs["Folder name"], da.attrs["Filename"] = os.path.split(filename)

    return da, img

def save_picture(filename, fig, ax, dpi=1200, transparent=False):
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, bbox_inches=extent,
                transparent=transparent, dpi=dpi)

def set_img_coordinates(da, ax, unit="µm",
                        rowcoord_arr=None, colcoord_arr=None):

    if rowcoord_arr is None:
        rowcoord_arr = np.unique(da[da.RowCoord].data)
    if colcoord_arr is None:
        colcoord_arr = np.unique(da[da.ColCoord].data)

    def row_coord(y, pos):
        yind = int(y)
        if yind <= len(rowcoord_arr):
            yy = f"{rowcoord_arr[yind]}{unit}"
        else:
            yy = ""
        return yy

    def col_coord(x, pos):
        xind = int(x)
        if xind < len(colcoord_arr):
            xx = f"{colcoord_arr[xind]}{unit}"
        else:
            xx = ""
        return xx

    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(col_coord))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(row_coord))
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False,
                   width=.2, labelsize=8)

def calculate_ss(func_name: str, input_spectra, xmin=None, xmax=None,
                 is_reduced=False):
    """What to calculate in vis.ShowSelected.

    Parameters:
    -----------
    func_name: str
        one of ['area', 'barycenter_x', 'max_value',
                'peak_position', 'peak_ratio']
    spectra: xarray.DataArray
        your spectra
    xmin: float
    xmax: float
    is_reduced: bool
        whether or not you want to use the reduced spectra in the calculation
        removes the straight line connecting y[xmin] and y[xmax]

    Returns:
    --------

    """

    def calc_max_value(spectra, x):
        return np.max(spectra, axis=-1).reshape(shape)

    def calc_area(spectra, x):
        if np.ptp(x) == 0:
            return np.ones(shape)
        else:
            return np.trapz(spectra, x=x).reshape(shape)

    def calc_peak_position(spectra, x):
        peak_pos = np.argmax(spectra, axis=-1).reshape(shape)
        return x[peak_pos]  # How cool is that? :)

    def calc_barycenter_x(spectra, x):
        """The simplest solution, supposes the values are equidistant on x."""
        x_ind = np.argmin(np.abs(np.cumsum(spectra, axis=-1) -
                                 np.sum(spectra, axis=-1, keepdims=True) / 2),
                          axis=-1) + 1
        return x[x_ind].reshape(shape)

    def calc_peak_ratio(spectra, x):
        return (spectra[:, 0] / spectra[:, -1]).reshape(shape)

    function_map = {"area": calc_area,
                    "barycenter_x": calc_barycenter_x,
                    "max_value": calc_max_value,
                    "peak_position": calc_peak_position,
                    "peak_ratio": calc_peak_ratio}

    x = input_spectra.shifts.data
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    # indmax = min(len(x) - 1, indmax)
    if indmax == indmin:  # if only one line
        indmax = indmin + 1
    x = x[indmin:indmax]
    shape = input_spectra.attrs["ScanShape"]
    spectra = input_spectra.data[:, indmin:indmax].copy()

    if is_reduced:  # Draw a (base)line `a*x + b`
        a = (spectra[:, -1] - spectra[:, 0]) / (xmax - xmin)
        b = spectra[:, 0] - a * xmin
        baseline = np.outer(a, x) + b[:, np.newaxis]
        spectra -= baseline  # Sustract this line from spectra
        spectra -= np.min(spectra, axis=-1, keepdims=True)  # no negative values

    return function_map.get(func_name, calc_max_value)(spectra, x)


class ShowSelected(object):
    """To be used for visual exploration of the maps.

    Select a span on the lower plot and a map of a chosen function
    will appear on the image axis.
    Click on the image to see the spectrum corresponding to that pixel
    on the bottom plot.
    Select `Draw profile` and then click and drag to draw line on the image.
    After the line is drawn, you'll see the profile of the value shown on the
    image appear on the lower plot.

    You can use your mouse to select a zone in the spectra and a map plot
    should appear in the upper part of the figure.
    On the left part of the figure you can select what kind of function
    you want to apply on the selected span.
    ['area', 'barycenter_x', 'max_value', 'peak_position', 'peak_ratio']

    Parameters:
    -----------
        da: xarray.DataArray object
            dataArray containing your spectral recordings and some metadata.
        interpolation: string
            indicates to matplotlib what interpolation to use between pixels.
            Must be one of:
            [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
             'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
             'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']
            Default is `'kaiser'`.
            Notes:
            https://ouvaton.link/GP0Jti
            https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html
        cmap: string
            matplotlib cmap to use. Default is "viridis".
        norm: matplotlib normalization
            Default is `mpl.colors.Normalize(vmin=0, vmax=1)`
            You can use for example `mpl.colors.CenteredNorm(vcenter=0.5))`
            https://matplotlib.org/stable/tutorials/colors/colormapnorms.html
        facecolor: string
            Any acceptable matplotlib color. Default is "`oldlace`".
        active_color: string
            Any acceptable matplotlib color. Default is `'lightcoral'`
        extent: string
            what extent you want matplotlib to use
        figsize: (int, int)
        kwargs: dict
            kwargs to pass on to plt.figure
    """

    def __init__(self, input_spectra, x=None, interpolation='kaiser',
                 epsilon=0.5, **kwargs):

        self.da = input_spectra.copy()
        self.interp = interpolation
        self.f_names = ['area',
                        'barycenter_x',
                        'max_value',
                        'peak_position',
                        'peak_ratio']
        self.shifts = self.da.shifts.data
        self.xmin = self.shifts.min()
        self.xmax = self.shifts.max()
        # Get some basic info on the spectra:
        self.nshifts = self.da.attrs["PointsPerSpectrum"]
        self.ny, self.nx = self.da.attrs["ScanShape"]
        self.scan_type = self.da.attrs["MeasurementType"]
        self.cmap = kwargs.pop("cmap", "cividis")
        self.facecolor = kwargs.pop("facecolor", "oldlace")
        self.active_color = kwargs.pop("active_color", "lightcoral")
        self.file_location = kwargs.pop("file_location",
                                        self.da.attrs["Folder name"])
        self.extent = kwargs.pop("extent", "full")
        self.norm = kwargs.pop("norm", mpl.colors.Normalize(vmin=0, vmax=1))
#                                       mpl.colors.CenteredNorm(vcenter=0.5))
#       #---------------------------- about labels ---------------------------#
        xlabel = self.da.attrs["ColCoord"]
        ylabel = self.da.attrs["RowCoord"]
        if (self.scan_type == "Map") and (self.da.MapAreaType != "Slice"):
            self.xlabel = f"{xlabel} [{input_spectra[xlabel].units}]"
            self.ylabel = f"{ylabel} [{input_spectra[ylabel].units}]"
        else:  # Not a map scan
            self.xlabel = xlabel
            self.ylabel = ylabel
#        #---------------------------------------------------------------------#
        # Preparing the plots:
        figsize = kwargs.pop("figsize", (14, 8))
        self.fig = plt.figure(figsize=figsize, facecolor=self.facecolor,
                              **kwargs)
        # Add all the axes:
        self.aximg = self.fig.add_axes([.21, .3, .77, .6])  # main frame
        self.axspectrum = self.fig.add_axes([.05, .075, .93, .15],
                                            facecolor=self.facecolor)
        self.axradio = self.fig.add_axes([.05, .3, .1, .6],
                                         facecolor=self.facecolor)
        self.axreduce = self.fig.add_axes([.05, .275, .1, .09],
                                          facecolor=self.facecolor)
        self.axabsscale = self.fig.add_axes([.05, .22, .1, .09],
                                            facecolor=self.facecolor)
        self.axprofile = self.fig.add_axes([.05, .9, .06, .05],
                                           facecolor=self.facecolor)
        # self.axscroll = self.fig.add_axes([.05, .02, .9, .02])
        self.axradio.axis('off')
        self.axreduce.axis('off')
        self.axabsscale.axis('off')
        # self.axprofile.axis('off')
        # self.axscroll.axis('off')
        self.first_frame = 0
#        if self.scan_type != "Single":
#            # Slider to scroll through spectra:
#            self.last_frame = len(self.da.data)-1
#            self.sframe = Slider(self.axscroll, 'S.N°',
#                                 self.first_frame, self.last_frame,
#                                 valinit=self.first_frame, valfmt='%d',
#                                 valstep=1)
#            self.sframe.on_changed(self.scroll_spectra)

        # Show the spectrum:
        self.spectrumplot, = self.axspectrum.plot(self.da.shifts.data,
                                                  self.da.data[self.first_frame])
        self.axspectrum.xaxis.set_major_formatter(
                                mpl.ticker.FuncFormatter(self._add_cm_units))
        self.titled(self.axspectrum, self.first_frame)
        self.vline = None
        # The span selector on the spectrumplot:
        self.span = SpanSelector(self.axspectrum, self.onselect, 'horizontal',
                                 useblit=True, interactive=True,
                                 props=dict(alpha=0.5,
                                            facecolor=self.active_color)
                                 )
        # Radio buttons for function selection:
        self.func = "area"  # Default function
        self.func_choice = RadioButtons(self.axradio, self.f_names,
                                        activecolor=self.active_color,
                                        radio_props={'s': [64]*len(self.f_names)})
        self.func_choice.on_clicked(self.determine_func)
        # The "reduced" button
        self.reduced_button = CheckButtons(self.axreduce, ["reduced"])
        self.reduced_button.on_clicked(self.is_reduced)
        self.reduced = self.reduced_button.get_status()[0]

        self.abs_scale_button = CheckButtons(self.axabsscale, ["abs. scale"])
        self.abs_scale_button.on_clicked(self.is_absolute_scale)
        self.absolute_scale = self.abs_scale_button.get_status()[0]

        self.draw_profile_button = Button(self.axprofile, "Draw\nProfile",
                                          color=self.facecolor)
        self.draw_profile_button.on_clicked(self.start_drawing)
        self.cc = None
        self.rr = None
        # self.func = self.func_choice.value_selected

        # Plot the initial "image":
        if self.scan_type == "Map":
            self.initial_image = calculate_ss(self.func, self.da)
            vmin, vmax = np.percentile(self.initial_image, (5, 95))
            self.imup = self.aximg.imshow(self.initial_image,
                                          interpolation=self.interp,
                                          aspect=self.nx/self.ny/1.4,
                                          cmap=self.cmap,
                                          # norm=self.norm,
                                          vmin=vmin,
                                          vmax=vmax
                                          )
            self.aximg.set_xlabel(f"{self.xlabel}")
            self.aximg.xaxis.set_label_position('top')
            self.aximg.set_ylabel(f"{self.ylabel}")
            try:
                set_img_coordinates(self.aximg, da=self.da, unit="")
            except:
                pass

            self.cbar = self.fig.colorbar(self.imup, ax=self.aximg)
            self.fig.canvas.draw_idle()

        elif self.scan_type == 'Single':
            self.aximg.axis('off')
            self.imup = self.aximg.annotate('calculation result', (.4, .8),
                                            style='italic', fontsize=14,
                                            xycoords='axes fraction',
                                            bbox={'facecolor': self.active_color,
                                            'alpha': 0.3, 'pad': 10})
        else:
            _length = np.max((self.ny, self.nx))
            self.imup, = self.aximg.plot(np.arange(_length),
                                         np.zeros(_length), '--o', alpha=.5)
#        #--------------For drawing the line----------------------
        self.drawing_enabled = False
        self.button_released = False
        if isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.axes = self.aximg
            self.epsilon = epsilon
            self.fixed_ind = 0
            self.moving_ind = None
            self.line = None
            self.move = False
            self.point = []
            self.line_coords = {0: np.full(2, np.nan),
                                1: np.full(2, np.nan)}
            self.canvas = self.axes.figure.canvas

        self.cidonclick = self.fig.canvas.mpl_connect('button_press_event',
                                                      self.onclick)
        plt.show()

    def connect_line_draw(self):
        self.fig.canvas.mpl_disconnect(self.cidonclick)
        self.draw_profile_button.color = self.active_color
        [spine.set_visible(False) for _, spine in self.axprofile.spines.items()]
        self.cidpressline = self.canvas.mpl_connect('button_press_event',
                                                    self.pressline)
        self.cidbuttonrelease = self.canvas.mpl_connect('button_release_event',
                                                        self.button_release_callback)
        self.cidmotionnotify = self.canvas.mpl_connect('motion_notify_event',
                                                       self.motion_notify_callback)

    def disconnect_line_draw(self):
        self.draw_profile_button.color = self.facecolor
        [spine.set_visible(True) for _, spine in self.axprofile.spines.items()]
        self.remove_line()
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_disconnect(self.cidbuttonrelease)
        self.canvas.mpl_disconnect(self.cidmotionnotify)
        self.canvas.mpl_disconnect(self.cidpressline)
#        #---------------------------------------------------------------------

    def draw_first_point(self, x0, y0):
        self.remove_line()
        self.point.append(self.axes.plot(x0, y0, 'sr', ms=5, alpha=0.4)[0])
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)
        self.line, = self.axes.plot(x0, y0, "-b", lw=5, alpha=.6)

    def remove_line(self):
        if isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.point = []
            self.line = None
            for ll in self.axes.get_lines():
                ll.remove()
            self.canvas.draw()

    def pressline(self, event):
        if event.inaxes == self.axes and isinstance(self.imup, mpl.image.AxesImage):
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)
            x1 = event.xdata
            y1 = event.ydata
            self.button_released = False
            self.move = True
            if len(self.point) == 2:  # If there are already two points
                assert self.line is not None, "2 points and no line?"
                xs, ys = self.line.get_data()
                d = np.sqrt((xs - x1)**2 + (ys - y1)**2)
                if min(d) > self.epsilon:  # Draw a new point
                    self.line_coords[0] = np.array((x1, y1))
                    self.fixed_ind = 0
                    self.draw_first_point(*self.line_coords[0])
                else:  # move the existing point and with it, the line
                    self.moving_ind = np.argmin(d)  # What point to move
                    self.fixed_ind = np.argmax(d)
                    self.line_coords[self.fixed_ind] =\
                        np.array((xs[self.fixed_ind], ys[self.fixed_ind]))
                    self.line_coords[self.moving_ind] =\
                        np.array((xs[self.moving_ind], ys[self.moving_ind]))
                    self.line.set_animated(True)
            else:  # No line present -> Draw the first point
                self.line_coords[0] = np.array((x1, y1))
                self.draw_first_point(*self.line_coords[0])

    def motion_notify_callback(self, event):  # (self, event):

        if (event.inaxes != self.axes)\
                or (self.button_released)\
                or (not self.move):
            return
        xmove = event.xdata
        ymove = event.ydata
        self.line.set_data([self.line_coords[self.fixed_ind][0], xmove],
                           [self.line_coords[self.fixed_ind][1], ymove])
        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):

        if (event.inaxes == self.axes) and \
                isinstance(self.imup, mpl.image.AxesImage):  # if image
            self.button_released = True
            self.move = False
            newx = event.xdata
            newy = event.ydata
            if len(self.point) == 2:  # This means we just moved en existing line
                self.line_coords[self.moving_ind] = np.array((newx, newy))
                self.line.set_animated(False)
                self.background = None
                for i, p in enumerate(self.point):
                    #  print(self.line_coords[i], type(self.line_coords[i]))
                    p.set_data(np.expand_dims(self.line_coords[i], -1))
                self.canvas.draw()
            else:  # This adds a second point and draws a new line (if dragged).
                self.line_coords[1] = np.array((newx, newy))
                # Check if we moved between clicking and releasing the button:
                if np.sqrt(np.sum((self.line_coords[1] -
                                   self.line_coords[0])**2)) > self.epsilon:
                    self.point.append(self.axes.plot(*self.line_coords[1],
                                                     "sr", ms=5, alpha=.4)[0])
                    self.canvas.draw()
                else:  # This would ammount to clicking without dragging:
                    self.onclick(event)  # Show spectrum of the pixel clicked on
                    return  # Dont go to the next line
            self.draw_pixel_values(redraw=True)
        else:
            return

    def draw_pixel_values(self, redraw=False):
        if len(self.point) == 2:  # If there are already two points
            if redraw:
                line_ends = np.round(np.array((*self.line_coords[0],
                                            *self.line_coords[1])).astype(int))

                self.cc, self.rr = draw.line(*line_ends)
                my_img = self.imup.get_array()[self.rr, self.cc]
                xs = self.da[self.da.ColCoord].data[self.cc]
                ys = self.da[self.da.RowCoord].data[self.rr*self.da.ScanShape[1]]
                # Normally, I should have `rr*self.da.ScanShape[1] + cc`
                # in the line above, but it doesn't change anything.
                line_lengths = np.sqrt((xs - xs[0])**2 + (ys - ys[0])**2)
                self.spectrumplot.set_xdata(line_lengths)
                self.axspectrum.xaxis.set_major_formatter(
                                    mpl.ticker.FuncFormatter(self._add_micron_units))
                self.spectrumplot.set_ydata(my_img)
                self.axspectrum.set_title(f"Values on the profile {line_ends}",
                                          x=0.28, y=-0.45)
                self.axspectrum.set_xlim(0, line_lengths.max())
            else:
                my_img = self.imup.get_array()[self.rr, self.cc]
                self.spectrumplot.set_ydata(my_img)
            self.axspectrum.set_ylim(my_img.min()*.99, my_img.max()*1.01)
            self.axspectrum.relim()
            self.axspectrum.figure.canvas.draw_idle()

    def highlight_pixel(self, x, y):
        [p.remove() for p in reversed(self.aximg.patches)]
        [crosshairs.remove() for crosshairs in self.aximg.get_lines()]
        pixel_cadre = mpl.patches.Rectangle((x-.5, y-.5), 1, 1,
                                            edgecolor=(1, 0.1, 0.1, 0.5),
                                            clip_on=False,
                                            facecolor=(1, 1, 1, 0))
        self.aximg.add_artist(pixel_cadre)
        self.aximg.axvline(x, c=(1, 0.1, 0.1, 0.2))
        self.aximg.axhline(y, c=(1, 0.1, 0.1, 0.2))

    def right_on_img(self):
        [p.remove() for p in reversed(self.aximg.patches)]
        [crosshairs.remove() for crosshairs in self.aximg.get_lines()]
        self.axspectrum.set_title(f"Median of {self.nx * self.ny} spectra",
                                  x=0.28, y=-0.45)
        self.spectrumplot.set_ydata(np.median(self.da.values, axis=0))
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.axspectrum.figure.canvas.draw_idle()

    def right_on_spectrum(self):
        """Calculate as if the whole spectrum were selected"""
        pass

    def onclick(self, event):
        """Left-Clicking on a pixel will show the spectrum
        corresponding to that pixel on the bottom plot. It will also highlight
        the pixel you clicked on."""
        if event.inaxes == self.aximg:
            if event.button in [1]:
                x_pos = round(event.xdata)
                y_pos = round(event.ydata)
                if isinstance(self.imup, mpl.image.AxesImage):
                    if x_pos <= self.nx and y_pos <= self.ny and x_pos * y_pos >= 0:
                        broj = round(y_pos * self.nx + x_pos)
                        # self.sframe.set_val(broj)
                        self.scroll_spectra(broj)
                        self.highlight_pixel(x_pos, y_pos)
                elif isinstance(self.imup, mpl.lines.Line2D):
                    self.scroll_spectra(x_pos)
                else:
                    pass
            else:
                self.right_on_img()

    def is_reduced(self, label):
        self.reduced = self.reduced_button.get_status()[0]
        self.draw_img()
        self.draw_pixel_values()

    def is_absolute_scale(self, label):
        self.absolute_scale = self.abs_scale_button.get_status()[0]
        self.draw_img()
        self.draw_pixel_values()

    def _add_micron_units(self, x, pos):
        return f"{x}µm"

    def _add_cm_units(self, x, pos):
        return f"{int(x)}cm-1 "

    def start_drawing(self, event):
        """Turns on and of the possiblity to draw a line on the image
        and show the corresponding profile"""
        self.draw_profile_button.hovercolor = "lightgray"
        if not self.drawing_enabled:
            self.connect_line_draw()
        else:
            self.disconnect_line_draw()
        self.drawing_enabled = not self.drawing_enabled
        self.axprofile.figure.canvas.draw()

    def determine_func(self, label):
        "Recover the function name from radio button clicked"""
        self.func = label
        self.draw_img()

    def onselect(self, xmin, xmax):
        """When you select a region of the spectra."""
        self.xmin = xmin
        self.xmax = xmax
        if self.vline:
            self.axspectrum.lines.remove(self.vline)
            self.vline = None
        self.draw_img()

    def normalize_data(self, mydata):
        return (mydata - np.min(mydata)) / np.ptp(mydata)

    def draw_img(self):
        """Draw/update the image."""
        # calculate the function:
        img = calculate_ss(self.func, self.da, self.xmin, self.xmax,
                           is_reduced=self.reduced)
        # img = self.normalize_data(img)
        if self.scan_type == "Map":
            if len(self.point) == 2:
                self.draw_pixel_values()
            limits = np.percentile(img, [5, 95])
#            limits = np.ptp(img)
            if self.absolute_scale:
                img = self.normalize_data(img)
                limits = [0, 1]
            self.imup.set_clim(limits)
            self.cbar.mappable.set_clim(*limits)
            self.imup.set_data(img)
        elif self.scan_type == 'Single':
            self.imup.set_text(f"{img[0][0]:.3G}")
        else:
            self.imup.set_ydata(img.squeeze())
            self.aximg.relim()
            self.aximg.autoscale_view(None, False, True)

        self.aximg.set_title(f"Calculated {'reduced'*self.reduced} {self.func} "
                             # f"between {self.xmin:.1f} and {self.xmax:.1f} cm-1"
                             # f" / {naj:.2f}\n"
                             )
        self.fig.canvas.draw_idle()

    def scroll_spectra(self, val):
        """Update the spectrum plot"""
        frame = val  # int(self.sframe.val)
        current_spectrum = self.da.data[frame]
        self.spectrumplot.set_xdata(self.shifts)
        self.spectrumplot.set_ydata(current_spectrum)
        self.axspectrum.set_xlim(self.shifts.min(), self.shifts.max())
        self.axspectrum.xaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(self._add_cm_units))
        self.axspectrum.set_ylim(current_spectrum.min(),
                                 current_spectrum.max()*1.05)
        self.axspectrum.relim()
        self.axspectrum.autoscale_view()
        self.titled(self.axspectrum, frame)
        self.fig.canvas.draw_idle()

    def titled(self, ax, frame):
        """Set the title for the spectrum plot"""
        if self.scan_type == "Single":
            new_title = self.da.attrs["Title"]
        elif self.scan_type == "Series":
            if "Temperature" in self.da.coords.keys():
                coordinate = f"{self.da.Temperature.data[frame]} °C"
            else:
                coordinate = f"{np.datetime64(self.da.Time.data[frame], 's')}"
            new_title = f"Spectrum @ {coordinate}"
        else:
            try:
                new_title = f"Spectrum N°{int(frame)} @ " +\
                    f"{self.da.RowCoord}: {self.da[self.da.RowCoord].data[frame]}µm"\
                  + f"; {self.da.ColCoord}: {self.da[self.da.ColCoord].data[frame]}µm"
            except (AttributeError, KeyError):
                # This needs to be made more generic:
                new_title = f"S{frame//self.nx:3d} {frame%self.nx:3d}"
        ax.set_title(new_title, x=0.28, y=-0.45)

    def set_facecolor(self, my_col):
        self.fig.set_facecolor(my_col)
        self.axspectrum.set_facecolor(my_col)
        self.axprofile.set_facecolor(my_col)
        self.axradio.set_facecolor(my_col)
#        self.axradio.update({"facecolor": my_col})
#        self.axradio.figure.canvas.draw()
        self.axreduce.set_facecolor(my_col)
        self.axabsscale.set_facecolor(my_col)
        self.draw_profile_button.color = my_col
        self.facecolor = my_col


my_filename = easygui.fileopenbox("Choose your .wdf file",
                                  default="../../../RamanData/*.wdf")
# my_filename = "/home/dejan/Documents/RamanData/frejus21/exampleA2.wdf"
da, img = read_WDF(my_filename)

vidju = ShowSelected(da)
