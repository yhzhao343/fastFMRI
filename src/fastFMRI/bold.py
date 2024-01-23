import numpy as np
import numba as nb
import math
import scipy
import scipy.stats
from numba import jit, prange, f4, f8, i4, i8, b1
from itertools import combinations
from functools import reduce
from .file_helpers import load_xsv, extract_confound_regressors
from .file_helpers import extract_bold_file_info_from_name, load_file


def scrub_motion_outlier(
    bold_image,
    outlier_index,
    mask_image=None,
    inplace=False,
    numba=True,
    mask_threshold=0,
):
    if inplace:
        bold_image_out = bold_image
    else:
        bold_image_out = np.copy(bold_image)
    x, y, z, f = bold_image.shape
    frame_index = np.empty(f, dtype=np.int32)
    keep_mask = np.empty(f, dtype=np.int32)
    frame_index_without_outlier = np.empty(f, dtype=np.int32)
    o = len(outlier_index)
    if mask_image is None:
        mask_image = get_mask_from_bold(bold_image, mask_threshold)
    if numba:
        type_to_use = bold_image.dtype
        time_series = np.empty(f, dtype=type_to_use)
        time_series_without_outlier = np.empty(f - o, dtype=type_to_use)
        interp_vals = np.empty(o, dtype=type_to_use)
        bold_image_out = _scrub_motion_outlier_numba(
            bold_image,
            mask_image,
            outlier_index,
            frame_index,
            keep_mask,
            frame_index_without_outlier,
            time_series,
            time_series_without_outlier,
            interp_vals,
            bold_image_out,
        )
    else:
        bold_image_out = _scrub_motion_outlier(
            bold_image,
            mask_image,
            outlier_index,
            frame_index,
            keep_mask,
            frame_index_without_outlier,
            bold_image_out,
        )

    return bold_image_out


def _scrub_motion_outlier(
    bold_image,
    mask_image,
    outlier_index,
    frame_index,
    keep_mask,
    frame_index_without_outlier,
    bold_image_out,
):
    x_size, y_size, z_size, num_frame = bold_image.shape
    frame_index = np.arange(num_frame)
    keep_mask = np.ones(num_frame, dtype=np.int32)
    keep_mask[outlier_index] = False
    frame_index_without_outlier = frame_index[keep_mask]
    for x in prange(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if mask_image[x, y, z]:
                    time_series = bold_image[x, y, z, :]
                    time_series_without_outlier = time_series[keep_mask]
                    interp_vals = np.interp(
                        outlier_index,
                        frame_index_without_outlier,
                        time_series_without_outlier,
                    )
                    for i in range(outlier_index.size):
                        bold_image_out[x, y, z, outlier_index[i]] = interp_vals[i]
    return bold_image_out


@jit(
    [
        f4[:, :, :, :](
            f4[:, :, :, :],
            b1[:, :, :],
            i4[:],
            i4[:],
            i4[:],
            i4[:],
            f4[:],
            f4[:],
            f4[:],
            f4[:, :, :, :],
        ),
        f8[:, :, :, :](
            f8[:, :, :, :],
            b1[:, :, :],
            i4[:],
            i4[:],
            i4[:],
            i4[:],
            f8[:],
            f8[:],
            f8[:],
            f8[:, :, :, :],
        ),
    ],
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _scrub_motion_outlier_numba(
    bold_image,
    mask_image,
    outlier_index,
    frame_index,
    keep_mask,
    frame_index_without_outlier,
    time_series,
    time_series_without_outlier,
    interp_vals,
    bold_image_out,
):
    x_size, y_size, z_size, num_frame = bold_image.shape
    frame_index = np.arange(num_frame)
    keep_mask = np.ones(num_frame, dtype=np.int32)
    keep_mask[outlier_index] = False
    frame_index_without_outlier = frame_index[keep_mask]
    for x in prange(x_size):
        for y in range(y_size):
            for z in range(z_size):
                if mask_image[x, y, z]:
                    time_series = bold_image[x, y, z, :]
                    time_series_without_outlier = time_series[keep_mask]
                    interp_vals = np.interp(
                        outlier_index,
                        frame_index_without_outlier,
                        time_series_without_outlier,
                    )
                    for i in range(outlier_index.size):
                        bold_image_out[x, y, z, outlier_index[i]] = interp_vals[i]
    return bold_image_out


FWHM_PER_SIGNAL = math.sqrt(8 * math.log(2))


# @jit(f4(f4), nopython=True, fastmath=True, cache=True)
def sigma_to_fwhm(sigma):
    # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
    return sigma * FWHM_PER_SIGNAL


# @jit(f4(f4), nopython=True, fastmath=True, cache=True)
def fwhm_to_sigma(fwhm):
    # https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/
    return fwhm / FWHM_PER_SIGNAL


def gaussian_kernel(
    fwhm=3, sigma=None, truncate=3.0, half_kernel_len=None, fmri_resolution=2, mode=None
):
    if sigma is None:
        sigma = fwhm_to_sigma(fwhm)
    if half_kernel_len is None:
        half_kernel_len = int((sigma * truncate) / fmri_resolution + 0.5)
    if mode == "scipy":
        kernel = _gaussian_kernel_scipy(sigma, half_kernel_len, fmri_resolution)
    else:
        kernel = _gaussian_kernel(sigma, half_kernel_len, fmri_resolution)
    return kernel


@jit(f8[:](f8, i4, f8), parallel=True, nopython=True, fastmath=True, cache=True)
def _gaussian_kernel_scipy(
    sigma=1.2739826440811157, half_kernel_len=2, fmri_resolution=2
):
    x = np.arange(-half_kernel_len, half_kernel_len + 1) * fmri_resolution
    kernel = np.exp(-0.5 * np.square(x) / np.square(sigma))
    kernel /= kernel.sum()
    return kernel


@jit(f8[:](f8[:], f8), parallel=True, nopython=True, fastmath=True, cache=True)
def _gaussian_cdf(x, sigma):
    denom = math.sqrt(2) * sigma
    for i in prange(len(x)):
        x[i] = 0.5 * (1 - math.erf(-x[i] / denom))
    return x


@jit(f8[:](f8, i4, f8), nopython=True, fastmath=True, cache=True)
def _gaussian_kernel(sigma=1.2739826440811157, half_kernel_len=2, fmri_resolution=2):
    x = (
        np.arange(-half_kernel_len - 0.5, half_kernel_len + 1.5, dtype=np.float32)
        * fmri_resolution
    )
    x_cdf = _gaussian_cdf(x, sigma)
    return x_cdf[1:] - x_cdf[0:-1]


def get_mask_from_bold(bold_image, mask_threshold):
    # Pure numpy is faster than numba for this function
    return np.alltrue(bold_image > mask_threshold, axis=-1)


# def filter_bold_image_with_3d_kernel(bold_image, kernel_xyz,
# mask_image=None, mask_threshold=0):
# if mask_image is None:
# mask_image = get_mask_from_bold(bold_image, mask_threshold)


def filter_bold_image_with_separable_kernel(
    bold_image,
    kernel_x,
    kernel_y=None,
    kernel_z=None,
    mask_image=None,
    x_min_ind=None,
    x_max_ind=None,
    y_min_ind=None,
    y_max_ind=None,
    z_min_ind=None,
    z_max_ind=None,
    mask_threshold=0,
    numba=True,
):
    if kernel_y is None:
        kernel_y = kernel_x
    if kernel_z is None:
        kernel_z = kernel_x
    if mask_image is None:
        mask_image = get_mask_from_bold(bold_image, mask_threshold)
    if (
        (x_min_ind is None)
        or (x_max_ind is None)
        or (y_min_ind is None)
        or (y_max_ind is None)
        or (z_min_ind is None)
        or (z_max_ind is None)
    ):
        (
            x_min_ind,
            x_max_ind,
            y_min_ind,
            y_max_ind,
            z_min_ind,
            z_max_ind,
        ) = get_mask_range(mask_image)

    type_to_use = bold_image.dtype
    filtered_bold = np.zeros_like(bold_image, dtype=type_to_use)
    if numba:
        kernel_x = kernel_x.astype(type_to_use)
        kernel_y = kernel_y.astype(type_to_use)
        kernel_z = kernel_z.astype(type_to_use)
        filtered_bold = _filter_bold_image_with_separable_kernel_numba(
            bold_image,
            mask_image,
            filtered_bold,
            kernel_x,
            kernel_y,
            kernel_z,
            x_min_ind,
            x_max_ind,
            y_min_ind,
            y_max_ind,
            z_min_ind,
            z_max_ind,
        )
    else:
        filtered_bold = _filter_bold_image_with_separable_kernel(
            bold_image,
            mask_image,
            filtered_bold,
            kernel_x,
            kernel_y,
            kernel_z,
            x_min_ind,
            x_max_ind,
            y_min_ind,
            y_max_ind,
            z_min_ind,
            z_max_ind,
        )
    return filtered_bold


@jit(
    [
        f4[:, :, :, :](
            f4[:, :, :, :],
            b1[:, :, :],
            f4[:, :, :, :],
            f4[:],
            f4[:],
            f4[:],
            i8,
            i8,
            i8,
            i8,
            i8,
            i8,
        ),
        f8[:, :, :, :](
            f8[:, :, :, :],
            b1[:, :, :],
            f8[:, :, :, :],
            f8[:],
            f8[:],
            f8[:],
            i8,
            i8,
            i8,
            i8,
            i8,
            i8,
        ),
    ],
    parallel=True,
    nopython=True,
    fastmath=True,
    cache=True,
)
def _filter_bold_image_with_separable_kernel_numba(
    bold_image,
    mask_image,
    out,
    kernel_x,
    kernel_y,
    kernel_z,
    x_min_ind,
    x_max_ind,
    y_min_ind,
    y_max_ind,
    z_min_ind,
    z_max_ind,
):
    x, y, z, f = bold_image.shape

    kernel_x_half_len = int((len(kernel_x) - 1) / 2)
    kernel_y_half_len = int((len(kernel_y) - 1) / 2)
    kernel_z_half_len = int((len(kernel_z) - 1) / 2)

    x_indices = np.arange(x_min_ind, x_max_ind)
    y_indices = np.arange(y_min_ind, y_max_ind)
    z_indices = np.arange(z_min_ind, z_max_ind)

    for f_i in prange(f):
        for y_i in prange(y_min_ind, y_max_ind):
            for z_i in range(z_min_ind, z_max_ind):
                x_mask = mask_image[x_indices, y_i, z_i]
                if np.any(x_mask):
                    ind_range = x_indices[x_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1
                    bold_x_i = bold_image[start_i:end_i, y_i, z_i, f_i]
                    out[start_i:end_i, y_i, z_i, f_i] = np.convolve(bold_x_i, kernel_x)[
                        kernel_y_half_len:-kernel_y_half_len
                    ]

        for x_i in prange(x_min_ind, x_max_ind):
            for z_i in range(z_min_ind, z_max_ind):
                y_mask = mask_image[x_i, :, z_i][y_indices]
                if np.any(y_mask):
                    ind_range = y_indices[y_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1
                    bold_y_i = out[x_i, start_i:end_i, z_i, f_i]
                    bold_y_i_no_val_mask = bold_y_i == 0.0
                    bold_y_i[bold_y_i_no_val_mask] = bold_image[
                        x_i, start_i:end_i, z_i, f_i
                    ][bold_y_i_no_val_mask]
                    out[x_i, start_i:end_i, z_i, f_i] = np.convolve(bold_y_i, kernel_y)[
                        kernel_y_half_len:-kernel_y_half_len
                    ]

        for x_i in prange(x_min_ind, x_max_ind):
            for y_i in range(y_min_ind, y_max_ind):
                z_mask = mask_image[x_i, y_i, :][z_indices]
                if np.any(z_mask):
                    ind_range = z_indices[z_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1

                    bold_z_i = out[x_i, y_i, start_i:end_i, f_i]
                    bold_z_i_no_val_mask = bold_z_i == 0.0
                    bold_z_i[bold_z_i_no_val_mask] = bold_image[
                        x_i, y_i, start_i:end_i, f_i
                    ][bold_z_i_no_val_mask]
                    out[x_i, y_i, start_i:end_i, f_i] = np.convolve(bold_z_i, kernel_z)[
                        kernel_z_half_len:-kernel_z_half_len
                    ]
    return out


def _filter_bold_image_with_separable_kernel(
    bold_image,
    mask_image,
    out,
    kernel_x,
    kernel_y,
    kernel_z,
    x_min_ind,
    x_max_ind,
    y_min_ind,
    y_max_ind,
    z_min_ind,
    z_max_ind,
):
    x, y, z, f = bold_image.shape

    kernel_x_half_len = int((len(kernel_x) - 1) / 2)
    kernel_y_half_len = int((len(kernel_y) - 1) / 2)
    kernel_z_half_len = int((len(kernel_z) - 1) / 2)

    x_indices = np.arange(x_min_ind, x_max_ind)
    y_indices = np.arange(y_min_ind, y_max_ind)
    z_indices = np.arange(z_min_ind, z_max_ind)

    for f_i in range(f):
        for y_i in range(y_min_ind, y_max_ind):
            for z_i in range(z_min_ind, z_max_ind):
                x_mask = mask_image[x_indices, y_i, z_i]
                if np.any(x_mask):
                    ind_range = x_indices[x_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1
                    bold_x_i = bold_image[start_i:end_i, y_i, z_i, f_i]
                    out[start_i:end_i, y_i, z_i, f_i] = np.convolve(bold_x_i, kernel_x)[
                        kernel_y_half_len:-kernel_y_half_len
                    ]

        for x_i in range(x_min_ind, x_max_ind):
            for z_i in range(z_min_ind, z_max_ind):
                y_mask = mask_image[x_i, :, z_i][y_indices]
                if np.any(y_mask):
                    ind_range = y_indices[y_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1
                    bold_y_i = out[x_i, start_i:end_i, z_i, f_i]
                    bold_y_i_no_val_mask = bold_y_i == 0.0
                    bold_y_i[bold_y_i_no_val_mask] = bold_image[
                        x_i, start_i:end_i, z_i, f_i
                    ][bold_y_i_no_val_mask]
                    out[x_i, start_i:end_i, z_i, f_i] = np.convolve(bold_y_i, kernel_y)[
                        kernel_y_half_len:-kernel_y_half_len
                    ]

        for x_i in range(x_min_ind, x_max_ind):
            for y_i in range(y_min_ind, y_max_ind):
                z_mask = mask_image[x_i, y_i, :][z_indices]
                if np.any(z_mask):
                    ind_range = z_indices[z_mask]
                    start_i = ind_range[0] - kernel_x_half_len
                    if start_i < 0:
                        start_i = 0
                    end_i = ind_range[-1] + kernel_x_half_len
                    if end_i > (x - 1):
                        end_i = x - 1

                    bold_z_i = out[x_i, y_i, start_i:end_i, f_i]
                    bold_z_i_no_val_mask = bold_z_i == 0.0
                    bold_z_i[bold_z_i_no_val_mask] = bold_image[
                        x_i, y_i, start_i:end_i, f_i
                    ][bold_z_i_no_val_mask]
                    out[x_i, y_i, start_i:end_i, f_i] = np.convolve(bold_z_i, kernel_z)[
                        kernel_z_half_len:-kernel_z_half_len
                    ]
    return out


@jit(i8[:](b1[:, :, :]), parallel=True, nopython=True, fastmath=True, cache=True)
def get_mask_range(mask_image):
    x, y, z = mask_image.shape
    x_indices = np.arange(x)
    y_indices = np.arange(y)
    z_indices = np.arange(z)
    ans = np.array([x, 0, y, 0, z, 0], dtype=np.int64)

    for y_i in range(y):
        for z_i in range(z):
            mask_x = mask_image[:, y_i, z_i]
            if np.any(mask_x):
                ind = x_indices[mask_x]
                if ind[0] < ans[0]:
                    ans[0] = ind[0]
                if ind[-1] > ans[1]:
                    ans[1] = ind[-1]

    for x_i in range(x):
        for z_i in range(z):
            mask_y = mask_image[x_i, :, z_i]
            if np.any(mask_y):
                ind = y_indices[mask_y]
                if ind[0] < ans[2]:
                    ans[2] = ind[0]
                if ind[-1] > ans[3]:
                    ans[3] = ind[-1]

    for x_i in range(x):
        for y_i in range(y):
            mask_z = mask_image[x_i, y_i, :]
            if np.any(mask_z):
                ind = z_indices[mask_z]
                if ind[0] < ans[4]:
                    ans[4] = ind[0]
                if ind[-1] > ans[5]:
                    ans[5] = ind[-1]
    return ans


def flatten_bold(bold_image, mask_image):
    return bold_image[mask_image.astype(np.bool_), :].T


def spm_hrf(
    resolution_s,
    peak_delay_s=6.0,
    undershoot_delay_s=16.0,
    peak_dispersion=1.0,
    undershoot_dispersion=1.0,
    peak_undershoot_ratio=6.0,
    onset_s=0.0,
    model_length_s=32.0,
):
    time_stamp = np.array(
        np.arange(0, np.ceil(model_length_s / resolution_s)) - onset_s / resolution_s
    )
    peak_values = scipy.stats.gamma.pdf(
        time_stamp,
        peak_delay_s / peak_dispersion,
        loc=0,
        scale=peak_dispersion / resolution_s,
    )
    undershoot_values = scipy.stats.gamma.pdf(
        time_stamp,
        undershoot_delay_s / undershoot_dispersion,
        loc=0,
        scale=undershoot_dispersion / resolution_s,
    )
    hrf = peak_values - undershoot_values / peak_undershoot_ratio
    return hrf, time_stamp * resolution_s


def spm_d_hrf(
    resolution_s,
    peak_delay_s=6.0,
    undershoot_delay_s=16.0,
    peak_dispersion=1.0,
    undershoot_dispersion=1.0,
    peak_undershoot_ratio=6.0,
    onset_s=0.0,
    model_length_s=32.0,
    delta_t_s=1.0,
):
    return (
        spm_hrf(
            resolution_s,
            peak_delay_s,
            undershoot_delay_s,
            peak_dispersion,
            undershoot_dispersion,
            peak_undershoot_ratio,
            onset_s,
            model_length_s,
        )[0]
        - spm_hrf(
            resolution_s,
            peak_delay_s,
            undershoot_delay_s,
            peak_dispersion,
            undershoot_dispersion,
            peak_undershoot_ratio,
            delta_t_s,
            model_length_s,
        )[0]
    )


def spm_dd_hrf(
    resolution_s,
    peak_delay_s=6.0,
    undershoot_delay_s=16.0,
    peak_dispersion=1.0,
    undershoot_dispersion=1.0,
    peak_undershoot_ratio=6.0,
    onset_s=0.0,
    model_length_s=32.0,
    d_dispersion=0.01,
):
    return (
        spm_hrf(
            resolution_s,
            peak_delay_s,
            undershoot_delay_s,
            peak_dispersion,
            undershoot_dispersion,
            peak_undershoot_ratio,
            onset_s,
            model_length_s,
        )[0]
        - spm_hrf(
            resolution_s,
            peak_delay_s,
            undershoot_delay_s,
            peak_dispersion + d_dispersion,
            undershoot_dispersion,
            peak_undershoot_ratio,
            onset_s,
            model_length_s,
        )[0]
    ) / 0.01


def dct_drift_basis(frame_timestamps, hf_cut_hz=0.008, dtype=np.float64):
    num_frame = len(frame_timestamps)
    n_times = np.arange(num_frame)
    order = max(
        int(np.floor(2 * hf_cut_hz * (frame_timestamps[-1] - frame_timestamps[0]))), 1
    )
    drift = np.zeros((num_frame, order), dtype=dtype)
    nfct = 1
    for k in range(1, order):
        drift[:, k] = nfct * np.cos((np.pi / num_frame) * (n_times + 0.5) * k)
    drift[:, 0] = 1.0
    return drift


def onset_array_convolve_with_basis(
    onset_array, basis, TR_s, resolution_s=0.01, dtype=np.float64
):
    len_onset_array = len(onset_array)
    steps = int(TR_s / resolution_s)
    num_frames = int(len(onset_array) / steps)
    basis_len, num_basis = basis.shape
    output = np.zeros((num_frames, num_basis), dtype=dtype)
    for i in range(num_basis):
        output[:, i] = np.convolve(onset_array, basis[:, i])[
            np.arange(0, len_onset_array, steps)[0:num_frames]
        ]
    return output


def event_list_to_onset_array(
    event_list, run_len_s, resolution_s=0.01, method="nonlinear"
):
    onset_array = np.zeros(int(run_len_s / resolution_s), dtype=np.float32)
    if method == "constant_impulse":
        step = int(4 / resolution_s)
        for event in event_list:
            onset_frame = int(event["onset_s"] / resolution_s)
            stop_frame = int((event["onset_s"] + event["duration_s"]) / resolution_s)
            onset_array[np.arange(onset_frame, stop_frame, step)] = 1
    elif method == "constant_epoch_2s":
        for event in event_list:
            onset_frame = int(event["onset_s"] / resolution_s)
            stop_frame = int((event["onset_s"] + 2) / resolution_s)
            onset_array[onset_frame:stop_frame] = 1 / (stop_frame - onset_frame)
    elif method == "variable_epoch":
        for event in event_list:
            onset_frame = int(event["onset_s"] / resolution_s)
            stop_frame = int((event["onset_s"] + event["duration_s"]) / resolution_s)
            onset_array[onset_frame:stop_frame] = 1 / (stop_frame - onset_frame)
    elif method == "nonlinear":
        gam_weight_vector = gam(np.arange(1, run_len_s + 1))
        gam_weight_mask = np.zeros(len(gam_weight_vector), dtype=bool)
        prev_onset_time_s = 0
        for event in event_list:
            onset_frame = int(event["onset_s"] / resolution_s)
            onset_time_s = event["onset_s"]
            gam_weight_mask = np.roll(
                gam_weight_mask, int(onset_time_s - prev_onset_time_s)
            )
            sat = np.sum(gam_weight_vector[gam_weight_mask])
            onset_array[onset_frame] = max(0, 1 - sat)
            gam_weight_mask[0] = True
            duration_left = int(round(event["duration_s"] - 1))
            for d_i in range(duration_left):
                onset_frame = int(onset_frame + 1 / resolution_s)
                onset_time_s += 1
                sat = np.sum(gam_weight_vector[gam_weight_mask])
                onset_array[onset_frame] = max(0, 1 - sat)
                gam_weight_mask = np.roll(gam_weight_mask, 1)
                gam_weight_mask[0] = True
            prev_onset_time_s = onset_time_s
    return onset_array


def event_dict_list_to_letter_event_list(event_dict_list):
    letter_event_list = []
    letter_event_list.append(
        {
            "onset_s": float(event_dict_list[0]["onset"]),
            "duration_s": float(event_dict_list[0]["duration"]),
            "letter": event_dict_list[0]["letter"],
        }
    )
    for event_dict in event_dict_list[1:]:
        if float(event_dict["onset"]) != letter_event_list[-1]["onset_s"]:
            letter_event_list.append(
                {
                    "onset_s": float(event_dict["onset"]),
                    "duration_s": float(event_dict["duration"]),
                    "letter": event_dict["letter"],
                }
            )
    return letter_event_list


def event_dict_list_to_event_by_type(
    event_dict_list, trial_types_to_ignore=["rest"], merge_adjacent=True
):
    event_by_type = {}
    for event_dict in event_dict_list:
        trial_type = event_dict["trial_type"]
        if trial_type in trial_types_to_ignore:
            continue
        if trial_type not in event_by_type:
            event_by_type[trial_type] = []

        # Check to see if the previous stimulus is adjacent to the current one
        if len(event_by_type[trial_type]):
            previous_event = event_by_type[trial_type][-1]
            previous_stimulus_end_time_s = (
                previous_event["onset_s"] + previous_event["duration_s"]
            )
            # If the previous stimulus extends to the onset of the current,
            # merge the two to one
            if merge_adjacent:
                if previous_stimulus_end_time_s == float(event_dict["onset"]):
                    event_by_type[trial_type][-1]["duration_s"] += float(
                        event_dict["duration"]
                    )
                    continue

        event_by_type[trial_type].append(
            {
                "onset_s": float(event_dict["onset"]),
                "duration_s": float(event_dict["duration"]),
            }
        )

    event_list = list(event_by_type.keys())
    return event_by_type


def gam(t, discount=0.7):
    return (1 - 0.6658) * (np.exp(-discount * t) / np.exp(-discount))


def prep_bold_info(bold_info):
    prepped_bold_info = {}
    prepped_bold_info["event_tsv_content"] = load_xsv(bold_info["event_path"])
    prepped_bold_info["event_by_type"] = event_dict_list_to_event_by_type(
        prepped_bold_info["event_tsv_content"]
    )
    prepped_bold_info["onset_array_by_type"] = {
        e_type: event_list_to_onset_array(
            events, bold_info["run_len_s"], bold_info["resolution_s"]
        )
        for e_type, events in prepped_bold_info["event_by_type"].items()
    }
    prepped_bold_info["regressors_by_type"] = {
        e_type: onset_array_convolve_with_basis(
            onset_array,
            bold_info["basis"],
            bold_info["TR_s"],
            bold_info["resolution_s"],
        )
        for e_type, onset_array in prepped_bold_info["onset_array_by_type"].items()
    }
    regressor_types = sorted(list(prepped_bold_info["regressors_by_type"].keys()))
    prepped_bold_info["regressor_types"] = regressor_types
    # all_stimuli_regressors = np.hstack([regressors_by_type for e_type,
    # regressors_by_type in prepped_bold_info['regressors_by_type'].items()])
    all_stimuli_regressors = np.hstack(
        [
            prepped_bold_info["regressors_by_type"][regressor_type]
            for regressor_type in regressor_types
        ]
    )
    timeseries_tsv_content = load_xsv(bold_info["timeseries_tsv_path"])
    prepped_bold_info["timeseries_tsv_raw"] = load_file(
        bold_info["timeseries_tsv_path"]
    )
    confound_regressors = extract_confound_regressors(timeseries_tsv_content)
    motion_outliers = extract_confound_regressors(
        timeseries_tsv_content,
        USE_MOTION=False,
        USE_MOTION_DERIVATIVE1=False,
        USE_MOTION_POW2=False,
        USE_MOTION_DERIVATIVE1_POW2=False,
        USE_CSF=False,
        USE_WM=False,
        USE_MOTION_OUTLIER=True,
    ).astype(dtype=bool)
    prepped_bold_info["motion_outlier_index"] = motion_outliers.nonzero()[0]
    prepped_bold_info["all_regressors"] = np.hstack(
        (all_stimuli_regressors, bold_info["dct_regressors"], confound_regressors)
    )
    return prepped_bold_info


DEFAULT_MASKS_REGIONS = [
    {"region_name": "left_mask_image", "vals": [13]},
    {"region_name": "right_mask_image", "vals": [14]},
    {"region_name": "mask_image", "vals": [13, 14]},
]


def prepare_bold_info(
    input_bold_path,
    input_event_path,
    template_flow_api,
    nib,
    atlas="HOCPAL",
    desc="th0",
    image_smoothing_fwhm_mm=2,
    regions=DEFAULT_MASKS_REGIONS,
):
    bold_image_handle = nib.load(input_bold_path)
    # Infer from the file name to get the templateflow template used by
    # fmriperep.
    info = extract_bold_file_info_from_name(input_bold_path)
    bold_file_info = extract_bold_file_info_from_name(input_bold_path)
    bold_image_resolution_mm = bold_image_handle.header["pixdim"][1:4]
    TR_s = bold_image_handle.header["pixdim"][4]
    num_frames = bold_image_handle.header["dim"][4]
    run_len_s = num_frames * TR_s
    atlas_image_path = template_flow_api.get(
        template=bold_file_info["space"],
        resolution=bold_image_resolution_mm[0],
        atlas=atlas,
        desc=desc,
    )
    atlas_image_handle = nib.load(atlas_image_path)
    atlas_image = atlas_image_handle.get_fdata(dtype=np.float32).astype(np.int32)
    mask_image_dictionary = {
        region["region_name"]: reduce(
            np.logical_or, [atlas_image == val for val in region["vals"]]
        )
        for region in regions
    }

    full_brain_mask_image_handle = nib.load(bold_file_info["brain_mask_file_path"])
    full_brain_mask_image = full_brain_mask_image_handle.get_fdata(
        dtype=np.float32
    ).astype(np.bool_)
    mask_image_dictionary = {
        key: np.logical_and(mask, full_brain_mask_image)
        for key, mask in mask_image_dictionary.items()
    }

    # Generate basis to use for all events
    # use a resoliution higher but devisable by TR_s and stimulus time
    resolution_s = 0.01
    bold, time_stamp = spm_hrf(resolution_s)
    dbold = spm_d_hrf(resolution_s)
    ddbold = spm_dd_hrf(resolution_s)
    bold_max = np.max(bold)
    bold, dbold, ddbold = bold / bold_max, dbold / bold_max, ddbold / bold_max
    basis = np.vstack((bold, dbold, ddbold)).T

    # Drift regressors
    frequency_cut_hz = 1 / 128  # SPM default
    TR_timestamp = np.arange(num_frames) * TR_s
    dct_regressors = dct_drift_basis(TR_timestamp, frequency_cut_hz)

    # load stimuli of the first run to use as a reference
    # sample_bold_stimuli_path = input_event_paths[0]
    # stimuli_events = load_xsv(sample_bold_stimuli_path)
    stimuli_events = load_xsv(input_event_path)
    letter_event_list = event_dict_list_to_letter_event_list(stimuli_events)
    event_by_type = event_dict_list_to_event_by_type(stimuli_events)

    # Generate regressors to use for all events
    onset_array_by_type = {
        e_type: event_list_to_onset_array(events, run_len_s, resolution_s)
        for e_type, events in event_by_type.items()
    }
    regressors_by_type = {
        e_type: onset_array_convolve_with_basis(onset_array, basis, TR_s, resolution_s)
        for e_type, onset_array in onset_array_by_type.items()
    }
    regressor_names = []
    for e_type, regressors in regressors_by_type.items():
        for i in range(regressors.shape[1]):
            regressor_names.append(f"{e_type}_{i}")
    all_stimuli_regressors = np.hstack(list(regressors_by_type.values()))
    all_stimuli_regressors_type = list(regressors_by_type.keys())

    # Create the contrast matrix
    num_regressors_and_reg_sum = []
    num_regressor_sum = 0
    for e_type, regressors in regressors_by_type.items():
        num_regressors_and_reg_sum.append((regressors.shape[1], num_regressor_sum))
        num_regressor_sum += regressors.shape[1]
    total_num_regressors = all_stimuli_regressors.shape[1]

    contrasts = []
    # contrast matrix - stimulus vs background
    for num_regressors, reg_sum in num_regressors_and_reg_sum:
        contrast = np.zeros((1, total_num_regressors))
        contrast[0, reg_sum : reg_sum + num_regressors] = 1 / num_regressors
        contrasts.append(contrast)
    # contrast matrix - combinations of 2 stimulus contrast
    for r_1, r_2 in combinations(num_regressors_and_reg_sum, 2):
        contrast = np.zeros((1, total_num_regressors))
        contrast[0, r_1[1] : r_1[1] + r_1[0]] = 1 / r_1[0]
        contrast[0, r_2[1] : r_2[1] + r_2[0]] = -1 / r_2[0]
        contrasts.append(contrast)
    contrasts = np.vstack(contrasts)

    derivative_boost_masks = []
    for contrast in contrasts:
        derivative_boost_mask_pos = contrast > 0
        derivative_boost_mask_neg = contrast < 0
        if np.any(derivative_boost_mask_neg):
            derivative_boost_mask = np.stack(
                (derivative_boost_mask_pos, derivative_boost_mask_neg)
            )
        else:
            derivative_boost_mask = derivative_boost_mask_pos[np.newaxis, :]
        derivative_boost_masks.append(derivative_boost_mask)

    bold_info = extract_bold_file_info_from_name(input_bold_path)
    bold_info["event_path"] = input_event_path
    bold_info["run_len_s"] = run_len_s
    bold_info["resolution_s"] = resolution_s
    bold_info["bold_image_resolution_mm"] = bold_image_resolution_mm
    bold_info["basis"] = basis
    bold_info["TR_s"] = TR_s
    bold_info["dct_regressors"] = dct_regressors
    bold_info["full_brain_mask_image"] = full_brain_mask_image
    bold_info["mask_image"] = mask_image_dictionary["mask_image"]
    # bold_info['expanded_mask'] = expanded_mask
    # bold_info['expanded_mask_image_range'] = expanded_mask_image_range
    # bold_info['image_smoothing_kernel'] = image_smoothing_kernel
    # bold_info['ar_smoothing_kernel'] = ar_smoothing_kernel
    bold_info["contrasts"] = contrasts
    bold_info["derivative_boost_masks"] = derivative_boost_masks
    bold_info["mask_image_dictionary"] = mask_image_dictionary
    bold_info["calibration_mask_dictionary"] = {
        "b": "mask_image",
        "f_l": "mask_image",
        "f_r": "mask_image",
        "h_l": "mask_image",
        "h_r": "mask_image",
        "t": "mask_image",
    }
    processed_bold_info = prep_bold_info(bold_info)
    print(
        f"Finished processing:\nBOLD path: {input_bold_path}\n"
        + f"Event path: {input_event_path}\n-------------------------"
    )
    return bold_info, processed_bold_info


def generate_design_matrix(num_frames, stimulus_timestamps, sf_Hz):
    # use a resoliution higher but devisable by TR_s and stimulus time
    resolution_s = 0.01
    bold, time_stamp = spm_hrf(resolution_s)
    dbold = spm_d_hrf(resolution_s)
    ddbold = spm_dd_hrf(resolution_s)
    bold_max = np.max(bold)
    bold, dbold, ddbold = bold / bold_max, dbold / bold_max, ddbold / bold_max
    basis = np.vstack((bold, dbold, ddbold)).T
    TR_s = 1 / sf_Hz

    # Drift regressors
    frequency_cut_hz = 1 / 128  # SPM default
    TR_timestamp = np.arange(num_frames) * TR_s
    dct_regressors = dct_drift_basis(TR_timestamp, frequency_cut_hz)
    print(f"dct_regressors: {dct_regressors.shape}")

    # Design matrix
    high_res_num_frames = int(np.ceil(num_frames * TR_s / resolution_s))
    onset_matrix_high_res = np.zeros((high_res_num_frames, len(stimulus_timestamps)))
    design_matrix = np.zeros((num_frames, 3 * len(stimulus_timestamps)))
    for s_i, stimulus in enumerate(stimulus_timestamps):
        for e_j, event_time in enumerate(stimulus):
            onset_matrix_high_res[int(np.ceil(event_time / resolution_s)), s_i] = 1
        design_matrix[:, s_i * 3 : s_i * 3 + 3] = onset_array_convolve_with_basis(
            onset_matrix_high_res[:, s_i], basis, TR_s, resolution_s
        )[0:num_frames, :]

    return TR_timestamp, design_matrix, dct_regressors
