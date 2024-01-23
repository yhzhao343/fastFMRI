import numpy as np
import numba as nb
import math
import scipy
import scipy.linalg
from numba import jit, prange, f4, f8, i4, i8


def glm(flatten_bold, regressors, regressors_T=None, numba=False):
    # Pure numpy is faster than numba for this function
    if regressors_T is None:
        regressors_T = regressors.T
    if numba:
        if not flatten_bold.data.c_contiguous:
            flatten_bold = np.ascontiguousarray(flatten_bold)
        if not regressors.data.c_contiguous:
            regressors = np.ascontiguousarray(regressors)

        if not regressors_T.data.c_contiguous:
            regressors_T = np.ascontiguousarray(regressors_T)

        t, r = regressors.shape
        type_to_use = flatten_bold.dtype
        v = flatten_bold.shape[1]
        regressors = regressors.astype(type_to_use)
        regressors_T = regressors_T.astype(type_to_use)
        xtx = np.empty((r, r), dtype=type_to_use)
        xty = np.empty((r, v), dtype=type_to_use)
        b = np.empty((r, flatten_bold.shape[1]), dtype=type_to_use)
        b = _glm_numba(flatten_bold, xtx, xty, regressors, regressors_T, b)
    else:
        b = np.linalg.solve(regressors_T @ regressors, regressors_T @ flatten_bold)
    return b


@jit(
    [
        f8[:, :](
            f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[:, ::1]
        ),
        f4[:, :](
            f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1], f4[:, ::1]
        ),
    ],
    nopython=True,
    fastmath=True,
    cache=True,
)
def _glm_numba(flatten_bold, xtx, xty, regressors, regressors_T, b):
    xtx = regressors_T @ regressors
    xty = regressors_T @ flatten_bold
    return np.linalg.solve(xtx, xty)


def pinv_xtx(regressors, regressors_T=None, numba=False, rcond=1e-15):
    if regressors_T is None:
        regressors_T = regressors.T
    if numba:
        if not regressors.data.c_contiguous:
            regressors = np.ascontiguousarray(regressors)
        if not regressors_T.data.c_contiguous:
            regressors_T = np.ascontiguousarray(regressors_T)
        t, r = regressors.shape
        xtx = np.empty((r, r), dtype=np.float64)
        inv_xtx = _pinv_xtx_numba_64(
            xtx, regressors.astype(np.float64), regressors_T.astype(np.float64), rcond
        )
    else:
        inv_xtx = np.linalg.pinv(
            (regressors_T.astype(np.float64) @ regressors.astype(np.float64)),
            rcond=rcond,
        )
    return inv_xtx


@jit(
    f8[:, :](f8[:, ::1], f8[:, ::1], f8[:, ::1], f8),
    nopython=True,
    fastmath=True,
    cache=True,
)
def _pinv_xtx_numba_64(xtx, regressors, regressors_T, rcond=1e-15):
    xtx = regressors_T @ regressors
    return np.linalg.pinv(xtx)


def glm_using_inv_xtx(inv_xtx, regressors_T, flatten_bold, numba=False):
    if numba:
        flatten_bold = flatten_bold.astype(inv_xtx.dtype)
        regressors_T = regressors_T.astype(inv_xtx.dtype)
        if not flatten_bold.data.c_contiguous:
            flatten_bold = np.ascontiguousarray(flatten_bold)
        if not inv_xtx.data.c_contiguous:
            inv_xtx = np.ascontiguousarray(inv_xtx)
        if not regressors_T.data.c_contiguous:
            regressors_T = np.ascontiguousarray(regressors_T)
        r, t = regressors_T.shape
        inv_xtx_xt = np.empty((r, t), dtype=inv_xtx.dtype)
        b = _glm_using_inv_xtx_numba(inv_xtx, inv_xtx_xt, regressors_T, flatten_bold)
    else:
        b = inv_xtx @ regressors_T @ flatten_bold
    return b.astype(flatten_bold.dtype)


@jit(
    f8[:, :](f8[:, ::1], f8[:, ::1], f8[:, ::1], f8[:, ::1]),
    nopython=True,
    fastmath=True,
    cache=True,
)
def _glm_using_inv_xtx_numba(inv_xtx, inv_xtx_xt, regressors_T, flatten_bold):
    inv_xtx_xt = inv_xtx @ regressors_T
    return inv_xtx_xt @ flatten_bold


def get_e_sse(y, y_pred, numba=True):
    y = y.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    if numba:
        sse = np.empty(y.shape[1], dtype=np.float64)
        e = np.empty_like(y, dtype=np.float64)
        return _get_e_sse_numba(y, y_pred, e, sse)
    else:
        e = y - y_pred
        e2 = e**2
        return e, np.sum(e2, axis=0)


@jit(
    nb.types.Tuple((f8[:, :], f8[:]))(f8[:, :], f8[:, :], f8[:, :], f8[:]),
    parallel=True,
    nopython=True,
    fastmath=True,
    cache=True,
)
def _get_e_sse_numba(y, y_pred, e, sse):
    e = y - y_pred
    v = e.shape[1]
    for v_i in prange(v):
        sse[v_i] = np.sum((e[:, v_i] ** 2))
    return e, sse


def batch_auto_corr(x, order=7):
    auto_corr = np.zeros((order + 1, x.shape[1]), dtype=np.float64)
    return _batch_auto_corr(x, order, auto_corr)


@jit(
    f8[:, :](f8[:, ::1], i4, f8[:, :]),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _batch_auto_corr(x, order, auto_corr):
    for i in prange(order + 1):
        auto_corr[i, :] = np.sum(x[0 : x.shape[0] - i, :] * x[i:, :], axis=0)
    return auto_corr


def toeplitz(arr, numba=True):
    arr_len = len(arr)
    out = np.empty((arr_len, arr_len), dtype=np.float64)
    if numba:
        out = _toeplitz(arr, out)
    else:
        for i in prange(arr_len):
            out[i, i:] = arr[0 : (arr_len - i)]
            out[i, :i] = arr[1 : i + 1][::-1]
    return out


@jit(f8[:, :](f8[:], f8[:, :]), nopython=True, parallel=True, fastmath=True, cache=True)
def _toeplitz(arr, out):
    arr_len = len(arr)
    for i in prange(arr_len):
        out[i, i:] = arr[0 : (arr_len - i)]
        out[i, :i] = arr[1 : i + 1][::-1]
    return out


def batch_ar_yule(auto_corr, use_Levinson_Recursion=False, numba=True, rcond=1e-15):
    ar_order, v = auto_corr.shape
    ar = np.empty_like(auto_corr)
    if numba:
        if rcond is not None:
            ar = _batch_ar_yule_pinv(auto_corr, ar, rcond=rcond)
        else:
            ar = _batch_ar_yule(auto_corr, ar)
    else:
        if use_Levinson_Recursion:
            for v_i in range(v):
                ar[1:, v_i] = scipy.linalg.solve_toeplitz(
                    auto_corr[:-1, v_i], -auto_corr[1:, v_i]
                )
        else:
            if rcond is not None:
                for v_i in range(v):
                    toeplitz_mat = toeplitz(auto_corr[:-1, v_i], numba=False)
                    y = -auto_corr[1:, v_i]
                    inv_toeplitz = np.linalg.pinv(toeplitz_mat, rcond=rcond)
                    ar[1:, v_i] = inv_toeplitz @ y
            else:
                for v_i in range(v):
                    toeplitz_mat = toeplitz(auto_corr[:-1, v_i], numba=False)
                    y = -auto_corr[1:, v_i]
                    ar[1:, v_i] = np.linalg.solve(toeplitz_mat, y)
        ar[0, :] = 1
    return ar


@jit(
    f8[:, :](f8[:, :], f8[:, :], f8),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _batch_ar_yule_pinv(auto_corr, ar, rcond=1e-15):
    ar_order, v = auto_corr.shape
    arr_len = ar_order - 1
    for v_i in prange(v):
        arr = auto_corr[:-1, v_i]
        toeplitz_mat = np.empty((arr_len, arr_len), dtype=np.float64)
        for i in prange(arr_len):
            toeplitz_mat[i, i:] = arr[0 : (arr_len - i)]
            toeplitz_mat[i, :i] = arr[1 : i + 1][::-1]
        y = -auto_corr[1:, v_i]
        inv_toeplitz = np.linalg.pinv(toeplitz_mat, rcond=rcond)
        ar[1:, v_i] = inv_toeplitz @ y
    ar[0, :] = 1
    return ar


@jit(
    f8[:, :](f8[:, :], f8[:, :]),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _batch_ar_yule(auto_corr, ar):
    ar_order, v = auto_corr.shape
    arr_len = ar_order - 1
    for v_i in prange(v):
        arr = auto_corr[:-1, v_i]
        toeplitz_mat = np.empty((arr_len, arr_len), dtype=np.float64)
        for i in prange(arr_len):
            toeplitz_mat[i, i:] = arr[0 : (arr_len - i)]
            toeplitz_mat[i, :i] = arr[1 : i + 1][::-1]
        y = -auto_corr[1:, v_i]
        ar[1:, v_i] = np.linalg.solve(toeplitz_mat, y)
    ar[0, :] = 1
    return ar


def batch_sse_from_auto_corr_and_ar(auto_corr, ar, numba=True):
    # Calculate the sum of squared error of the input white noise
    # Divide result by the len of data to the the variance, which in matlab is:
    # e in [ar, e]=aryule(x, order)
    if numba:
        v = auto_corr.shape[1]
        sse = np.empty(v, dtype=np.float64)
        sse = _batch_sse_from_auto_corr_and_ar(auto_corr, ar[1:, :], sse)
    else:
        sse = auto_corr[0, :] - np.sum(ar[1:, :] * (-auto_corr[1:, :]), axis=0)
    return sse


@jit(
    f8[:](f8[:, :], f8[:, :], f8[:]),
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _batch_sse_from_auto_corr_and_ar(auto_corr, ar, sse):
    sse = auto_corr[0, :] - np.sum(ar * (-auto_corr[1:, :]), axis=0)
    return sse


@jit(
    [
        f4[:, ::1](f4[:], f4[:, ::1], f4[:, ::1]),
        f8[:, ::1](f8[:], f8[:, ::1], f8[:, ::1]),
    ],
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _get_inv_v_from_ar(ar, A_eye, inv_v):
    len_ar = len(ar)
    for i in prange(len_ar - 1):
        A_eye[i, (1 + i) :] = -ar[1 : len_ar - i]
    inv_v[:len_ar, :len_ar] = A_eye.T @ A_eye
    for i in prange(len_ar, inv_v.shape[0]):
        inv_v[i, i - len_ar + 1 : i + 1] = inv_v[len_ar - 1, :len_ar]
        inv_v[i - len_ar + 1 : i, i] = inv_v[len_ar - 1, : len_ar - 1]
    return inv_v


def get_inv_v_from_ar(ar, num_frame):
    # Reference: https://mandymejia.com/2016/11/06/how-to-efficiently-prewhiten-fmri-timeseries-the-right-way/
    A_eye = np.eye(len(ar), dtype=np.float32)
    inv_v = np.zeros((num_frame, num_frame), dtype=np.float32)
    inv_v = _get_inv_v_from_ar(ar, A_eye, inv_v)
    return inv_v


def inv_v_to_prewhiten_matrix(inv_v):
    u, s, ut = np.linalg.svd(inv_v, hermitian=True)
    w = u @ np.diag(np.sqrt(s)) @ ut
    return w


@jit(
    [f8[:, :](f8[:], i4, f8[:], f8[:, :]), f8[:, :](f8[:], i8, f8[:], f8[:, :])],
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _get_ar_covar_numba(ar, ts_len, weight, covar):
    ts_len = weight.size
    len_ar = ar.size
    weight[0] = 1
    i_end = ts_len - len_ar
    for i in prange(i_end + 1):
        weight[(i + 1) : (i + len_ar)] += ar[1:] ** (i + 1)
    for j in prange(0, len_ar - 2):
        weight[(i_end + j + 2) : ts_len] += ar[1 : len_ar - j - 1] ** (i_end + j + 2)
    covar = _toeplitz(weight, covar)
    return covar


def get_ar_covar(ar, ts_len, numba=True):
    weight = np.zeros(ts_len, dtype=np.float64)
    len_ar = ar.size
    if numba:
        covar = np.empty((ts_len, ts_len), dtype=np.float64)
        covar = _get_ar_covar_numba(ar, ts_len, weight, covar)
    else:
        weight[0] = 1
        i_end = ts_len - len_ar
        for i in range(i_end + 1):
            weight[(i + 1) : (i + len_ar)] += ar[1:] ** (i + 1)
        for j in range(0, len_ar - 2):
            weight[(i_end + j + 2) : ts_len] += ar[1 : len_ar - j - 1] ** (
                i_end + j + 2
            )
        covar = scipy.linalg.toeplitz(weight)
    return covar


def batch_prewhiten_re_estimates(
    ar,
    regressors,
    flatten_bold,
    error,
    regressors_T=None,
    numba=True,
    exact=False,
    rcond=1e-15,
):
    len_ar = ar.shape[0]
    r = regressors.shape[1]
    t, v = flatten_bold.shape
    inv_xt_iv_x = np.empty((v, r, r), dtype=np.float64)
    b_out = np.empty((v, r), dtype=flatten_bold.dtype)
    sse = np.empty(v, dtype=np.float64)
    ar = ar.astype(np.float64)
    inv_v = np.zeros((t, t), dtype=np.float64)

    regressors = regressors.astype(np.float64)
    if regressors_T is None:
        regressors_T = regressors.T
    else:
        regressors_T = regressors_T.astype(np.float64)

    if numba:
        if not regressors_T.data.c_contiguous:
            regressors_T = np.ascontiguousarray(regressors_T)
        if not regressors.data.c_contiguous:
            regressors = np.ascontiguousarray(regressors)
        if exact:
            weight = np.zeros(t, dtype=np.float64)
            covar = np.empty((t, t), dtype=np.float64)
            inv_xt_iv_x, b_out, sse = _batch_prewhiten_re_estimates_exact_numba(
                ar,
                regressors_T,
                regressors,
                inv_v,
                covar,
                weight,
                flatten_bold,
                error,
                inv_xt_iv_x,
                b_out,
                sse,
                rcond,
            )
        else:
            A_eye = np.eye(len_ar, dtype=np.float64)
            inv_xt_iv_x, b_out, sse = _batch_prewhiten_re_estimates_numba(
                ar,
                regressors_T,
                regressors,
                A_eye,
                inv_v,
                flatten_bold,
                error,
                inv_xt_iv_x,
                b_out,
                sse,
                rcond,
            )
    else:
        if exact:
            weight = np.zeros(t, dtype=np.float64)
            covar = np.empty((t, t), dtype=np.float64)
            inv_xt_iv_x, b_out, sse = _batch_prewhiten_re_estimates_exact(
                ar,
                regressors_T,
                regressors,
                inv_v,
                covar,
                weight,
                flatten_bold,
                error,
                inv_xt_iv_x,
                b_out,
                sse,
                rcond,
            )
        else:
            A_eye = np.eye(len_ar, dtype=np.float64)
            inv_xt_iv_x, b_out, sse = _batch_prewhiten_re_estimates(
                ar,
                regressors_T,
                regressors,
                A_eye,
                inv_v,
                flatten_bold,
                error,
                inv_xt_iv_x,
                b_out,
                sse,
                rcond,
            )

    return inv_xt_iv_x, b_out.T, sse


@jit(
    [
        nb.types.Tuple((f8[:, :, ::1], f8[:, ::1], f8[::1]))(
            f8[:, :],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, :],
            f8[:, ::1],
            f8[:, :, ::1],
            f8[:, ::1],
            f8[::1],
            f8,
        ),
        nb.types.Tuple((f8[:, :, ::1], f4[:, ::1], f8[::1]))(
            f8[:, :],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f4[:, :],
            f8[:, ::1],
            f8[:, :, ::1],
            f4[:, ::1],
            f8[::1],
            f8,
        ),
    ],
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def _batch_prewhiten_re_estimates_numba(
    ar,
    regressors_T,
    regressors,
    A_eye,
    inv_v,
    flatten_bold,
    error,
    inv_xt_iv_x,
    b_out,
    sse,
    rcond=1e-15,
):
    t, v = flatten_bold.shape
    len_ar = ar.shape[0]
    for v_i in range(v):
        inv_v = _get_inv_v_from_ar(ar[:, v_i], A_eye, inv_v)
        xt_iv = regressors_T @ inv_v
        inv_xt_iv_x[v_i] = np.linalg.pinv(xt_iv @ regressors)
        xt_iv_y = xt_iv.astype(flatten_bold.dtype) @ np.copy(flatten_bold[:, v_i])
        b_out[v_i] = inv_xt_iv_x[v_i].astype(flatten_bold.dtype) @ xt_iv_y
        e_i = np.copy(error[:, v_i])
        e_i_T = np.copy(e_i.T)
        sse[v_i] = (e_i_T @ inv_v) @ e_i
    return inv_xt_iv_x, b_out, sse


def _batch_prewhiten_re_estimates(
    ar,
    regressors_T,
    regressors,
    A_eye,
    inv_v,
    flatten_bold,
    error,
    inv_xt_iv_x,
    b_out,
    sse,
    rcond=1e-15,
):
    t, v = flatten_bold.shape
    len_ar = ar.shape[0]
    for v_i in range(v):
        inv_v = _get_inv_v_from_ar(ar[:, v_i], A_eye, inv_v)
        xt_iv = regressors_T @ inv_v
        inv_xt_iv_x[v_i] = np.linalg.pinv(xt_iv @ regressors)
        xt_iv_y = xt_iv.astype(flatten_bold.dtype) @ np.copy(flatten_bold[:, v_i])
        b_out[v_i] = inv_xt_iv_x[v_i].astype(flatten_bold.dtype) @ xt_iv_y
        e_i = np.copy(error[:, v_i])
        e_i_T = np.copy(e_i.T)
        sse[v_i] = (e_i_T @ inv_v) @ e_i
    return inv_xt_iv_x, b_out, sse


@jit(
    [
        nb.types.Tuple((f8[:, :, ::1], f8[:, ::1], f8[::1]))(
            f8[:, :],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[::1],
            f8[:, :],
            f8[:, ::1],
            f8[:, :, ::1],
            f8[:, ::1],
            f8[::1],
            f8,
        ),
        nb.types.Tuple((f8[:, :, ::1], f4[:, ::1], f8[::1]))(
            f8[:, :],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[:, ::1],
            f8[::1],
            f4[:, :],
            f8[:, ::1],
            f8[:, :, ::1],
            f4[:, ::1],
            f8[::1],
            f8,
        ),
    ],
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def _batch_prewhiten_re_estimates_exact_numba(
    ar,
    regressors_T,
    regressors,
    inv_v,
    covar,
    weight,
    flatten_bold,
    error,
    inv_xt_iv_x,
    b_out,
    sse,
    rcond=1e-15,
):
    t, v = flatten_bold.shape
    len_ar = ar.shape[0]
    for v_i in range(v):
        covar = _get_ar_covar_numba(ar[:, v_i], t, weight, covar)
        inv_v = np.linalg.pinv(covar)
        xt_iv = regressors_T @ inv_v
        inv_xt_iv_x[v_i] = np.linalg.pinv(xt_iv @ regressors)
        xt_iv_y = xt_iv.astype(flatten_bold.dtype) @ np.copy(flatten_bold[:, v_i])
        b_out[v_i] = inv_xt_iv_x[v_i].astype(flatten_bold.dtype) @ xt_iv_y
        e_i = np.copy(error[:, v_i])
        e_i_T = np.copy(e_i.T)
        sse[v_i] = (e_i_T @ inv_v) @ e_i
    return inv_xt_iv_x, b_out, sse


def _batch_prewhiten_re_estimates_exact(
    ar,
    regressors_T,
    regressors,
    inv_v,
    covar,
    weight,
    flatten_bold,
    error,
    inv_xt_iv_x,
    b_out,
    sse,
    rcond=1e-15,
):
    t, v = flatten_bold.shape
    len_ar = ar.shape[0]
    for v_i in range(v):
        covar = _get_ar_covar_numba(ar[:, v_i], t, weight, covar)
        inv_v = np.linalg.pinv(covar)
        xt_iv = regressors_T @ inv_v
        inv_xt_iv_x[v_i] = np.linalg.pinv(xt_iv @ regressors)
        xt_iv_y = xt_iv.astype(flatten_bold.dtype) @ np.copy(flatten_bold[:, v_i])
        b_out[v_i] = inv_xt_iv_x[v_i].astype(flatten_bold.dtype) @ xt_iv_y
        e_i = np.copy(error[:, v_i])
        e_i_T = np.copy(e_i.T)
        sse[v_i] = (e_i_T @ inv_v) @ e_i
    return inv_xt_iv_x, b_out, sse


def pad_contrast(num_regressor, c):
    num_relevant_regressor, num_contrast = c.shape
    c_padded = np.zeros((num_regressor, num_contrast), dtype=c.dtype)
    c_padded[:num_relevant_regressor, :] = c
    return c_padded


def batch_diag(ct_inv_xt_iv_x_c, numba=True):
    v, c, _ = ct_inv_xt_iv_x_c.shape
    out = np.empty((v, c), dtype=ct_inv_xt_iv_x_c.dtype)
    if numba:
        out = _batch_diag_numba(ct_inv_xt_iv_x_c, out)
    else:
        for v_i in range(v):
            out[v_i] = np.diag(ct_inv_xt_iv_x_c[v_i])
    return out


@jit(
    [f8[:, ::1](f8[:, :, ::1], f8[:, ::1]), f4[:, ::1](f4[:, :, ::1], f4[:, ::1])],
    nopython=True,
    parallel=True,
    fastmath=True,
    cache=True,
)
def _batch_diag_numba(ct_inv_xt_iv_x_c, out):
    v, c = out.shape
    for v_i in prange(v):
        out[v_i] = np.diag(ct_inv_xt_iv_x_c[v_i])
    return out


def batch_ct_inv_xt_iv_x_c(inv_xt_iv_x, c, numba=True):
    v, r, _ = inv_xt_iv_x.shape
    num_c = c.shape[1]
    out = np.empty((v, num_c, num_c), dtype=inv_xt_iv_x.dtype)
    if numba:
        c_T = np.ascontiguousarray(c.T)
        out = _batch_ct_inv_xt_iv_x_c_numba(inv_xt_iv_x, c, c_T, out)
    else:
        for v_i in range(inv_xt_iv_x.shape[0]):
            out[v_i] = c.T @ inv_xt_iv_x[v_i] @ c
    return out


@jit(
    [
        f4[:, :, ::1](f4[:, :, :], f4[:, ::1], f4[:, ::1], f4[:, :, ::1]),
        f8[:, :, ::1](f8[:, :, :], f8[:, ::1], f8[:, ::1], f8[:, :, ::1]),
    ],
    nopython=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def _batch_ct_inv_xt_iv_x_c_numba(inv_xt_iv_x, c, c_T, out):
    for v_i in prange(inv_xt_iv_x.shape[0]):
        out[v_i] = c_T @ np.copy(inv_xt_iv_x[v_i]) @ c
    return out


def get_cb_with_derivative_boost(b, c, derivative_boost_masks):
    num_voxel = b.shape[1]
    num_contrast = c.shape[1]

    out = np.zeros((num_voxel, num_contrast), dtype=np.float32)
    for i, c_row, derivative_boost_mask in zip(
        np.arange(len(derivative_boost_masks)), c.T, derivative_boost_masks
    ):
        num_mask, num_relevant_regressor_col = derivative_boost_mask.shape
        derivative_boost_mask_padded = np.zeros((num_mask, len(c_row)), dtype=bool)
        derivative_boost_mask_padded[
            :, :num_relevant_regressor_col
        ] = derivative_boost_mask
        for j in range(derivative_boost_mask.shape[0]):
            relevant_b = b.T[:, derivative_boost_mask_padded[j, :]]
            relevant_c = c_row[derivative_boost_mask_padded[j, :]]
            sign = np.sign(relevant_b[:, 0])
            b_derivative_boost = (
                sign * np.sqrt(np.sum(relevant_b**2, axis=1)) * np.sum(relevant_c)
            )
            out[:, i] += b_derivative_boost
    return out
