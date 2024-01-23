import os
import csv
import json
import shutil
import numpy as np
import nibabel as nib


def write_file(file_content, file_path, flag="w+"):
    with open(file_path, flag) as fp:
        fp.write(file_content)


def load_file(file_path, flag="r"):
    with open(file_path, flag) as fp:
        return fp.read()


def delete_file_if_exists(file_name):
    if os.path.exists(file_name):
        os.remove(file_name)


def load_json(json_file_path):
    with open(json_file_path) as fp:
        return json.load(fp)


def write_json(json_data, json_file_path, pretty=True):
    with open(json_file_path, "w+") as fp:
        if pretty:
            fp.write(json.dumps(json_data, indent=2))
        else:
            fp.write(json.dumps(json_data))


def create_dir_if_not_exist(path, full_path=False):
    if not os.path.isdir(path):
        if full_path:
            os.makedirs(path)
        else:
            os.mkdir(path)


def delete_dir_if_exist(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def create_symlink(source, link_name):
    if os.path.islink(link_name):
        delete_file_if_exists(link_name)
    elif os.path.exists(link_name):
        raise ValueError(f"link_name: {link_name} is a file that exists")
    os.symlink(source, link_name)


def load_xsv(xsv_file_path, dialect="excel-tab"):
    # TODO: DictReader returns[{'key1':v1,'key2':v2,...},{'key':v1,...},...]
    # which seems quite very wasteful?
    # Maybe change it to {'key1':[val1,val2,...],'key2':[val1,val2,...],...}
    return [dict(i) for i in list(csv.DictReader(open(xsv_file_path), dialect=dialect))]


def load_xsv_from_raw(raw_file_content, dialect="excel-tab"):
    return [dict(i) for i in list(csv.DictReader(raw_file_content, dialect=dialect))]


def export_dict_list_to_xsv(file_path, dict_list, delimiter="\t"):
    columns = list(dict_list[0].keys())
    delete_file_if_exists(file_path)
    content = (
        delimiter.join(columns)
        + "\n"
        + "\n".join(
            [
                delimiter.join([str(entry[col]) for col in columns])
                for entry in dict_list
            ]
        )
    )
    with open(file_path, "w+") as fp:
        fp.write(content)


def extract_bold_file_info_from_name(file_path):
    path_segments = file_path.split("/")
    func_path = "/".join(path_segments[0:-1])
    bold_file_name = path_segments[-1]
    bold_file_name_segments = bold_file_name.split("_")
    sub = [
        str_segment
        for str_segment in bold_file_name_segments
        if str_segment.startswith("sub")
    ][0].split("-")[-1]
    run = [
        str_segment
        for str_segment in bold_file_name_segments
        if str_segment.startswith("run")
    ][0].split("-")[-1]
    task = [
        str_segment
        for str_segment in bold_file_name_segments
        if str_segment.startswith("task")
    ][0].split("-")[-1]
    space = [
        str_segment
        for str_segment in bold_file_name_segments
        if str_segment.startswith("space")
    ][0].split("-")[-1]
    brain_mask_file_name = f'{bold_file_name.split("_desc-")[0]}_desc-brain_mask.nii.gz'

    ses = file_path.split("ses-")[1].split("/")[0]
    timeseries_file = f"sub-{sub}_task-{task}_run-{run}_desc-confounds_timeseries"
    derivatives_path = f'{file_path.split("derivatives")[0]}derivatives'
    if not os.path.exists(f"{func_path}/{timeseries_file}.tsv"):
        timeseries_file = (
            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-confounds_timeseries"
        )
    info = {
        "sub": sub,
        "run": run,
        "ses": ses,
        "task": task,
        "space": space,
        "func_path": func_path,
        "bold_file_name": bold_file_name,
        "timeseries_json_name": f"{timeseries_file}.json",
        "timeseries_tsv_name": f"{timeseries_file}.tsv",
        "brain_mask_file_name": brain_mask_file_name,
        "brain_mask_file_path": f"{func_path}/{brain_mask_file_name}",
        "derivatives_path": derivatives_path,
    }
    info["timeseries_json_path"] = f'{info["func_path"]}/{info["timeseries_json_name"]}'
    info["timeseries_tsv_path"] = f'{info["func_path"]}/{info["timeseries_tsv_name"]}'
    info["bold_file_path"] = f'{info["func_path"]}/{info["bold_file_name"]}'
    return info


MOTION_CONFOUND_LIST = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
MOTION_DERIVATIVE1_CONFOUND_LIST = [
    "trans_x_derivative1",
    "trans_y_derivative1",
    "trans_z_derivative1",
    "rot_x_derivative1",
    "rot_y_derivative1",
    "rot_z_derivative1",
]
MOTION_POW2_CONFOUND_LIST = [
    "trans_x_power2",
    "trans_y_power2",
    "trans_z_power2",
    "rot_x_power2",
    "rot_y_power2",
    "rot_z_power2",
]
MOTION_DERIVATIVE1_POW2_CONFOUND_LIST = [
    "trans_x_derivative1_power2",
    "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1_power2",
    "rot_y_derivative1_power2",
    "rot_z_derivative1_power2",
]
CSF_CONFOUND_LIST = ["csf", "csf_derivative1", "csf_derivative1_power2", "csf_power2"]
WM_CONFOUND_LIST = [
    "white_matter",
    "white_matter_derivative1",
    "white_matter_derivative1_power2",
    "white_matter_power2",
]


def extract_confound_regressors(
    confound_dict_list,
    USE_MOTION=True,
    USE_MOTION_DERIVATIVE1=True,
    USE_MOTION_POW2=True,
    USE_MOTION_DERIVATIVE1_POW2=True,
    USE_CSF=True,
    USE_WM=False,
    USE_MOTION_OUTLIER=True,
    dtype=np.float64,
):
    total_confound_list = []
    if USE_MOTION:
        total_confound_list += MOTION_CONFOUND_LIST
    if USE_MOTION_DERIVATIVE1:
        total_confound_list += MOTION_DERIVATIVE1_CONFOUND_LIST
    if USE_MOTION_POW2:
        total_confound_list += MOTION_POW2_CONFOUND_LIST
    if USE_MOTION_DERIVATIVE1_POW2:
        total_confound_list += MOTION_DERIVATIVE1_POW2_CONFOUND_LIST
    if USE_CSF:
        CSF_regressor_indices = len(total_confound_list) + np.arange(
            len(CSF_CONFOUND_LIST)
        )
        total_confound_list += CSF_CONFOUND_LIST
    if USE_WM:
        WM_regressor_indices = len(total_confound_list) + np.arange(
            len(WM_CONFOUND_LIST)
        )
        total_confound_list += WM_CONFOUND_LIST
    if USE_MOTION_OUTLIER:
        total_confound_list += [
            confound_key
            for confound_key in list(confound_dict_list[0].keys())
            if confound_key.startswith("motion_outlier")
        ]
    confound_regressors = np.array(
        [
            [
                float(row[confound]) if row[confound] != "n/a" else float("nan")
                for confound in total_confound_list
            ]
            for row in confound_dict_list
        ],
        dtype=dtype,
    )
    derivative1_confound_mask = np.array(
        ["derivative1" in confound for confound in total_confound_list], dtype=bool
    )
    confound_regressors[0, derivative1_confound_mask] = confound_regressors[
        1, derivative1_confound_mask
    ]
    if USE_CSF:
        confound_regressors[:, CSF_regressor_indices] = confound_regressors[
            :, CSF_regressor_indices
        ] / np.max(confound_regressors[:, CSF_regressor_indices], axis=0)
    if USE_WM:
        confound_regressors[:, WM_regressor_indices] = confound_regressors[
            :, WM_regressor_indices
        ] / np.max(confound_regressors[:, WM_regressor_indices], axis=0)

    return confound_regressors.astype(dtype)


def get_LETTERS_TO_DOT_from_processed_info(processed_bold_info):
    events = processed_bold_info["event_tsv_content"]
    regressor_types = processed_bold_info["regressor_types"]
    letter_labels = [
        event["letter"]
        for i, event in enumerate(events)
        if (i > 0 and (event["onset"] != events[i - 1]["onset"])) or i == 0
    ]
    letter_labels = [letter if letter != "_" else " " for letter in letter_labels]
    # transition_letter_labels = [
    #     f'{l}{letter_labels[i + 1]}'
    #     for i, l in enumerate(letter_labels[:-1])
    # ]
    letters = np.unique(letter_labels)
    LETTERS_TO_DOT = {
        letter: {regressor: 0 for regressor in regressor_types} for letter in letters
    }
    letter_time_dict = {}
    for e in events:
        letter = " " if e["letter"] == "_" else e["letter"]
        t = int(float(e["onset"]))
        if letter not in letter_time_dict:
            letter_time_dict[letter] = t
        if (letter_time_dict[letter] == t) and (e["trial_type"] in regressor_types):
            LETTERS_TO_DOT[letter][e["trial_type"]] = 1
    return LETTERS_TO_DOT


def load_bold_image(bold_image_path):
    bold_image_handle = nib.load(bold_image_path)
    return bold_image_handle.get_fdata(dtype=np.float64)
