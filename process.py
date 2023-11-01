import SimpleITK
import glob
import numpy as np
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

import SimpleITK as sitk
import time
import os
from skimage import measure, filters
import subprocess
import shutil
from pathlib import Path


# from nnunet.inference.predict import predict_from_folder
# from predict import predict_from_folder
# # from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
# from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch

os.environ["nnUNet_raw_data_base"] = "nnUNet_raw_data_base/"
os.environ["RESULTS_FOLDER"] = "nnUNet_trained_models/"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed/"
os.environ["MKL_THREADING_LAYER"] = "GNU"

network_training_output_dir = "nnUNet_trained_models/"


class Autopet(SegmentationAlgorithm):
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # set some paths and parameters
        self.input_path = (
            "/input"  # according to the specified grand-challenge interfaces
        )
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"  # according to the specified grand-challenge interfaces
        self.nii_path = "nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs/"
        self.result_path = "Task001_TCIA/"
        self.nii_seg_file = "TCIA_001.nii.gz"

        # make directories
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.nii_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_path).mkdir(parents=True, exist_ok=True)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        self.ref_img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(self.ref_img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = sitk.ReadImage(nii_input_path)
        img.CopyInformation(self.ref_img)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def dice_coef(self, y_true, y_pred, smooth=1):

        y_true_f = np.ndarray.flatten(y_true)
        y_pred_f = np.ndarray.flatten(y_pred)
        intersection = np.sum(y_true_f * y_pred_f)
        dice_coef_ = (2.0 * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) + smooth
        )
        return dice_coef_

    def no_lesion(self, ensemble_2d):
        ensemble_2d[ensemble_2d < 0.6] = 0
        ensemble_2d[ensemble_2d >= 0.6] = 1
        return ensemble_2d

    def lesion(self, ensemble_lesion):
        # ensemble_lesion = (ensemble_2d + ensemble_3d + ensemble_residual)/3
        ensemble_lesion[ensemble_lesion >= 0.6] = 1
        ensemble_lesion[ensemble_lesion < 0.6] = 0
        return ensemble_lesion

    def adaptive_ensemble(self, output_list, final_ensemble):
        adaptive = []
        for index, i in enumerate(output_list):
            temp = i.copy()
            temp[temp >= 0.5] = 1
            temp[temp < 0.5] = 0
            dice_score = self.dice_coef(final_ensemble, temp)
            if dice_score >= 0.9:
                adaptive.append(i)
        ensemble = sum(adaptive) / len(adaptive)
        ensemble[ensemble >= 0.5] = 1
        ensemble[ensemble < 0.5] = 0
        return ensemble
    def n_connected(self, img_data, label):
        img_data_mask = np.zeros(img_data.shape)
        img_data_mask[img_data == label] = 1

        img_filtered = np.zeros(img_data_mask.shape)
        blobs_labels = measure.label(img_data_mask, background=0)
        lbl, counts = np.unique(blobs_labels, return_counts=True)
        lbl_dict = {}
        for i, j in zip(lbl, counts):
            lbl_dict[i] = j
        sorted_dict = dict(sorted(lbl_dict.items(), key=lambda x: x[1], reverse=True))
        count = 0

        for key, value in sorted_dict.items():
            if value < 25:
                img_filtered[blobs_labels == key] = 1
            count += 1

        img_data_mask[img_filtered == 1] = 0
        return img_filtered
    def bbox2_3D(self, img):
        r = np.any(img, axis=(1, 2))
        c = np.any(img, axis=(0, 2))
        z = np.any(img, axis=(0, 1))

        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]

        return rmin, rmax, cmin, cmax, zmin, zmax


    def mask_labels(self, labels, ts):
        """
        Create a mask based on given labels.

        Args:
            labels (list): List of labels to be masked.
            ts (np.ndarray): Image data.

        Returns:
            np.ndarray: Masked image data.
        """
        lung = np.zeros(ts.shape)
        for lbl in labels:
            lung[ts == lbl] = 1
        return lung    
    def ensemble(self):
        for fold in range(5):
            img = sitk.ReadImage(os.path.join(self.output_path,str(fold), self.nii_seg_file))
            img = img>8
            img_data = sitk.GetArrayFromImage(img)
            # print(np.unique(img))
            brain = self.mask_labels([6], np.copy(img_data))
            if brain.max() != 0:
                x1, x2, y1, y2, z1, z2 = self.bbox2_3D(brain)
                img_data[x2:, :, :] = 0
            img_data[:5, :5, :5] = 0
            img_data[-5:, -5:, -5:] = 0
            if fold ==0:
                seg_data = np.copy(img_data)
            else:
                seg_data += img_data            
        seg_data = seg_data/5.
        seg_data[seg_data<0.6]=0
        seg_data[seg_data>=0.6]=1
        values,counts = np.unique(seg_data,return_counts=True)
        print(values)
        if counts[-1]<25:
            seg_data[seg_data>0]=0
        img_th = sitk.GetImageFromArray(seg_data)
        img_th.CopyInformation(img)
        sitk.WriteImage(
            img_th, os.path.join(self.output_path, self.nii_seg_file), True
        )

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.output_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        for fold in range(5):
            os.system(
                f"nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs/ -o /output/images/automated-petct-lesion-segmentation/{fold} -t Task001_TCIA -tr nnUNetTrainerV2 -m 3d_fullres -p nnUNetPlansv2.1 --overwrite_existing -f {fold}"
            )
        print(f"Prediction finished")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.ensemble()
        self.write_outputs(uuid)


if __name__ == "__main__":
    Autopet().process()
    # docker save autopet | xz -T0 -c > autopet2023.tar.xz