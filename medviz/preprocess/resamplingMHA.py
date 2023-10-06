import numpy as np
import SimpleITK as sitk
from ..utils import path_in, save_path_file

def resampleMHA(input_path, output_path, new_voxel_size, method):
    input_path = path_in(input_path)

    # Load the input MHA binary image
    r = sitk.ImageFileReader()
    r.SetFileName(input_path)
    img = r.Execute()

    # Find resolution and size with SITK
    current_voxel_size = img.GetSpacing()
    current_size = img.GetSize()

    print("Current voxel size", current_voxel_size)
    current_voxel_size = np.array(current_voxel_size)
    scaling_factors = current_voxel_size / new_voxel_size
    print("Scaling factors", scaling_factors)

    # Compute the new shape after resampling
    print("Input shape", current_size)
    new_shape = np.ceil(current_size * scaling_factors).astype(int)
    print("Output shape", new_shape)

    # Initialize resampling image filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_voxel_size)
    resample.SetSize(new_shape)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img.GetPixelIDValue())

    # Set correct interpolation method
    # SimpleITK has lots of interpolation methods built into the system, can add more if you prefer
    if method == "nearest":
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    elif method == "trilinear":
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        print("Only methods are nearest or trilinear, defaulting to nearest - please fix!")
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Use the filter to resample automatically
    resampled_img = resample.Execute(img)

    #  Save the resampled image to the output path
    save_path = save_path_file(output_path, suffix=".mha")
    w = sitk.ImageFileWriter()
    w.SetFileName(save_path)
    w.Execute(resampled_img)
