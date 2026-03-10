import numpy as np
import omero_toolbox as omero
from getpass import getpass
import sys
from skimage import draw


# Define variables
USER = input("username: ") if len(sys.argv) < 2 else sys.argv[1]
PASS = getpass("password: ") if len(sys.argv) < 3 else sys.argv[2]
GROUP = input("group: ") if len(sys.argv) < 4 else sys.argv[3]
DATASET = input("dataset ID: ") if len(sys.argv) < 5 else sys.argv[4]
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
# ROI_COMMENTS = ("core-i", "core-ni", "Lsh-i", "Lsh-ni", "Msh-i", "Msh-ni",)
ROI_COMMENTS = ("ibla", "ila", "nibla", "nila")
# ROI_COMMENTS = None
C_RANGE = None


try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=USER,
                                 password=PASS,
                                 host=HOST,
                                 port=PORT,
                                 group=GROUP)

    dataset_id = int(DATASET)

    roi_filter = ROI_COMMENTS

    dataset = omero.get_dataset(conn, dataset_id)

    project = dataset.getParent()

    new_dataset_name = f'{dataset.getName()}_rois'
    new_dataset_description = f'Source Dataset ID: {dataset.getId()}'
    new_dataset = omero.create_dataset(conn,
                                       name=new_dataset_name,
                                       description=new_dataset_description,
                                       parent_project=project)

    images = dataset.listChildren()

    # Loop through images, get ROIs the intensity values, project and save as .npy
    roi_service = conn.getRoiService()

    for counter, image in enumerate(images):
        result = roi_service.findByImage(image.getId(), None)
        for roi in result.rois:
            shape = roi.getPrimaryShape()
            try:
                shape_comment = shape.getTextValue()._val
            except AttributeError:
                shape_comment = None
            if roi_filter is not None and shape_comment not in roi_filter:
                continue
            data = omero.get_shape_intensities(image, shape, c_range=C_RANGE, zero_edge=True, zero_value="zero")
            mip_data = data.max(axis=0, keepdims=True)
            aip_data = data.mean(axis=0, keepdims=True)
            aip_data = aip_data.astype(data.dtype)

            new_image_name = f'{image.getName().strip()}_{shape_comment}'
            omero.create_image_from_numpy_array(connection=conn,
                                                data=mip_data,
                                                image_name=f'{new_image_name}_MIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId(),
                                                force_whole_planes=False)

            omero.create_image_from_numpy_array(connection=conn,
                                                data=aip_data,
                                                image_name=f'{new_image_name}_AIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId(),
                                                force_whole_planes=False)

        print(f'Processed image {counter}')

finally:
    conn.close()
    print('Done')
