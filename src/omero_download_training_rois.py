import numpy as np
import omero_toolbox as omero
from getpass import getpass
from skimage import draw


# Define variables
OUTPUT_DIR = '/run/media/julio/DATA/DOPAVALUE/training_images'
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
ROI_COMMENTS = ['training']
channel = 0


# Helper functions
def get_tagged_images(dataset, tag):
    images = dataset.listChildren()
    tagged_images = list()
    for image in images:
        for ann in image.listAnnotations():
            if ann.OMERO_TYPE == omero.model.TagAnnotationI and ann.getTextValue() == tag:
                tagged_images.append(image)
                break

    return tagged_images


def run_script():
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=input("Username: "),
                                     password=getpass("OMERO Password: ", None),
                                     host=HOST,
                                     port=PORT,
                                     group=input('Group: '))

        # get tagged images in dataset
        dataset_id = int(input('Dataset ID:'))
        tag_text = str(input('Tag_text ("training_set"):') or 'training_set')

        dataset = conn.getObject('Dataset', dataset_id)

        images = get_tagged_images(dataset, tag_text)


        # Loop through images, get ROIs the intensity values, project and save as .npy
        roi_service = conn.getRoiService()

        counter = 0
        for image in images:
            result = roi_service.findByImage(image.getId(), None)
            for roi in result.rois:
                shape = roi.getPrimaryShape()
                shape_comment = shape.getTextValue()._val
                shape_id = roi.getId()._val
                if shape_comment not in ROI_COMMENTS:
                    continue
                data = omero.get_shape_intensities(image, shape, zero_edge=True)
                # Do a MIP. Data in order zctyx
                data = data.max(axis=0, keepdims=True)
                data = data[:, channel, ...]
                data = np.expand_dims(data, axis=1)

                file_name = f'{OUTPUT_DIR}/{image.getName()}_ROI_label-{shape_comment}_id-{shape_id}.npy'
                np.save(file_name, data)
            counter += 1
            print(f'Processed image {counter} of {len(images)}')
    finally:
        conn.close()
        print('Done')


if __name__ =='__main__':
    run_script()