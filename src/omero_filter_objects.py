import numpy as np
from omero.gateway import FileAnnotationWrapper

import omero_toolbox as omero_tb
import omero2pandas
from getpass import getpass
from omero.model import MaskI, PointI
import pandas as pd
import logging

HOST = 'omero.mri.cnrs.fr'
PORT = 4064

ch_names = [
    "DAPI",
    "EGFP",
    "DsRed",
    "Cy5"
]

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    try:
        # Open the connection to OMERO
        conn = omero_tb.open_connection(
            username=input("Username: "),
            password=getpass("OMERO Password: ", None),
            host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
            port=int(input('port (4064): ') or PORT),
            group=str(input("Group (DOPAVALUE): ") or "DOPAVALUE")
        )

        # get tagged images in dataset
        dataset_id = int(input('Dataset ID: '))
        dataset = omero_tb.get_dataset(conn, dataset_id)
        project = dataset.getParent()
        images = dataset.listChildren()
        images_names_ids = {i.getName(): i.getId() for i in images}
        image_root_names = list({n[:-4] for n in images_names_ids})

        table_col_names = [
            'image_id',
            'image_name',
            'mouse_id',
            'AP',
            'section_nr',
            'date',
            'magnification',
            'markers',
            'roi_area',
        ]

        for ch_name in ch_names:
            table_col_names.extend([f'roi_intensity_{ch_name}',
                                    f'object_count_{ch_name}',
                                    f'mean_area_{ch_name}',
                                    f'median_area_{ch_name}',
                                    f'sum_area_{ch_name}',
                                    f'sum_intensity_{ch_name}',
                                    f'mean_intensity_{ch_name}',
                                    f'sum_area_bg_{ch_name}',
                                    f'sum_intensity_bg_{ch_name}',
                                    f'mean_intensity_bg_{ch_name}'
                                    ])
        table_col_values = [[] for _ in range(len(table_col_names))]

        roi_service = conn.getRoiService()

        for counter, image_root_name in enumerate(image_root_names):
            logger.info(f'Analyzing image {image_root_name}')
            print(f'Processing image {counter+1}/{len(image_root_names)}: {image_root_name}')

            mip_image = conn.getObject('Image', images_names_ids[f'{image_root_name}_MIP'])
            aip_image = conn.getObject('Image', images_names_ids[f'{image_root_name}_AIP'])
            aip_data = omero_tb.get_intensities(aip_image)

            # Filling data table
            name_md = image_root_name.strip()
            name_md = name_md.replace(' ', '_').split('_')

            table_col_values[0].append(aip_image)  # image_id
            table_col_values[1].append(image_root_name)  # image_name
            table_col_values[2].append(name_md[0])  # mouse_id
            table_col_values[3].append(name_md[1])  # AP
            table_col_values[4].append(name_md[2])  # section_nr
            table_col_values[5].append(name_md[3])  # date
            table_col_values[6].append(name_md[4])  # magnification
            table_col_values[7].append(name_md[5])  # markers

            roi_area = np.count_nonzero(aip_data[0, 0, 0, ...])
            table_col_values[8].append(roi_area)  # 'roi_area'

            response = roi_service.findByImage(mip_image.getId(), None)

            labels_to_keep = []
            for roi in response.rois:
                shape = roi.getPrimaryShape()
                if not isinstance(shape, PointI):
                    continue
                shape_comment = shape.getTextValue()._val
                if not shape_comment:
                    raise ValueError(f"ROI ({roi.getId()}) has no text value")
                parsed_label = int(shape_comment.split('; ')[0].split(':')[-1])
                labels_to_keep.append(parsed_label)

            annotation_links = conn.getAnnotationLinks(parent_type="Image", parent_ids=[aip_image.getId()])
            obj_dataframes = {}
            bg_dataframes = {}
            for annotation_link in annotation_links:
                annotation = annotation_link.getAnnotation()
                if not isinstance(annotation, FileAnnotationWrapper):
                    continue
                if "_object_df" in annotation.getFile().getName():
                    obj_dataframes[int(annotation.getFile().getName().split('_')[0][2:])] = (
                        omero2pandas.read_csv(annotation.getFile().getId(), omero_connector=conn)
                    )
                elif "_bg_df" in annotation.getFile().getName():
                    bg_dataframes[int(annotation.getFile().getName().split('_')[0][2:])] = (
                        omero2pandas.read_csv(annotation.getFile().getId(), omero_connector=conn)
                    )
                else:
                    continue

            # filter the dataframe using the labels_to_keep on the label column
            for ch_nr in obj_dataframes.keys():
                obj_df = obj_dataframes[ch_nr]
                filtered_out_obj_df = obj_df[~obj_df['label'].isin(labels_to_keep)]
                obj_df = obj_df[obj_df['label'].isin(labels_to_keep)]
                bg_df = bg_dataframes[ch_nr]
                bg_df = pd.concat([bg_df, filtered_out_obj_df])
                omero2pandas.upload_table(
                    source=obj_df,
                    table_name=f"ch{ch_nr}_object_df_filtered",
                    parent_id=aip_image.getId(),
                    parent_type="Image",
                    omero_connector=conn,
                )
                omero2pandas.upload_table(
                    source=bg_df,
                    table_name=f"ch{ch_nr}_bg_df_filtered",
                    parent_id=aip_image.getId(),
                    parent_type="Image",
                    omero_connector=conn,
                )

                if len(obj_df) > 0:
                    table_col_values[table_col_names.index(f'roi_intensity_{ch_names[ch_nr]}')].append(np.sum(aip_data[0, ch_nr]).item())
                    table_col_values[table_col_names.index(f'object_count_{ch_names[ch_nr]}')].append(len(obj_df))

                    table_col_values[table_col_names.index(f'mean_area_{ch_names[ch_nr]}')].append(obj_df['area'].mean().item())
                    table_col_values[table_col_names.index(f'median_area_{ch_names[ch_nr]}')].append(obj_df['area'].median().item())
                    table_col_values[table_col_names.index(f'sum_area_{ch_names[ch_nr]}')].append(obj_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_intensity_{ch_names[ch_nr]}')].append(obj_df['integrated_intensity'].sum().item())
                    table_col_values[table_col_names.index(f'mean_intensity_{ch_names[ch_nr]}')].append(obj_df['integrated_intensity'].sum().item() /
                                                                                                               obj_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_area_bg_{ch_names[ch_nr]}')].append(bg_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_intensity_bg_{ch_names[ch_nr]}')].append(bg_df['integrated_intensity'].sum().item())
                    table_col_values[table_col_names.index(f'mean_intensity_bg_{ch_names[ch_nr]}')].append(bg_df['integrated_intensity'].sum().item() /
                                                                                                                  bg_df['area'].sum().item())
                else:
                    logger.warning(f'No objects were detected for image {image_root_name}')

                    table_col_values[table_col_names.index(f'roi_intensity_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'object_count_{ch_names[ch_nr]}')].append(0)

                    table_col_values[table_col_names.index(f'mean_area_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'median_area_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_area_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_intensity_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'mean_intensity_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_area_bg_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_intensity_bg_{ch_names[ch_nr]}')].append(0)
                    table_col_values[table_col_names.index(f'mean_intensity_bg_{ch_names[ch_nr]}')].append(0)

        # Remove empty columns
        table_col_names = [col_name for col_name, col_values in zip(table_col_names, table_col_values) if len(col_values) > 0]
        table_col_values = [col_values for col_values in table_col_values if len(col_values) > 0]

        # Create a table annotation with the results also as a pandas dataframe
        dataframe = pd.DataFrame(table_col_values, index=table_col_names).T

        table = omero_tb.create_annotation_table(
            connection=conn,
            table_name="Aggregated_measurements_filtered",
            column_names=table_col_names,
            column_descriptions=table_col_names,
            values=table_col_values,
        )
        omero_tb.link_annotation(dataset, table)

        # omero2pandas.upload_table(
        #     source=dataframe,
        #     table_name="Aggregated_measurements_filtered_O2P",
        #     parent_id=dataset.getId(),
        #     parent_type="Dataset",
        #     omero_connector=conn,
        # )

    except Exception as e:
        raise e

    finally:
        conn.close()
        logger.info('Done')
