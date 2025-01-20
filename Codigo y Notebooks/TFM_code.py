import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import cv2
from PIL import Image
from ultralytics import YOLO
from sklearn.model_selection import train_test_split

############################ FUNCIONES DE PREPROCESADO DE IMÁGENES ############################

def convert_yolo_labels_to_standard(
    img_tensor: np.ndarray,
    yolo_coordinates: list,
    class_id_is_string=False
):
    """
    Convert YOLO formatted label coordinates to standard bounding box coordinates.

    Parameters:
    ----------
    img_tensor: np.ndarray
        The image as a numpy array.
    yolo_coordinates: list
        A list containing YOLO formatted coordinates [class_id, center_x, center_y, width, height].

    Returns:
    ----------
    tuple: A tuple containing (class_id, x_min, y_min, x_max, y_max) where:
        class_id (int): The class ID of the object.
        x_min (int): The x-coordinate of the top-left corner of the bounding box.
        y_min (int): The y-coordinate of the top-left corner of the bounding box.
        x_max (int): The x-coordinate of the bottom-right corner of the bounding box.
        y_max (int): The y-coordinate of the bottom-right corner of the bounding box.
    """
    image_height, image_width, _ = img_tensor.shape

    # DEBUG
    # print('class index:', yolo_coordinates[0])
    
    # Parse the YOLO coordinates.
    if class_id_is_string is False:
        class_id = int(yolo_coordinates[0])
    else:
        class_id = yolo_coordinates[0]

    center_x = float(yolo_coordinates[1])
    center_y = float(yolo_coordinates[2])
    width    = float(yolo_coordinates[3])
    height   = float(yolo_coordinates[4])

    # Convert to absolute coordinates.
    box_center_x = center_x * image_width
    box_center_y = center_y * image_height
    box_width    = width    * image_width
    box_height   = height   * image_height
    
    x_min = int(box_center_x - (box_width/2))
    y_min = int(box_center_y - (box_height/2))
    x_max = int(box_center_x + (box_width/2))
    y_max = int(box_center_y + (box_height/2))

    return (class_id, x_min, y_min, x_max, y_max)

#####################################################################################

def display_annotated_image(
    img_fname,
    class_list,
    img_dir,
    label_dir
):
    """
    Displays an image with bounding boxes and class labels annotated from YOLO format label files.

    Parameters:
    ----------
    img_fname: str
        The filename of the image (including the extension) to be annotated and displayed.
    class_list: list
        List of class names where each index corresponds to the class ID used in the YOLO label files.
    img_dir: str
        Directory containing the image files (.jpg format) to be annotated.
    label_dir: str
        Directory containing the YOLO label files (.txt format) that provide bounding box coordinates 
        and class IDs for each object in the image.
    """
    img_name = '.'.join(img_fname.split('.')[:-1])
    img_path = os.path.join(img_dir, f'{img_name}.jpg')
    label_path = os.path.join(label_dir, f'{img_name}.txt')
    
    img_tensor = cv2.imread(img_path)
    with open(label_path, mode='r') as f:
        yolo_coords = f.read()
        
    # a txt file might contain more than 1 object coords.
    yolo_coords_lists = yolo_coords.splitlines()
    for yolo_coords_list in yolo_coords_lists:
        class_id, x_min, y_min, x_max, y_max = convert_yolo_labels_to_standard(
             img_tensor=img_tensor,
             yolo_coordinates=yolo_coords_list.split(' ')
        )
        
        # Add bounding box in img.
        img_labeled = cv2.rectangle(
            img=img_tensor,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=(0, 0, 255),  # Note in cv2: BGR (Red)
            thickness=1
        )
        
        text = f"{class_list[class_id]} ({class_id})"
        
        # Add class text in img.
        text_x = x_min + 5  # Positioning the text slightly to the right wrt bounding box (0,0).
        text_y = y_min - 5  # Positioning the text slightly to the top.
        img_labeled = cv2.putText(
            img = img_labeled,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.4,
            color=(255, 255, 255),  # (White)
            thickness=1  
        )
    
    print(img_name)
    plt.imshow(img_labeled[..., ::-1])
    plt.axis('off')
    plt.show()

#####################################################################################

def copy_img_and_label_to_dirs(
    img_name,
    img_dir,
    new_img_dir,
    label_dir,
    new_label_dir,
):
    img_fname = f'{img_name}.jpg'
    shutil.copyfile(
        src=os.path.join(img_dir, img_fname),
        dst=os.path.join(new_img_dir, img_fname)
    )
    
    label_fname = f'{img_name}.txt'
    shutil.copyfile(
        src=os.path.join(label_dir, label_fname),
        dst=os.path.join(new_label_dir, label_fname)
    )

#####################################################################################

def recategorize_labels_txt(
    label_dir,
    old_class_list,
    new_class_list,
    old_to_new_class_dict
):
    """
    Recategorizes YOLO label files by changing the class IDs according to a provided mapping from old to new classes.

    Parameters:
    ----------
    label_dir: str
        Directory containing the YOLO label files (.txt format) to be recategorized.
    old_class_list: list
        List of the original class names corresponding to the class IDs in the YOLO label files.
    new_class_list: list
        List of the new class names to which the old class names will be mapped.
    old_to_new_class_dict: dict
        Dictionary that maps old class names (keys) to new class names (values), allowing the recategorization of labels.
    """
    def replace_old_to_new_class_txt(
        yolo_coords,
        old_class_list,
        new_class_list,
        old_to_new_class_dict
    ):
        old_clase = old_class_list[int(yolo_coords.split(' ')[0])]
        new_clase = old_to_new_class_dict[old_clase]
        new_clase_id = str(new_class_list.index(new_clase))
        
        new_text = re.sub(
            pattern='^\d+',
            repl=new_clase_id,
            string=yolo_coords
        )
        return new_text

    print(len(os.listdir(label_dir)), 'labels txt to recategorize...')
    for label_fname in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_fname)
        with open(label_path, mode='r') as f:
            yolo_coords = f.read()

        # a txt file might contain more than 1 object coords.
        yolo_coords_lists = yolo_coords.splitlines()
        for i, yolo_coord in enumerate(yolo_coords_lists):
            # change class_id in yolo coord line in txt.
            new_yolo_coord = replace_old_to_new_class_txt(
                yolo_coord,
                old_class_list,
                new_class_list,
                old_to_new_class_dict
            )

            if i == 0:
                with open(label_path, mode='w') as file:
                    file.write(new_yolo_coord)
            else:
                with open(label_path, mode='a') as file:
                    file.write(f'\n{new_yolo_coord}')
    print('Done!')

#####################################################################################

def delete_class_labels(
    label_dir,
    class_list: list,
    class_to_delete: list
):
    """
    Deletes specific class annotations from YOLO label files based on a given list of classes to remove.

    Parameters:
    ----------
    label_dir: str
        Directory containing the YOLO label files (.txt format) to be processed.
    class_list: list
        List of all class names corresponding to the class IDs in the YOLO label files.
    class_to_delete: list
        List of class names to be removed from the label files.
    """
    print('Total labels txt:', len(os.listdir(label_dir)))
    txts_affected = set()
    anno_affected = 0
    for label_fname in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_fname)
        with open(label_path, mode='r') as f:
            yolo_coords = f.read()
    
        # a txt file might contain more than 1 object coords.
        new_yolo_coords = []
        for i, yolo_coord in enumerate(yolo_coords.splitlines()):
            class_id = int(yolo_coord.split(' ')[0])
            clase = class_list[class_id]
            if clase in class_to_delete:
                txts_affected.add(label_path)
                anno_affected += 1
            else:
                new_yolo_coords.append(yolo_coord)

        # Generate whole label string and strip empty lines.
        new_yolo_coords_text = '\n'.join(new_yolo_coords).strip()

        # Overwrite label file with new label annotations.
        with open(label_path, mode='w') as file:
            file.write(new_yolo_coords_text)

    print('Total label txts affected:', len(txts_affected))
    print('Total annotations eliminated:', anno_affected)


#####################################################################################

def count_n_classes_(
    labels_dir,
    class_list
):
    """
    Counts the occurrences of each class in YOLO label files and calculates their percentage contribution.

    Parameters:
    ----------
    labels_dir: str
        Directory containing the YOLO label files (.txt format) to be analyzed.
    class_list: list
        List of class names corresponding to the class IDs in the YOLO label files.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the count and percentage of occurrences for each class. The DataFrame includes:
        - 'count': Total number of instances of each class found across all label files.
        - 'perc': Percentage of each class in relation to the total number of class instances.
        A row with the label 'total' contains the sum of all class counts and the total percentage (which sums to 100%).
    """
    class_count_dict = {clase:0 for clase in class_list}
    total_count = 0
    for label_fname in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_fname)
        with open(label_path, mode='r') as f:
            yolo_coords = f.read()

        # print('\n', label_fname)
        # a txt file might contain more than 1 object coords.
        yolo_coords_lists = yolo_coords.splitlines()
        for yolo_coords_list in yolo_coords_lists:
            clase = class_list[int(yolo_coords_list.split(' ')[0])]
            # print('class_id:', yolo_coords_list.split(' ')[0], '->', 'clase:', clase)
            class_count_dict[clase] += 1
            total_count += 1

    df = pd.DataFrame(columns=['count', 'perc'])
    for key in class_count_dict:
        count = class_count_dict[key]
        perc  = np.round(count / total_count * 100, 2)
        df.loc[key] = [int(count), perc]
    df.loc['total'] = [df['count'].sum(), df['perc'].sum()]

    return df

#####################################################################################

def get_class_count_per_image_df(
    label_dir,
    class_list,
    exclude_background_imgs=True
): 
    """
    Generates a DataFrame with the count of each class and their percentage per image, based on YOLO label files.

    Parameters:
    ----------
    label_dir: str
        Directory containing YOLO label files (.txt format) to analyze.
    class_list: list
        List of class names corresponding to the class IDs in the YOLO label files.
    exclude_background_imgs: bool, optional
        Whether to exclude images with no class annotations (background images) from the output. 
        Defaults to True.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to an image, containing:
        - 'name': Name of the image (without extension).
        - Counts of each class for that image.
        - 'total': Total number of class annotations in the image.
        - Percentage columns for each class representing the proportion of that class in relation to the total. 
    """
    data = []
    for label_fname in os.listdir(label_dir):
        # DEBUG
        # print(label_fname)
        
        # set up dict with individual img class count info.
        name = '.'.join(label_fname.split('.')[:-1])
        class_count_dict = {'name': name, 'total': 0}
        class_count_dict.update({clase:0 for clase in class_list})
        
        label_path = os.path.join(label_dir, label_fname)
        with open(label_path, mode='r') as f:
            yolo_coords = f.read()
            
        # a txt file might contain more than 1 object coords.
        yolo_coords_lists = yolo_coords.splitlines()
        for yolo_coords_list in yolo_coords_lists:
            # DEBUG
            # print(int(yolo_coords_list.split(' ')[0]))
            clase = class_list[int(yolo_coords_list.split(' ')[0])]
            # print(clase)
            # update class count in dict.
            class_count_dict[clase]   += 1
            class_count_dict['total'] += 1

        # add individual class count dict to data (list).
        data.append(class_count_dict)

    # Create df.
    df = pd.DataFrame(data)
    # Create perc prop for each category.
    for clase in class_list:
        df[f"{clase}_perc"] = np.round(df[clase] / df['total'] * 100, 2)

    # Exclude background imgs (total = 0).
    if exclude_background_imgs is True:
        df = df.loc[df.total > 0]

    return df

####################################################################################

def get_class_count_abs_perc_df(
    class_count_dict
):
    """
    Generates a DataFrame containing the absolute counts and percentage distribution of class instances.

    Parameters:
    ----------
    class_count_dict: dict
        Dictionary where the keys represent class names and the values represent the count of instances for each class.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with two columns:
        - 'count': Absolute count of instances for each class.
        - 'perc': Percentage of each class relative to the total count.
        The DataFrame includes a 'total' row, which sums up the counts and percentages for all classes.
    """
    df = pd.DataFrame(columns=['count', 'perc'])
    total_count = sum([v for k, v in class_count_dict.items()])
    
    for key in class_count_dict:
        count = class_count_dict[key]
        perc  = np.round(count / total_count * 100, 2)
        df.loc[key] = [int(count), perc]
        
    df.loc['total'] = [df['count'].sum(), df['perc'].sum()]
    
    return df

####################################################################################

def simulate_undersampling_yolo(
    df_img_class_count,
    class_list,
    target_class,
    n_instances,
    perc_threshold=0.0,
    only_int_imgs=False,
    show_bef_aft=True,
):
    """
    Simulates undersampling of images in a YOLO dataset to reduce the number of instances of a target class.

    Parameters:
    ----------
    df_img_class_count: pd.DataFrame
        DataFrame containing image-level class counts and percentages, where each row corresponds to an image.
    class_list: list
        List of all class names corresponding to the class IDs in the YOLO label files.
    target_class: str
        Class name to be undersampled.
    n_instances: int
        Target number of instances for the target class after undersampling.
    perc_threshold: float, optional
        Minimum percentage of the target class in an image to be considered for undersampling. Defaults to 0.0.
    only_int_imgs: bool, optional
        If True, only images with numeric names (TACO images) are considered. Defaults to False.
    show_bef_aft: bool, optional
        If True, displays the class counts and percentages before and after undersampling. Defaults to True.

    Returns:
    --------
    pd.DataFrame
        A subset of the original DataFrame, containing images such that the number of instances of the target class 
        is reduced to the specified `n_instances`.
    """
    df = df_img_class_count.copy()

    # Exclude background images.
    df = df.loc[df.total > 0]
    
    # get class count dict from orig df.
    class_count_dict_bef = df.sum()[class_list].to_dict()
    n_imgs_bef = len(df)

    # Df filters:
    ## TACO images (only digit name).
    if only_int_imgs is True:
        taco_imgs_mask = df.name.str.isdigit()
    else:
        taco_imgs_mask = pd.Series([True] * len(df))

    df = df.loc[taco_imgs_mask]

    ## filter by target_class & perc_threshold.
    df = df.loc[df[f"{target_class}_perc"] >= perc_threshold]

    # shuffle df.
    df = df.sample(frac=1).reset_index(drop=True)
    
    # get cumsum col for target class.
    df[f"{target_class}_cumsum"] = df[target_class].cumsum()
    # get 1st img that adds up to n_instances in target_class (first row is the smallest).
    n_instances_index = df.loc[df[f"{target_class}_cumsum"] >= n_instances].index[0]
    # filter df up to n_instances_index.
    df = df.loc[:n_instances_index]
    
    # get class count dict from subset df.
    class_count_dict_subset = df.sum()[class_list].to_dict()
    n_imgs_subset = len(df)

    # get class count dict of orig dict - subset dict.
    class_count_dict_aft = {k:class_count_dict_bef[k] - v for k, v in class_count_dict_subset.items()}
    n_imgs_aft = n_imgs_bef - n_imgs_subset 
    
    if show_bef_aft is True:
        df_count_bef = get_class_count_abs_perc_df(class_count_dict_bef)
        df_count_sub = get_class_count_abs_perc_df(class_count_dict_subset)
        df_count_aft = get_class_count_abs_perc_df(class_count_dict_aft)

        # display counts.
        print(f"Before ({n_imgs_bef}):")
        print(df_count_bef, '\n')
        print(f"Subset ({n_imgs_subset}):")
        print(df_count_sub, '\n')
        print(f"After ({n_imgs_aft}):")
        print(df_count_aft, '\n')
    
    return df

#####################################################################################

def get_resized_imgs(
    img_dir,
    save_dir,
    dsize=(700, 700),
    shape_thresh=(2500, 2500),  # (height, width)
    size_thresh_bytes=1_000_000,
    num_tracking=100
):
    """
    Resizes images larger than a specified size or resolution and saves them to a directory. Smaller images are copied 
    directly. Displays statistics on the resizing process.

    Parameters:
    ----------
    img_dir: str
        Directory containing the source images.
    save_dir: str
        Directory where the resized or copied images will be saved.
    dsize: tuple, optional
        Target dimensions (width, height) to resize images to. Defaults to (700, 700).
    shape_thresh: tuple, optional
        Threshold dimensions (height, width). Images larger than this will be resized. Defaults to (2500, 2500).
    size_thresh_bytes: int, optional
        File size threshold in bytes. Images larger than this size will be resized. Defaults to 1,000,000 bytes (1MB).
    num_tracking: int, optional
        Number of images processed before printing progress. Defaults to 100.

    Returns:
    --------
    None
        The function saves the resized or copied images to `save_dir` and prints a summary of the operation, including 
        the total number of resized images, total sizes of the source and save directories, and the percentage reduction 
        in size after resizing.
    """
    src_dir_bytes = 0
    save_dir_bytes = 0
    resize_counter = 0
    print(f'Resizing images larger than {size_thresh_bytes/1_000_000}MB or witdth > {shape_thresh[0]} or height > {shape_thresh[1]} to {dsize}...\n')
    for i, img_file in enumerate(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img_file)
        img_size = os.path.getsize(img_path)
        src_dir_bytes += img_size

        img_tensor = cv2.imread(img_path)
        if img_tensor is None:
            print(f"Error reading image as tensor: {img_path}")
            continue

        height, width, _ = img_tensor.shape
        
        # check file size.
        if (img_size >= size_thresh_bytes) or ((height > shape_thresh[0]) or (width > shape_thresh[1])):
            # If equal or larger than 1MB, then resize it.
            img_resized = cv2.resize(
                src=img_tensor,
                dsize=dsize,
                interpolation=cv2.INTER_LINEAR
            )
            
            # Save resized img.
            img_file_jpg = f"{img_file.split('.')[0]}.jpg"
            save_img_path = os.path.join(save_dir, img_file_jpg)
            cv2.imwrite(save_img_path, img_resized)

            # Increase values of counters.
            resize_counter += 1
            save_dir_bytes += os.path.getsize(save_img_path)
        else:
            # Copy original img from img_dir to save_dir.
            shutil.copyfile(
                src=img_path,
                dst=os.path.join(save_dir, img_file)
            )
            save_dir_bytes += img_size
 
        # Tracking.
        if (i + 1) % num_tracking == 0:
            print(f"{i + 1} images processed...")
    
    # Info.
    total_images = len(os.listdir(img_dir))
    src_dir_GB = src_dir_bytes / (1024**3)
    src_dir_MB = src_dir_bytes / (1024**2)
    save_dir_GB = save_dir_bytes / (1024**3)
    save_dir_MB = save_dir_bytes / (1024**2)
    print(f"""
-Total images resized: {resize_counter} out of {total_images} ({resize_counter/total_images*100:.2f}%).

-Total size of  src directory "{img_dir}":
    - {src_dir_GB:.2f} GB ({src_dir_MB:.2f} MG).
-Total size of save directory "{save_dir}":
    - {save_dir_GB:.2f} GB ({save_dir_MB:.2f} MG).
-Total reduction in size after resizing:
    - {(src_dir_MB - save_dir_MB)/src_dir_MB*100:.2f}%!
    """)       
    
############################ FUNCIONES DE EVALUACIÓN ############################

def extract_best_models(
    runs_dir,
    save_dir,
    models_included
):
    """
    Extracts the best YOLO model weights ('best.pt') from multiple model training runs and saves them to a specified directory.

    Parameters:
    ----------
    runs_dir: str
        Directory containing the YOLO training runs, where each model's weights are saved.
    save_dir: str
        Directory where the extracted 'best.pt' files will be saved.
    models_included: list
        List of models that have already been processed and should be excluded from extraction.

    Returns:
    --------
    None
        The function prints the list of newly included models and copies their 'best.pt' files to the `save_dir`. 
        If no new models are found or any errors occur during the file extraction, relevant messages are printed.
    """
    model_names = [model_name for model_name in os.listdir(runs_dir) if model_name not in models_included]
    
    if len(model_names) == 0:
        print('No new models added to YOLO_runs!')
    
    if len(model_names) > 0:
        print(f"New models included in YOLO_runs:\n{model_names}")
        print(f"\nExtracting best.pt files to {save_dir}...")
        for model_name in model_names:
            best_model_path = os.path.join(runs_dir, model_name, 'detect', 'train', 'weights', 'best.pt')
            new_model_path  = os.path.join(save_dir, f"{model_name}.pt")
            # copy file to new dir with changed name.
            shutil.copy(
                src=best_model_path,
                dst=new_model_path
            )
            # verify the file actually exists.
            if not os.path.exists(new_model_path):
                print('Error extracting best.pt files to:', new_model_path)
        print('\nDone!')

############################################################################################################

def show_different_args(
    runs_dir,
    model_1,
    model_2
):
    """
    Compares the training arguments of two YOLO models and returns the differences.

    Parameters:
    ----------
    runs_dir: str
        Directory containing the YOLO model runs.
    model_1: str
        Name of the first model to compare.
    model_2: str
        Name of the second model to compare.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing only the arguments where the two models differ. 
    """
    # Get parameter names to add to a df as columns.
    yaml_path = os.path.join(runs_dir, [model_1, model_2][0], 'detect', 'train', 'args.yaml')
    with open(yaml_path, mode='r') as file:
        args_yaml = file.read()
    
    params = list(map(lambda x: x.split(': ')[0], args_yaml.strip().splitlines()))
    df = pd.DataFrame(columns=params)
    
    # Add each model as independent row to the df with args as columns.
    for i, model_name in enumerate([model_1, model_2]):
        yaml_path = os.path.join(runs_dir, model_name, 'detect', 'train', 'args.yaml')
        
        with open(yaml_path, mode='r') as file:
            args_yaml = file.read()
        
        paired_args = args_yaml.strip().splitlines()
        args = list(map(lambda x: x.split(': ')[1], paired_args))
        
        df.loc[model_name] = args
    
    # return df with only args with differences.
    model_1_row = df.iloc[0]
    model_2_row = df.iloc[1]
    diff_ser = model_1_row != model_2_row

    diff_cols = []
    for i, arg in enumerate(diff_ser):
        col = diff_ser.index[i]
        if (arg is True) & (col not in ['name', 'save_dir']):
            diff_cols.append(col)
            
    return df[diff_cols] 
        
############################################################################################################

def compare_main_metrics_df(
    runs_dir,
    models,
    int_names=False
):
    """
    Compares the main metrics from the last epoch of training for multiple YOLO models.

    Parameters:
    ----------
    runs_dir: str
        Directory containing the YOLO model runs, where each model's training results are saved.
    models: list
        List of model names to compare.
    int_names: bool, optional
        If True, model names in the output DataFrame will be replaced with sequential integers (e.g., 'model_1', 'model_2'). 
        Defaults to False, where the original model names are used.

    Returns:
    --------
    pd.DataFrame
        A DataFrame where each row corresponds to a model, containing the metrics from the last epoch of training. 
        The index of the DataFrame is the model name (or integer if `int_names=True`), and columns are the metric names.
    """
    for i, model_name in enumerate(models):
        model_results_path = os.path.join(runs_dir, model_name, 'detect', 'train')
    
        ### Stage 1: show last epoc metrics for each model ###
        epoch_metrics = pd.read_csv(os.path.join(model_results_path, 'results.csv'))
        epoch_metrics.columns = epoch_metrics.columns.str.strip()  # Large blank space before text in cols.
        epoch_metrics.rename(columns={'epoch':'last_epoch'}, inplace=True)

        if int_names:
            name = 'model_' + str(i + 1)
        else:
            name = model_name
        
        epoch_metrics['model'] = name
    
        if i == 0:
            df = pd.DataFrame(epoch_metrics.iloc[-1]).T
        else:
            df.loc[i] = epoch_metrics.iloc[-1]
    
    df = df.set_index('model')
    return df

############################################################################################################

def compare_metrics_plot(
    runs_dir,
    model_1,
    model_2,
    figsize=(35, 35)
):
    """
    Generates and displays plots comparing training batch images and main metrics between two YOLO models.

    Parameters:
    ----------
    runs_dir: str
        Directory containing the YOLO model runs, where each model's training results are stored.
    model_1: str
        Name of the first model to compare.
    model_2: str
        Name of the second model to compare.
    figsize: tuple, optional
        Size of the figure for the plots. Defaults to (35, 35).

    Returns:
    --------
    None
        The function displays images of training batches and various metrics for both models. 
        It does not return any values but shows the plots directly.
    """
    for i, model_name in enumerate([model_1, model_2]):
        model_results_path = os.path.join(runs_dir, model_name, 'detect', 'train')
        
        ### display 3 training batch images for each model ###
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        for j, train_img_fname in enumerate(['train_batch0.jpg', 'train_batch1.jpg', 'train_batch2.jpg']):
            train_img_path = os.path.join(model_results_path, train_img_fname)
            image = Image.open(train_img_path)
            axes[j].imshow(np.array(image))
            axes[j].set_title(f"model_{i + 1}: {train_img_fname}")
            axes[j].axis('off')
            plt.subplots_adjust(wspace=0.05)  # Default is 0.2
    
    ### show main metric pics (loss, conf matrices, PR curve) ###
    for g, img_fname in enumerate(['results.png', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'PR_curve.png']):
        fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        for i, model in enumerate([model_1, model_2]):
            img_path = os.path.join(runs_dir, model, 'detect', 'train', img_fname)
            image = Image.open(img_path)
        
            axes2[i].imshow(np.array(image))
            axes2[i].set_title(model)
            axes2[i].axis('off')
            # Adjust the spacing between the plots
            plt.subplots_adjust(wspace=0.05)  # Default is 0.2
         
############################################################################################################

def show_pred_images(
    yolo_models_dict,
    pred_img_path,
    mode='detect',
    conf=0.25,
    iou=0.7,
    show_orig_img=False,
    figsize=(10, 10)
):
    """
    Displays prediction results from multiple YOLO models on a given image.

    Parameters:
    ----------
    yolo_models_dict: dict
        Dictionary where keys are model names and values are YOLO model instances. 
        Each model instance should have `predict` or `track` methods for generating predictions.
    pred_img_path: str
        Path to the image file on which predictions are to be made.
    mode: str, optional
        Mode for prediction. Can be 'detect' for object detection or 'track' for object tracking. Defaults to 'detect'.
    conf: float, optional
        Confidence threshold for predictions. Defaults to 0.25.
    iou: float, optional
        Intersection over Union (IoU) threshold for non-maximum suppression. Defaults to 0.7.
    show_orig_img: bool, optional
        If True, displays the original image before showing the predictions. Defaults to False.
    figsize: tuple, optional
        Size of the figure for displaying the plots. Defaults to (10, 10).

    Returns:
    --------
    None
        The function displays the original image (if `show_orig_img` is True) and the predictions from each model 
        in separate subplots. It does not return any values but shows the plots directly.
    """
    img_name = pred_img_path.split('/')[-1]
    print(img_name)

    if show_orig_img:
        plt.figure(figsize=figsize)
        plt.title(img_name)
        plt.imshow(Image.open(pred_img_path))
        plt.axis('off')
        
    fig, axes = plt.subplots(nrows=len(yolo_models_dict), ncols=1, figsize=figsize)
    
    for i, (model_name, yolo_model) in enumerate(yolo_models_dict.items()):
        if mode == 'detect':
            pred_img_results = yolo_model.predict(pred_img_path, verbose=False, iou=iou, conf=conf)[0]
        elif mode == 'track':
            pred_img_results = yolo_model.track(pred_img_path, verbose=False, iou=iou, conf=conf)[0]
        else:
            raise Exception("mode must be one of the following: ['detect', 'track'].")
            
        n_pred_objects   = len(pred_img_results.boxes)
        pred_img_tensor  = pred_img_results.plot(conf=False, labels=True)
        axes[i].imshow(pred_img_tensor[..., ::-1])
        axes[i].set_title(f"{model_name} ({n_pred_objects})")
        axes[i].axis('off')


############################################################################################################

def show_individual_detections(
    model_path,
    img_path,
    class_list,
    figsize_detect=(3, 3)
):
    """
    Displays the predictions from a YOLO model on a given image, including the annotated image with bounding boxes 
    and individual cropped detections.

    Parameters:
    ----------
    model_path: str
        Path to the YOLO model weights file.
    img_path: str
        Path to the image file on which predictions are to be made.
    class_list: list
        List of class names corresponding to class IDs used by the YOLO model.
    figsize_detect: tuple, optional
        Size of the figure for displaying individual cropped detections. Defaults to (3, 3).
    """
    yolo_model = YOLO(model_path, verbose=False)
    pred_results = yolo_model.predict(img_path, iou=0.3)[0]
    
    pred_img = pred_results.plot(conf=False, labels=False, line_width=2)
    plt.imshow(pred_img[..., ::-1])
    img_name = img_path.split('/')[-1]
    plt.title(f"{img_name} ({len(pred_results)})")
    plt.axis('off')
    plt.show()
    
    orig_img = cv2.imread(img_path)
    for i, pred_box in enumerate(pred_results.boxes):
        class_id = int(pred_box.cls[0])
        class_name = class_list[class_id]
    
        conf_score = np.round(float(pred_box.conf[0]), 2)
        
        detection_xyxy = pred_box.xyxy
        x_min, y_min, x_max, y_max = detection_xyxy[0]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cropped_img = orig_img[y_min:y_max, x_min:x_max]
    
        plt.figure(figsize=figsize_detect)
        plt.imshow(cropped_img[..., ::-1])
        plt.title(f"Detection {i+1} ({class_name} {conf_score})")
        plt.axis('off')
        plt.show()
    
############################################################################################################

def put_text_in_img_middle_upper(
    img_tensor,
    text,
    font_scale=0.8,
    font_thickness=2,
    color=(255, 255, 255)  # white.
):
    """
    Adds text to the upper middle of an image tensor.

    Parameters:
    ----------
    img_tensor: numpy.ndarray
        Image tensor (in the form of a NumPy array) where the text will be added.
    text: str
        The text to be added to the image.
    font_scale: float, optional
        Scale factor that is multiplied by the font-specific base size. Defaults to 0.8.
    font_thickness: int, optional
        Thickness of the text strokes. Defaults to 2.
    color: tuple, optional
        Color of the text in BGR format. Defaults to white (255, 255, 255).

    Returns:
    --------
    numpy.ndarray
        The modified image tensor with the text added at the upper middle position.
    """
    # Load image tensor (np.array).
    image = img_tensor
    
    # Get image dimensions.
    height, width, _ = image.shape
    
    # Define the text and font.
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate the size of the text.
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate X position to center the text.
    x = (width - text_width) // 2
    
    # Calculate Y position (slightly below the top of the image).
    y = int(height * 0.05) + text_height  # Adjust the 0.1 multiplier to move the text up or down.
    
    # Add text to the image.
    mod_image_tensor = cv2.putText(image, text, (x, y), font, font_scale, color, font_thickness)
    return mod_image_tensor

############################################################################################################

def generate_video_tracking(
    raw_video_dir,
    raw_video_name,
    save_dir,
    model_dir,
    model_name,
    fps_slowdon_factor=1.0,
    conf_threshold=0.25,  # default in YOLO.
    iou_threshold=0.5,  # default in YOLO.
    show_detect_counts=False,
    text_separator='|',
    class_list=None,
):
    """
    Processes a video by applying object tracking with a YOLO model and generates a new video with tracking annotations.

    Parameters:
    ----------
    raw_video_dir: str
        Directory where the raw video is located.
    raw_video_name: str
        Name of the raw video file (without extension).
    save_dir: str
        Directory where the processed video will be saved.
    model_dir: str
        Directory where the YOLO model weights are located.
    model_name: str
        Name of the YOLO model to be used (without extension).
    fps_slowdon_factor: float, optional
        Factor to slow down the video. A value of 1.0 keeps the original speed. Values less than 1.0 slow down the video. Defaults to 1.0.
    conf_threshold: float, optional
        Confidence threshold for object detection. Defaults to 0.25.
    iou_threshold: float, optional
        Intersection over Union (IoU) threshold for object detection. Defaults to 0.5.
    show_detect_counts: bool, optional
        Whether to show detection counts on the video frames. If True, `class_list` must be provided. Defaults to False.
    text_separator: str, optional
        Separator used in the text showing detection counts. Defaults to '|'.
    class_list: list of str, optional
        List of class names corresponding to YOLO model class indices. Required if `show_detect_counts` is True.

    Returns:
    -------
    None
        The function saves the processed video with annotations to the `save_dir` and prints a success message.
    """
    if (show_detect_counts is True) & (class_list is None):
        raise Exception('Provide YOLO model class_list to show detection counts in frames!')
        
    model_path = os.path.join(model_dir, f"{model_name}.pt")
    raw_video_path = os.path.join(raw_video_dir, f"{raw_video_name}.mp4")

    video_tracking_name = f"{raw_video_name}_{model_name}_fpsFactor{fps_slowdon_factor}_conf{conf_threshold}_iou{iou_threshold}_showDetectCounts{show_detect_counts}"
    video_tracking_path = os.path.join(save_dir, f"{video_tracking_name}.mp4")
    
    # Initialize videoCapture object.
    cap = cv2.VideoCapture(raw_video_path)
    ret, frame = cap.read()
    
    # Reading 1st frame.
    # ret is True if the frame was successfully read, and False if there was an error or the video has ended.
    if not ret:
        print('Error: Could not read the first frame of the video.')
        cap.release()
        exit()
        
    height, width, _ = frame.shape
    
    # Set new FPS (slowing down the video by 0.6x, so new FPS is half + little more of the original FPS).
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_fps = int(original_fps * fps_slowdon_factor)
    
    # Initialize VideoWriter with exact dimensions as original video.
    output = cv2.VideoWriter(
        video_tracking_path,  # where the ouput will be saved.
        cv2.VideoWriter_fourcc(*'MP4V'), # Four Character Code, commonly used for writing .mp4.
        new_fps,  # Get the frame rate per second of the original video (same speed).
        (width, height)  # dimensions of the original video frames.
    )

    # Load yolo model.
    yolo_model = YOLO(model_path)

    # Contar detecciones únicas y por clase.
    if show_detect_counts is True:
        unique_ids = set()
        detect_class_count_dict = {clase:0 for clase in class_list}
    
    # Process all frames from video file (stream).
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        pred_frame_results = yolo_model.track(frame,
                                        # stream=True,
                                        persist=True,
                                        verbose=False,
                                        conf=conf_threshold,
                                        iou=iou_threshold
                                        )[0]

        labeled_frame = pred_frame_results.plot(conf=False)

        if show_detect_counts is True:
            for detect_img_box in pred_frame_results.boxes:
                # DEBUG.
                # print(detect_img_box)
                if detect_img_box.is_track:
                    detect_id = int(detect_img_box.id)
                    if detect_id not in unique_ids:
                        detect_class = class_list[int(detect_img_box.cls)]
                        detect_class_count_dict[detect_class] += 1
                        unique_ids.add(detect_id)
            
            # Definir texto de conteo.
            total_count = str(len(unique_ids))
            text = 'Total:' + total_count
            tuplas_clase_count = [(clase.capitalize(), str(count)) for clase, count in detect_class_count_dict.items()]
            for tupla in tuplas_clase_count:
                text += f" {text_separator} {':'.join(tupla)}"
            
            # Añadir texto en imagen anotado.
            labeled_frame = put_text_in_img_middle_upper(
                img_tensor=labeled_frame,
                text=text,
                font_scale=0.6,
                font_thickness=2,
                color=(255, 255, 255)  # white.
            )
        
        # Write processed frame to an output video.
        output.write(labeled_frame)
    
    cap.release()            # Release the video capture object.
    output.release()         # Release the video writer object.
    cv2.destroyAllWindows()  # Close all OpenCV windows.
    
    print(f"{video_tracking_path} has been successfully created!") 