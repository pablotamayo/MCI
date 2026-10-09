
def write_dataset(dataset, file_path, descriptions = None, sep = '\t'):
    """ 
    Writes a dataset with extensions 'gct', 'txt, 'tsv' or 'csv'.

        Parameters: 
            dataset (pandas DataFrame): dataset to be saved in file_path.
            file_path (str): The pathname where to write the dataset.
            sep (str): The separator between entries in the dataset.
            descriptions (list): for gct files the descriptions for the second column of the file.
        
    """
    
    dataset = dataset.copy()

    file_extension = file_path.split('.')[-1]
    
    if file_extension == 'gct':
    
        if not (descriptions is None):
            dataset.insert(0, "Description", descriptions)

        dataset.index.name = "Name"
        dataset.columns.name = None

        with open(file_path, mode = "w") as output_file:

            output_file.writelines("#1.2\n{}\t{}\n".format(dataset.shape[0], dataset.shape[1] - 1))
            dataset.to_csv(output_file, sep= '\t')

    else:
        with open(file_path, mode="w") as output_file:

            dataset.to_csv(output_file, sep = sep)

    return    
