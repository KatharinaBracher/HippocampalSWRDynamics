import os
import click
import scipy.io as spio
from typing import Optional

from replay_structure.read_write import save_data
from replay_structure.config import RatDay_Preprocessing_Parameters
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.metadata import DATA_PATH


def run_preprocessing(
    data,
    bin_size_cm: int,
    rotate_placefields: bool,
    save: bool,
) -> None:
    print(f"Running with {bin_size_cm}cm bins")
    params = RatDay_Preprocessing_Parameters(
        bin_size_cm=bin_size_cm, rotate_placefields=rotate_placefields
    )
    preprocessed_data = RatDay_Preprocessing(data, params)
    if save:
        filename = os.path.join(
        DATA_PATH, "preprocessed", f"_{bin_size_cm}cm.obj")
    save_data(preprocessed_data, filename)


@click.command()
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--rotate_placefields", type=click.BOOL, default=False)
def main(
    bin_size_cm: int,
    rotate_placefields: bool,
):

    # load data
    print("loading data")
    file_path = os.path.join(DATA_PATH, "OpenFieldData.mat")
    data = spio.loadmat(file_path)

    # preprocess, calculate place fields
    save = False
    run_preprocessing(
            data,
            bin_size_cm,
            rotate_placefields,
            save,
        )
    
    


if __name__ == "__main__":
    main()
