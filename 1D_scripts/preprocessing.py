import os
import click
import scipy.io as spio
from typing import Optional
import pickle

from replay_structure.metadata import (
    Poisson,
)

from replay_structure.read_write import load_data, save_data
from replay_structure.config import RatDay_Preprocessing_Parameters, Ripple_Preprocessing_Parameters
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.metadata import DATA_PATH


def run_preprocessing(
    data,
    bin_size_cm: int,
    time_window_ms: int,
    running_direction: bool,
    rotate_placefields: bool,
) -> None:
    
    print(f"Running with {bin_size_cm}cm bins")
    params = RatDay_Preprocessing_Parameters(
        bin_size_cm=bin_size_cm, rotate_placefields=rotate_placefields
    )
    ripple_params = Ripple_Preprocessing_Parameters(
            params, time_window_ms=time_window_ms)
    
    preprocessed_data = RatDay_Preprocessing(data, params, running_direction=running_direction)
    filename = os.path.join(
    DATA_PATH, f"preprocessed_{bin_size_cm}cm.obj")
    save_data(preprocessed_data, filename)
    
    spikemat_data = Ripple_Preprocessing(preprocessed_data, ripple_params)
    filename = os.path.join(
    DATA_PATH, f"spikemat_{bin_size_cm}cm_{time_window_ms}ms.obj")
    save_data(spikemat_data, filename)

    # reformat data for structure analysis
    print("Reformatting data")
    structure_analysis_input = Structure_Analysis_Input.reformat_ripple_data(
            spikemat_data, Poisson(), running_direction=False)
    folder = "structure_analysis_input"
    likelihood_function = "poisson"
    filename = os.path.join(
        DATA_PATH, folder,
        f"ripple_spikemat_{bin_size_cm}cm_{time_window_ms}ms_{likelihood_function}.obj"
    )
    save_data(structure_analysis_input, filename)


@click.command()
@click.option("--data_file", type=click.STRING, required=True)
@click.option("--bin_size_cm", type=click.INT, default=5)
@click.option("--time_window_ms", type=click.INT, default=15)
@click.option("--running_direction", type=click.BOOL, default=False)
@click.option("--rotate_placefields", type=click.BOOL, default=False)
def main(
    data_file: str,
    bin_size_cm: int,
    time_window_ms: int,
    running_direction: bool,
    rotate_placefields: bool,
):

    # load data
    print("loading data")
    # file_path = os.path.join(DATA_PATH, data_file)
    # data = spio.loadmat(file_path)

    file_path = os.path.join(DATA_PATH, data_file)
    data = load_data(file_path, False)

    # preprocess, calculate place fields, ripple preprocessing
    run_preprocessing(
            data,
            bin_size_cm,
            time_window_ms,
            running_direction,
            rotate_placefields
        )
    
    


if __name__ == "__main__":
    main()
