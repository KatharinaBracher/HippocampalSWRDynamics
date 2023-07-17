import os
import click
import numpy as np
import scipy.io as spio
from typing import Optional

from replay_structure.metadata import (
    Poisson,
)

from replay_structure.read_write import save_data, load_data
from replay_structure.config import Run_Snippet_Preprocessing_Parameters, Selected_Data_Preprocessing_Parameters
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing, Selected_Data_Preprocessing
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.metadata import DATA_PATH, RESULTS_PATH


def run_preprocessing(
    preprocessed_data,
    spikemat_data,
    bin_size_cm: int,
    time_window_ms: int,
) -> None:
    
    print(f"Running with {bin_size_cm}cm bins")
    run_snippet_params = Run_Snippet_Preprocessing_Parameters(
            preprocessed_data.params, time_window_ms=time_window_ms)
    spikemat_run_snippets = Run_Snippet_Preprocessing(
            preprocessed_data, spikemat_data, run_snippet_params)
    filename = os.path.join(
    DATA_PATH, "run_snippets_spikemat", f"_{bin_size_cm}cm_{time_window_ms}ms.obj")
    save_data(spikemat_run_snippets, filename)

    full_data_params = Selected_Data_Preprocessing_Parameters(
        preprocessed_data.params, time_window_ms=time_window_ms)
    full_data_times = np.array([[preprocessed_data.data['pos_times_s'][0], 
                                 preprocessed_data.data['pos_times_s'][-1]]])
    spikemat_full_data = Selected_Data_Preprocessing(
            preprocessed_data, full_data_params, full_data_times)
    filename = os.path.join(
    DATA_PATH, "full_data_spikemat", f"_{bin_size_cm}cm_{time_window_ms}ms.obj")
    save_data(spikemat_full_data, filename)

    # reformat data for structur analysis
    folder = "structure_analysis_input"
    likelihood_function = "poisson"

    structure_analysis_input = Structure_Analysis_Input.reformat_run_snippet_data(
            spikemat_run_snippets, Poisson())
    filename = os.path.join(
        DATA_PATH, 
        folder,
        f"run_snippet_spikemat_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}.obj"
    )
    save_data(structure_analysis_input, filename)

    structure_analysis_input = Structure_Analysis_Input.reformat_selected_data(
            spikemat_full_data, Poisson())
    filename = os.path.join(
        DATA_PATH, 
        folder,
        f"full_data_spikemat_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}.obj"
    )
    save_data(structure_analysis_input, filename)



@click.command()
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=15)

def main(
    bin_size_cm: int,
    time_window_ms: int,
):

    # load data
    print("loading data")
    filename = os.path.join(
    DATA_PATH, "preprocessed", f"_{bin_size_cm}cm.obj")
    preprocessed_data = load_data(filename, print_filename=True)
    filename = os.path.join(
    DATA_PATH, "spikemat", f"_{bin_size_cm}cm_{time_window_ms}ms.obj")
    spikemat_data = load_data(filename, print_filename=True)


    # preprocess, calculate place fields, ripple preprocessing
    run_preprocessing(
            preprocessed_data,
            spikemat_data,
            bin_size_cm,
            time_window_ms
        )
    
    


if __name__ == "__main__":
    main()