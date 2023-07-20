import click
import os
import re

import replay_structure.structure_models_gridsearch as gridsearch
from replay_structure.config import (
    Structure_Model_Gridsearch_Parameters,
    MAX_LIKELIHOOD_SD_METERS_RIPPLES,
    MAX_LIKELIHOOD_SD_METERS_RUN_SNIPPETS,
)
from replay_structure.metadata import (
    Model,
    Diffusion,
    Momentum,
    Stationary_Gaussian,
    string_to_model
)
from replay_structure.metadata import DATA_PATH, RESULTS_PATH
import replay_structure.structure_models as models
from replay_structure.read_write import (
    load_data,
    save_data,
)


def run_gridsearch(
    model: Model,
    structure_data,
    model_name,
    data_file,
    filename_ext: str,
):
    
    if isinstance(model.name, Diffusion):
        print('running gridsearch for difussion model')
        params = Structure_Model_Gridsearch_Parameters.ripple_diffusion_params()
        gs_results = gridsearch.Diffusion(structure_data, params)
    if isinstance(model.name, Momentum):
        print('running gridsearch for momentum model, this can take some time.')
        params = Structure_Model_Gridsearch_Parameters.ripple_momentum_params()
        gs_results = []
        adjust_params = False
        for spikemat_ind in list(structure_data.spikemats.keys()):
            momentum_gridsearch = gridsearch.Momentum(
                    structure_data, params, spikemat_ind, adjust_params=adjust_params
                )
            gs_results.append(momentum_gridsearch)
    if isinstance(model.name, Stationary_Gaussian):
        print('running gridsearch for stationary gaussian model')
        params = Structure_Model_Gridsearch_Parameters.ripple_stationary_gaussian_params()
        gs_results = gridsearch.Stationary_Gaussian(structure_data, params)

    data_file_ = re.split('(spikemat_)', data_file) 
    filename = os.path.join(
        RESULTS_PATH,
        f"{data_file_[0]}gridsearch_{model_name}_{data_file_[-1]}{filename_ext}.obj",
    )
    save_data(gs_results, filename)


@click.command()
@click.option(
    "--model_name",
    type=click.Choice(
        ["diffusion", "momentum", "stationary_gaussian"]
    ),
    required=True,
)
@click.option("--structure_data_file", required=True)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    model_name: str,
    structure_data_file: str,
    filename_ext: str,
):
    model = string_to_model(model_name)
    filename = os.path.join(
    DATA_PATH, 'structure_analysis_input', structure_data_file)
    structure_data = load_data(filename, False)

    print(
            f"running {model.name} model on {structure_data_file} data, "
        )
    run_gridsearch(
        model,
        structure_data,
        model_name,
        structure_data_file,
        filename_ext,
    )


if __name__ == "__main__":
    main()