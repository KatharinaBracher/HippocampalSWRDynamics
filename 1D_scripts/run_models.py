import click
import os
import re

from typing import Optional

from replay_structure.metadata import (
    Likelihood_Function,
    Model,
    Diffusion,
    Momentum,
    Stationary,
    Stationary_Gaussian,
    Random,
    string_to_model
)
from replay_structure.metadata import DATA_PATH, RESULTS_PATH
import replay_structure.structure_models as models
from replay_structure.read_write import (
    load_data,
    save_data,
)


def run_model(
    model: Model,
    structure_data,
    model_name,
    data_file,
    filename_ext: str,
):
    
    if isinstance(model.name, Diffusion):
        pass
    if isinstance(model.name, Momentum):
        pass
    if isinstance(model.name, Stationary):
        model_results = models.Stationary(structure_data).get_model_evidences()
    if isinstance(model.name, Stationary_Gaussian):
        pass
    if isinstance(model.name, Random):
        model_results = models.Random(structure_data).get_model_evidences()

    data_file_ = re.split('(spikemat_)', data_file) 
    filename = os.path.join(
        RESULTS_PATH,
        f"{data_file_[0]}model_evidence_{model_name}_{data_file_[-1]}",
    )
    save_data(model_results, filename)


@click.command()
@click.option(
    "--model_name",
    type=click.Choice(
        ["diffusion", "momentum", "stationary", "stationary_gaussian", "random"]
    ),
    required=True,
)
@click.option("--data_file", required=True,)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    model_name: str,
    data_file: str,
    filename_ext: str,
):
    model = string_to_model(model_name)
    filename = os.path.join(
    DATA_PATH,
    'structure_analysis_input',
    data_file)
    structure_data = load_data(filename, False)

    print(
            f"running {model.name} model on {data_file} data, "
        )
    run_model(
        model,
        structure_data,
        model_name,
        data_file,
        filename_ext,
    )


if __name__ == "__main__":
    main()