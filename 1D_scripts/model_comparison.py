import click
import os
import numpy as np
import re
from typing import Optional, Dict

from replay_structure.metadata import DATA_PATH, RESULTS_PATH
from replay_structure.structure_models_gridsearch import Structure_Gridsearch
from replay_structure.model_comparison import (
    Gridsearch_Marginalization,
    Model_Comparison,
)
from replay_structure.read_write import (
    load_data,
    save_data,
    aggregate_momentum_gridsearch_
)

from replay_structure.metadata import (
    MODELS,
    Likelihood_Function,
    string_to_likelihood_function,
)


def run_gridsearch_marginalization(
    structure_data,
    structure_data_name: str,
    filename_ext: str,
):
    
    for model in ['momentum', 'diffusion', 'stationary_gaussian']:
        filename = os.path.join(
        RESULTS_PATH,
        f"{structure_data_name[0]}gridsearch_{model}_{structure_data_name[-1]}{filename_ext}.obj",
        )
        gs_results = load_data(filename)
        assert isinstance(gs_results, Structure_Gridsearch)

        if model=='momentum':
            gs_results = aggregate_momentum_gridsearch_(structure_data,
                                       gs_results
                                       )
    
        marginalized_gridsearch = Gridsearch_Marginalization(gs_results)

        filename = os.path.join(
        RESULTS_PATH,
        f"{structure_data_name[0]}marginalized_gridsearch_{model}_{structure_data_name[-1]}{filename_ext}.obj",
        )
        save_data(marginalized_gridsearch, filename)


def load_model_evidences(
    structure_data_name: str,
    filename_ext: str,
):

    model_evidences: Dict[str, np.ndarray] = dict()
    for model in MODELS:
        if model.n_params is not None:
            filename = os.path.join(
            RESULTS_PATH,
            f"{structure_data_name[0]}marginalized_gridsearch_{model.name}_{structure_data_name[-1]}{filename_ext}.obj",
            )
            model_evidences[str(model.name)] = load_data(filename).marginalized_model_evidences
        else:
            filename = os.path.join(
            RESULTS_PATH,
            f"{structure_data_name[0]}model_evidence_{model.name}{filename_ext}.obj",
            )
            model_evidences[str(model.name)] = load_data()
    return model_evidences


def run_model_comparison(
    structure_data,
    structure_data_name,
    filename_ext: str,
):

    run_gridsearch_marginalization(
        structure_data,
        structure_data_name,
        filename_ext,
    )

    model_evidences: dict = load_model_evidences(
        structure_data_name,
        filename_ext,
    )
    random_effects_prior = 10

    mc_results = Model_Comparison(
        model_evidences, random_effects_prior=random_effects_prior
    )

    filename = os.path.join(
        RESULTS_PATH,
        f"{structure_data_name[0]}model_comparison_{structure_data_name[-1]}{filename_ext}.obj",
    )
    save_data(mc_results, filename)

@click.command()
@click.option("--structure_data_file", type=click.STRING, default=None)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    strucutre_data_file: str,
    filename_ext: str
):
    filename = os.path.join(
    DATA_PATH,
    'structure_analysis_input',
    strucutre_data_file
    )
    structure_data = load_data(filename, False)
    structure_data_name = re.split('(spikemat_)', strucutre_data_file) 

    run_model_comparison(
        structure_data,
        structure_data_name,
        filename_ext,
    )


if __name__ == "__main__":
    main()
