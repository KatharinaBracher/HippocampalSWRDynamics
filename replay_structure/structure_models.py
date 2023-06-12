from replay_structure.metadata import Neg_Binomial_Params, Poisson_Params
import numpy as np
import scipy.stats as sp
import torch
from typing import Optional
from scipy.special import logsumexp

import replay_structure.utils as utils
import replay_structure.forward_backward as fb
from replay_structure.structure_analysis_input import Structure_Analysis_Input


class Structure_Model:
    """Base class implementation of structure model. This class defines the
    methods that are common across all models (getting the model evidence ond marginals
    across all SWRs in a session). The model-specific calculation of the model
    evidence is defined in each model child class."""

    def __init__(self, structure_data: Structure_Analysis_Input):
        self.structure_data = structure_data
        # get effective time_window_s if using scaled Poisson likelihood
        if isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            if (
                self.structure_data.params.likelihood_function_params.rate_scaling
                is not None
            ):
                self.emission_prob_time_window = (
                    self.structure_data.params.time_window_s
                    * self.structure_data.params.likelihood_function_params.rate_scaling
                )
            else:
                self.emission_prob_time_window = (
                    self.structure_data.params.time_window_s
                )

    def get_model_evidences(self) -> np.ndarray:
        model_evidence = np.zeros(len(self.structure_data.spikemats))
        for spikemat_ind in range(len(self.structure_data.spikemats)):
            model_evidence[spikemat_ind] = self.get_spikemat_model_evidence(
                spikemat_ind
            )
        return model_evidence

    def get_spikemat_model_evidence(self, spikemat_ind: int) -> float:
        if self.structure_data.spikemats[spikemat_ind] is not None:
            model_evidence, _ = self._calc_model_evidence(spikemat_ind)
        else:
            model_evidence = np.nan
        return model_evidence

    def get_marginals(self) -> dict:
        marginals = dict()
        for spikemat_ind in range(len(self.structure_data.spikemats)):
            marginals[spikemat_ind] = self.get_spikemat_marginals(spikemat_ind)
        return marginals

    def get_spikemat_marginals(self, spikemat_ind: int) -> np.ndarray:
        if self.structure_data.spikemats[spikemat_ind] is not None:
            _, marginals = self._calc_model_evidence(spikemat_ind)
        else:
            marginals = np.nan
        return marginals

    def _calc_model_evidence(self, spikemat_ind: int):
        """Implemented in child classes."""
        pass

    def _calc_emission_probabilities(self, spikemat_ind: int) -> np.ndarray:

        ######## (pfpos, pfneg) = (self.structure_data.pf_matrix[:,:self.structure_data.params.n_bins_x],
        ########                self.structure_data.pf_matrix[:,self.structure_data.params.n_bins_x:])
        ######## pf = (pfpos+pfneg)/2

        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probabilities = utils.calc_neg_binomial_emission_probabilities(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probabilities = utils.calc_poisson_emission_probabilities(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities

    def _calc_emission_probabilities_log(self, spikemat_ind: int) -> np.ndarray:

        ########(pfpos, pfneg) = (self.structure_data.pf_matrix[:,:self.structure_data.params.n_bins_x],
        ########                self.structure_data.pf_matrix[:,self.structure_data.params.n_bins_x:])
        ######## pf = (pfpos+pfneg)/2
        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probabilities_log = utils.calc_neg_binomial_emission_probabilities_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probabilities_log = utils.calc_poisson_emission_probabilities_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities_log

    def _calc_emission_probability_log(self, spikemat_ind: int) -> np.ndarray:

        ########(pfpos, pfneg) = (self.structure_data.pf_matrix[:,:self.structure_data.params.n_bins_x],
        ########                self.structure_data.pf_matrix[:,self.structure_data.params.n_bins_x:])
        ########pf = (pfpos+pfneg)/2
        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probability_log = utils.calc_neg_binomial_emission_probability_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probability_log = utils.calc_poisson_emission_probability_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probability_log


class Diffusion(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input, sd_meters: float):
        super().__init__(structure_data)
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.forward_backward_input = self._initialize_forward_backward_input()

    def _calc_model_evidence(self, spikemat_ind: int):
        ######## self.structure_data.running_direction=False
        # calculating emission probabilities over time p(x_t|z_t)
        self.forward_backward_input[
            "emission_probabilities"
        ] = self._calc_emission_probabilities(spikemat_ind)
        # calculating model evidence p(x_1:T|M) and marginals
        forward_backward_output = fb.Forward_Backward(
            self.forward_backward_input
        ).run_forward_backward_algorithm("no joints")
        # model evidence p(x_1:T)
        model_ev = forward_backward_output["data_likelihood"]
        # marginals p(z_t|x_1:T, M)
        marginals = forward_backward_output["latent_marginals"].T
        return model_ev, marginals

    def _initialize_forward_backward_input(self) -> dict:
        if self.structure_data.running_direction:
            # account for postitive and negative direction 
            n_bins_x = self.structure_data.params.n_bins_x*2
        else:
            n_bins_x = self.structure_data.params.n_bins_x

        forward_backward_input = dict()
        # calculate p(z_t)
        forward_backward_input["initial_state_prior"] = np.ones(
            n_bins_x) / n_bins_x
        # calculate p(z_t|z_t-1) 
        forward_backward_input["transition_matrix"] = self._calc_transition_matrix(
            self.sd_bins
        )
        return forward_backward_input

    def _calc_transition_matrix(self, sd_bins: float) -> np.ndarray:
        """KxK matrix
        transition probability p(z_t|z_t-1) 
        with gaussian transition structure N(t_t|z_t-1, sd**2I)"""
        norm = 1
        if self.structure_data.running_direction:
            norm = 2  # ensure sum to 1

        transition_mat = np.zeros(
            (self.structure_data.params.n_bins_x, self.structure_data.params.n_bins_x)
        )
        j = np.arange(self.structure_data.params.n_bins_x)  # t
        for i in range(self.structure_data.params.n_bins_x):  # t-1
            this_transition = np.exp(
                -((j - i) ** 2)
                / (2 * sd_bins ** 2 * self.structure_data.params.time_window_s)
            )
            transition_mat[:, i] = this_transition / (norm*np.sum(this_transition))

        # stacking transition matrix, to account for same locations for postive and negative rd
        if self.structure_data.running_direction:
            transition_mat = np.vstack((transition_mat,transition_mat))
            transition_mat = np.hstack((transition_mat,transition_mat))

        return transition_mat


class Momentum(Structure_Model):
    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        sd_0_meters: float,
        sd_meters: float,
        decay: float,
        emission_probabilities: Optional[np.ndarray] = None,
        plotting: bool = False
    ):
        super().__init__(structure_data)
        self.sd_0_meters = sd_0_meters
        self.sd_meters = sd_meters
        self.sd_0_bins = utils.meters_to_bins(
            sd_0_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.decay = decay
        self.emission_probabilities = emission_probabilities
        self.forward_backward_input = self._initialize_forward_backward_input()
        self.plotting = plotting
        
    def _calc_model_evidence(self, spikemat_ind: int):
        # calculating emission probabilities over time p(x_t|z_t)
        ######## self.structure_data.running_direction=False
        self.emission_probabilities=None

        self.forward_backward_input[
            "emission_probabilities"
        ] = self.get_emission_probabilities(spikemat_ind)

        # calculating model evidence p(x_1:T|M) and marginals p(z_t|x_1:T, M)
        forward_backward_output = fb.Forward_Backward_order2(
            self.forward_backward_input
        ).run_forward_backward_algorithm(
            plotting=self.plotting
        )
        model_ev = forward_backward_output["data_likelihood"]  # .numpy()
        if self.plotting:
            marginals = forward_backward_output["latent_marginals"].numpy().T
        else:
            marginals = forward_backward_output["alphas"].numpy().T
        return model_ev, marginals

    def _initialize_forward_backward_input(self) -> dict:
        if self.structure_data.running_direction:
            # account for postitive and negative direction 
            n_bins_x = self.structure_data.params.n_bins_x*2
        else:
            n_bins_x = self.structure_data.params.n_bins_x

        forward_backward_input = dict()
        forward_backward_input["initial_state_prior"] = torch.from_numpy(
            np.ones(n_bins_x)/n_bins_x
        )
        forward_backward_input["initial_transition"] = torch.from_numpy(
            self._calc_order1_transition_matrix(self.sd_0_bins)
        )
        forward_backward_input["transition_matrix"] = torch.from_numpy(
            self._calc_order2_transition_matrix(self.sd_bins, self.decay)
        )
        return forward_backward_input

    def _calc_order1_transition_matrix(self, sd: float):
        """(KxK) matrix"""
        norm = 1
        if self.structure_data.running_direction:
            norm = 2  # ensure sum to 1

        initial_transition = np.zeros(
            (self.structure_data.params.n_bins_x, self.structure_data.params.n_bins_x)
        )
        j = np.arange(self.structure_data.params.n_bins_x)  
        for i in range(self.structure_data.params.n_bins_x):  # t-1 x
            this_transition = np.exp(
            -((j - i) ** 2)
            / (2 * sd ** 2 * self.structure_data.params.time_window_s)
            )
            # this_transition = this_transition.reshape(-1)
            initial_transition[:, i] = this_transition / (norm*np.sum(this_transition))

        if self.structure_data.running_direction:
            initial_transition = np.vstack((initial_transition,initial_transition))
            initial_transition = np.hstack((initial_transition,initial_transition))

        return initial_transition

    def _calc_order2_transition_matrix(self, sd: float, decay: float):
        """(n x n x n) matrix"""
        norm = 1
        if self.structure_data.running_direction:
            norm = 2

        var_scaled = (
            (sd ** 2 * self.structure_data.params.time_window_s ** 2)
            / (2 * decay)
            * (1 - np.exp(-2 * decay * self.structure_data.params.time_window_s))
        )
        transition_mat = np.zeros(
            (
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_x,
            )
        )
        m = np.arange(self.structure_data.params.n_bins_x)  # t x/y
        for i in range(self.structure_data.params.n_bins_x):  # t-2 x/y
            for k in range(self.structure_data.params.n_bins_x):  # t-1 x/y
                mean = (
                    1 + np.exp(-self.structure_data.params.time_window_s * decay)
                ) * k - (np.exp(-self.structure_data.params.time_window_s * decay)) * i
                this_transition = np.exp(-((m - mean) ** 2) / (2 * var_scaled))
                norm_sum = norm*np.sum(this_transition)
                if norm_sum == 0:
                    max_prob_ind = (
                        0 if mean < 0 else self.structure_data.params.n_bins_x - 1
                    )
                    this_transition[max_prob_ind] = 1
                    transition_mat[:, k, i] = this_transition
                else:
                    transition_mat[:, k, i] = this_transition / norm_sum

        if self.structure_data.running_direction:
            transition_mat = np.concatenate((transition_mat, transition_mat), axis=0)
            transition_mat = np.concatenate((transition_mat, transition_mat), axis=1)
            transition_mat = np.concatenate((transition_mat, transition_mat), axis=2)
            
        return transition_mat

    def get_emission_probabilities(self, spikemat_ind: int):
        if self.emission_probabilities is None:
            emission_probabilities = torch.from_numpy(
                self._calc_emission_probabilities(spikemat_ind)
            )
        else:
            emission_probabilities = torch.from_numpy(self.emission_probabilities)
        return emission_probabilities


class Stationary(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input):
        super().__init__(structure_data)

    def _calc_model_evidence(self, spikemat_ind: int):
        # calculating log emission probability of spikes summed over time p(x|z)
        emission_probability_log = self._calc_emission_probability_log(spikemat_ind)
        # calculating model evidence p(x_1:T|M)
        norm_bins = emission_probability_log.shape[0]
        joint_probability = emission_probability_log - np.log(
            norm_bins
        )
        # approximate maximum
        model_evidence = logsumexp(joint_probability)
        # calculating marginals p(x_t|z_t)/p(x_1:T) = p(z_t|x_1:T, M)
        marginals = np.exp(emission_probability_log - emission_probability_log.max())
        return model_evidence, marginals


class Stationary_Gaussian(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input, sd_meters: float):
        super().__init__(structure_data)
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.latent_probabilities_normalized = self._calc_latent_probabilities()

    def _calc_model_evidence(self, spikemat_ind: int):
        # calculating emission probabilities over time p(x_t|z_t)
        ######## self.structure_data.running_direction=False
        emission_probabilities = self._calc_emission_probabilities(spikemat_ind)
        # calculating model evidence p(x_1:T|M, params)
        sum_z = np.matmul(
            emission_probabilities.T, self.latent_probabilities_normalized
        )
        sum_t = np.sum(np.log(sum_z), axis=0)
        # marginalizing over uniform prior µ yileds 1/K
        norm_bins = emission_probabilities.shape[0]
        model_evidence = logsumexp(-np.log(norm_bins) + sum_t)
        # calculating marginals p(x_t|z_t)*p(z_t) = p(z_t|x_1:T, M)
        marginals = np.matmul(
            emission_probabilities.T, self.latent_probabilities_normalized.T
        ).T
        return model_evidence, marginals

    def _calc_latent_probabilities(self):
        # M = gaussian, p(z_t|M, params) = N(z_t|µ, sd**2I))
        # initialze matrix (Nz x Nu)
        if self.structure_data.running_direction:
            # account for postitive and negative direction 
            n_bins_x = self.structure_data.params.n_bins_x*2
        else:
            n_bins_x = self.structure_data.params.n_bins_x

        latent_mat = np.zeros(
            (
                n_bins_x,
                n_bins_x,
            )
        )
        x = np.arange(n_bins_x)
        for m in range(n_bins_x):
            this_prob = sp.multivariate_normal(
                m, self.sd_bins ** 2
            ).pdf(np.transpose(x))
            latent_mat[:, m] = this_prob / this_prob.sum()
        latent_mat = latent_mat.reshape(
            (n_bins_x, n_bins_x)
        )
        return latent_mat


class Random(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input):
        super().__init__(structure_data)

    def _calc_model_evidence(self, spikemat_ind: int):
        # calculating log emission probabilities over time p(x_t|z_t)
        emission_probabilities_log = self._calc_emission_probabilities_log(spikemat_ind)
        # calculating marginals p(z_t|x_1:T, M)
        marginals = np.exp(emission_probabilities_log)

        # _marginalize over running direction to get evidence
        '''if self.structure_data.running_direction:
            emission_probabilities_log = emission_probabilities_log[
                :self.structure_data.params.n_bins_x,:] + emission_probabilities_log[
                    self.structure_data.params.n_bins_x:,:]'''
        
        # calculating model evidence p(x_1:T|M)
        norm_bins = emission_probabilities_log.shape[0]
        full_sum = emission_probabilities_log - np.log(norm_bins)
        # aproximate maximum
        sum_z = logsumexp(full_sum, axis=0)
        model_evidence = np.sum(sum_z)

        return model_evidence, marginals
