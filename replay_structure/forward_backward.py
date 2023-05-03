"""
Implementation of the forward backward algorithm for one-step and two-step Hidden
Markov Models (Bishop 2009, Chapter 13).
"""

import numpy as np
import os
import torch
import pickle


def load_data(filename, print_filename=True):
    if print_filename:
        print("loading ", filename)
    with open(filename, "rb") as file_object:
        raw_data = file_object.read()
        deserialized = pickle.loads(raw_data)
    return deserialized


def save_data(data, filename, print_filename=True):
    if print_filename:
        print("saving ", filename)
    serialized = pickle.dumps(data)
    with open(filename, "wb") as file_object:
        file_object.write(serialized)


class Forward_Backward:
    """Implementation of the forward backward algorithim for a one-step Hidden Markov
    Model."""

    def __init__(self, HMM_params: dict):
        """HMM params dictionary:
        'emission_probabilities' p(x_t|z_t) - shape (K x T)
        'transition_matrix' p(z_t|z_(t-1)) - shape (K x K)
        'initial_state_prior' p(z_t) - shape (1 x K)"""
        self.params = HMM_params
        (self.n_states, self.n_timesteps) = np.shape(
            self.params["emission_probabilities"]
        )

    def run_forward_backward_algorithm(self, *args) -> dict:
        """FB Algorithm Outputs Dictionary:
        'data_likelihood': scalar
        'latent_marginals': (T x K)
        'latent_joints': (T x K x K)"""
        if len(args) > 0:
            if args[0] == "no joints":
                joints = False
        else:
            joints = True
        HMM_outputs = dict()
        # run forward and backward pass (n * k)
        alphas, conditionals = self.forward_pass()
        betas = self.backward_pass(conditionals)
        HMM_outputs["alphas"] = alphas
        HMM_outputs["betas"] = betas
        HMM_outputs["conditionals"] = conditionals
        # use results to calcualte the data likelihood, marginals and joints.
        HMM_outputs["data_likelihood"] = self.calculate_data_likelihood(conditionals)  # p()
        HMM_outputs["latent_marginals"] = self.calculate_latent_marginals(
            alphas, betas, conditionals
        )
        if joints:
            HMM_outputs["latent_joints"] = self.calculate_latent_joints(alphas, betas)
            print("joints calculated")
        return HMM_outputs

    def forward_pass(self):
        # initialize alphas
        alphas = np.zeros((self.n_timesteps, self.n_states))
        conditionals = np.zeros(self.n_timesteps)
        # calculate alpha_0
        alphas_init = (
            self.params["initial_state_prior"]
            * self.params["emission_probabilities"][:, 0]
        )
        alphas[0] = alphas_init / np.sum(alphas_init)
        conditionals[0] = np.sum(alphas_init)
        # calculate alpha_0:T recursively forward in time
        for n in range(1, self.n_timesteps):
            alphas_init = self.params["emission_probabilities"][:, n] * np.matmul(
                alphas[n - 1], self.params["transition_matrix"].T
            )
            alphas[n] = alphas_init / np.sum(alphas_init)  # normalize p(z_t, x_t)
            conditionals[n] = np.sum(alphas_init)  # sum over z -> p(x_1:t)
        return alphas, conditionals

    def backward_pass(self, conditionals):
        # initialize betas
        betas = np.zeros((self.n_timesteps, self.n_states))
        betas[-1] = 1
        # calculate betas recursively backward in time
        for n in range(self.n_timesteps - 2, -1, -1):
            betas_init = np.matmul(
                self.params["transition_matrix"],
                betas[n + 1] * self.params["emission_probabilities"][:, n + 1],
            )
            betas[n] = betas_init / conditionals[n]
        return betas

    def calculate_data_likelihood(self, conditionals):
        data_likelihood = np.sum(np.log(conditionals))
        return data_likelihood

    def calculate_latent_marginals(self, alphas, betas, conditionals):
        latent_marginals = np.zeros((self.n_timesteps, self.n_states))
        for n in range(self.n_timesteps):
            latent_marginals[n] = (alphas[n] * betas[n]) / conditionals[n]
        return latent_marginals

    def calculate_latent_joints(self, alphas, betas):
        latent_joints = np.zeros((self.n_timesteps - 1, self.n_states, self.n_states))
        for n in range(self.n_timesteps - 1):
            latent_joints[n] = (
                (
                    self.params["emission_probabilities"][:, n + 1]
                    * betas[n + 1]
                    * self.params["transition_matrix"]
                ).T
                * alphas[n]
            ).T
        return latent_joints


class Forward_Backward_xy(Forward_Backward):
    """Efficient implementation of the forward backward algorithim taking into account
    xy symmetry. Used by the Diffusion model."""

    def __init__(self, HMM_params):
        """HMM params dictionary:
        'emission_probabilities' p(x_t|z_t) - shape (K x T)
        'transition_matrix' p(z_t|z_(t-1)) - shape (K x K)
        'initial_state_prior' p(z_t) - shape (1 x K)"""
        self.params = HMM_params
        (self.n_states, self.n_timesteps) = np.shape(
            self.params["emission_probabilities"]
        )
        self.n_states_sqrt = np.sqrt(self.n_states).astype(int)

    def forward_pass(self):
        # calculate alpha_0:T recursively forward i
        # initialize alphas
        alphas = np.zeros((self.n_timesteps, self.n_states))
        conditionals = np.zeros(self.n_timesteps)
        # calculate alpha_0
        alphas_init = (
            self.params["initial_state_prior"]
            * self.params["emission_probabilities"][:, 0]
        )
        alphas[0] = alphas_init / np.sum(alphas_init)
        conditionals[0] = np.sum(alphas_init)
        # calculate alpha_0:T recursively forward in time
        for n in range(1, self.n_timesteps):
            # do operation on x and y separately
            alphas_2d = np.reshape(
                alphas[n - 1], (self.n_states_sqrt, self.n_states_sqrt)
            )
            alphas_xy = np.matmul(alphas_2d, self.params["transition_matrix"].T)
            alphas_xy = np.matmul(self.params["transition_matrix"], alphas_xy)
            alphas_xy = self.params["emission_probabilities"][:, n] * alphas_xy.reshape(
                -1
            )
            alphas[n] = alphas_xy / np.sum(alphas_xy)
            conditionals[n] = np.sum(alphas_xy)
        return alphas, conditionals

    def backward_pass(self, conditionals):
        # initialize betas
        betas = np.zeros((self.n_timesteps, self.n_states))
        betas[-1] = 1
        # calculate betas recursively backward in time
        for n in range(self.n_timesteps - 2, -1, -1):
            betas_xy = betas[n + 1] * self.params["emission_probabilities"][:, n + 1]
            betas_xy_2d = np.reshape(betas_xy, (self.n_states_sqrt, self.n_states_sqrt))
            betas_xy = np.matmul(betas_xy_2d, self.params["transition_matrix"].T)
            betas_xy = np.matmul(self.params["transition_matrix"], betas_xy)
            betas[n] = betas_xy.reshape(-1) / conditionals[n]
        return betas


class Forward_Backward_order2:
    """Implementation of the forward backward algorithim for a two-step hidden markov
    model. Used by the Momentum model."""

    def __init__(self, HMM_params):
        """HMM params dictionary:
        'emission_probabilities' p(x_t|z_t) - shape (K x T)
        'transition_matrix' p(z_t|z_(t-1)) - shape (K x K)
        'initial_state_prior' p(z_t) - shape (1 x K)"""
        self.params = HMM_params
        (self.n_states, self.n_timesteps) = np.shape(
            self.params["emission_probabilities"]
        )
        # self.params_to_torch()

    def run_forward_backward_algorithm(self, plotting=False):
        """FB Algorithm Outputs Dictionary:
        'data_likelihood': scalar
        'latent_marginals': (T x K)"""
        HMM_outputs = dict()
        conditionals, alphas, alphas_marginals = self.forward_pass(
            plotting=plotting
        )
        HMM_outputs["alphas"] = alphas
        HMM_outputs["conditionals"] = conditionals
        HMM_outputs["data_likelihood"] = self.calculate_data_likelihood(conditionals)
        # the backward pass is computationally intensive, so only want to run this
        # for visualization (not when running over parameter gridsearch).
        if plotting:
            betas = self.backward_pass(conditionals)
            HMM_outputs["latent_marginals"] = self.calculate_latent_marginals(
                conditionals, alphas_marginals, betas
            )
        return HMM_outputs

    def forward_pass(self, plotting=False):
        # calculate alpha_0:T recursively forward i

        # initialize alphas and conditionals
        alphas = torch.zeros((self.n_timesteps, self.n_states))
        conditionals = np.zeros(self.n_timesteps, dtype=np.longdouble)
        # initialize list of alphas to calculate marginals
        alphas_marginals = []

        # calculate alpha_0
        alpha_0 = (
            self.params["initial_state_prior"]
            * self.params["emission_probabilities"][:, 0]
        )
        conditionals[0] = torch.sum(alpha_0)
        # print(conditionals)
        alpha_0 = alpha_0 / conditionals[0]
        alphas[0] = alpha_0
        alphas_marginals.append(alpha_0)

        # calculate alpha_1
        alpha_1 = (
            (self.params["initial_transition"] * alpha_0).t()
            * self.params["emission_probabilities"][:, 1]
        ).t()
        conditionals[1] = torch.sum(alpha_1)
        alpha_1 = alpha_1 / conditionals[1]
        alphas[1] = torch.sum(alpha_1, 0)
        alphas_marginals.append(alpha_1)

        alpha_t = alpha_1.detach().clone()
        # calculate alpha_2:T
        for t in range(2, self.n_timesteps):
            x_sum = torch.einsum(
                "nlj,lj->nl", self.params["transition_matrix"], alpha_t
            )
            alpha_t = (x_sum.t() * self.params["emission_probabilities"][:, t]).t()
            alpha_t = alpha_t.numpy().astype(np.longdouble)
            conditionals[t] = np.sum(alpha_t)
            alpha_t = alpha_t / np.sum(alpha_t)
            alpha_t = torch.from_numpy(alpha_t.astype(np.double))
            alphas[t] = torch.sum(alpha_t, 1)

            alphas_marginals.append(alpha_t)

        return conditionals, alphas, alphas_marginals

    def backward_pass(self, conditionals):
        # initialize list of betahs to calculate marginals
        betas = []

        # initiate with beta_T
        beta_t = torch.ones(
            (
                self.n_states,
                self.n_states,
            ),
            dtype=torch.double,
        )
        betas.append(beta_t.t())

        # calculate betas recursively backward in time
        for n in range(self.n_timesteps - 2, -1, -1):
            betas_x = beta_t * self.params["emission_probabilities"][:, n + 1]
            
            x_sum = torch.einsum(
                "jln,ln->jl",
                self.params["transition_matrix"].permute(2, 1, 0),
                betas_x,
            )
            beta_t = x_sum / conditionals[n]

            beta_t_save = beta_t.permute(1, 0)  #0,1 or 1,0?
            betas.append(beta_t_save)
        
        # reverse order of betas to 0 to T
        return betas[::-1]


    def calculate_data_likelihood(self, conditionals):
        data_likelihood = np.sum(np.log(conditionals))
        # print("LIKELIHOOD: ", data_likelihood)
        return data_likelihood

    def calculate_latent_marginals(self, conditionals, alphas, betas):
        # initialize marginals
        latent_marginals = torch.zeros((self.n_timesteps, self.n_states))

        for n in range(1, self.n_timesteps):

            alphas[n][alphas[n] == 0] = np.power(10.0, -30)
            betas[n][betas[n] == 0] = np.power(10.0, -30)

            marginal = torch.sum(((alphas[n] * betas[n]) * conditionals[n]), 0)
            latent_marginals[n] = marginal

        return latent_marginals
