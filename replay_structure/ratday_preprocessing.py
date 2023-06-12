import numpy as np
from scipy.ndimage import gaussian_filter

from replay_structure.config import RatDay_Preprocessing_Parameters
import replay_structure.utils as utils


class RatDay_Preprocessing:
    """
    Preprocesses data. The primary functionality is to:
    1. Reformat data from original MATLAB struct format.
    2. Calculate ripple data
    3. Clean the position and spiking recordings of recording gaps.
    4. Calculate velocity.
    5. Calculate 1D place fields.
    """

    def __init__(self, matlab_data, params: RatDay_Preprocessing_Parameters, running_direction=False) -> None:
        """
        Reformats and preprocesses the data.
        """
        self.params = params
        print("Reformating data")
        self.raw_data = self.reformat_data(matlab_data)
        print("Cleaning data")
        self.data = self.clean_recording_data(self.raw_data)
        print("Calculating run periods")
        if running_direction:
            self.velocity_info_pos = self.calculate_velocity_info(rd='+')
            print("Calculating positive place fields")
            np.random.seed(0)
            self.place_field_data_pos = self.calculate_place_fields(self.velocity_info_pos)

            self.velocity_info_neg = self.calculate_velocity_info(rd='-')
            print("Calculating negative place fields")
            np.random.seed(0)
            self.place_field_data_neg = self.calculate_place_fields(self.velocity_info_neg)

            self.place_field_data = self.combine_place_field_data()
            print("DONE")
        else:
            self.velocity_info = self.calculate_velocity_info()
            print("Calculating place fields")
            np.random.seed(0)
            self.place_field_data = self.calculate_place_fields(self.velocity_info)
            print("DONE")

    # @staticmethod
    def reformat_data(self, matlab_data) -> dict:
        """Reformat original data loaded from Matlab file.
        """
        spike_flat_sorted, spike_ids_sorted = self.transform_spike_data(matlab_data)
        data = dict()
        data["significant_ripples"] = np.arange(0, matlab_data['RippleTimes'].shape[0],1)
        data["ripple_info"] = matlab_data['RippleTimes'] # missing peak times
        data["spike_ids"] = spike_ids_sorted
        data["spike_times_s"] = spike_flat_sorted
        data["pos_times_s"] = matlab_data['pos']['timestamp']
        data["pos_xy_cm"] = matlab_data['pos']['data']
        if matlab_data['pos']['units'] == 'meters':
            data["pos_xy_cm"] = data["pos_xy_cm"]*100
        data["ripple_times_s"] = matlab_data['RippleTimes']
        data["n_ripples"] = len(data["ripple_times_s"])
        data["n_cells"] = np.max(data["spike_ids"] + 1)
        data["well_locations"] = np.array([[np.nanmin(data["pos_xy_cm"]),1], 
                                          [np.nanmax(data["pos_xy_cm"]),2]])
        
        return data

    # ----------------------------
 
    def transform_spike_data(self, matlab_data):
        """Reformat spike data loaded from Matlab file to 1D array.
        """
        spikes_flat = np.array([element 
                                for sublist in matlab_data['SpikeTimes'] 
                                for element in sublist])
        spike_ids = [np.full_like(cell, i) 
                     for i,cell in enumerate(matlab_data['SpikeTimes'])]
        spike_ids_flat = np.array([element 
                                   for sublist in spike_ids 
                                   for element in sublist])
        spike_flat_sorted = sorted(spikes_flat)
        spike_ids_sorted = [int(x) for _, x in sorted(zip(spikes_flat, spike_ids_flat))]
        return np.array(spike_flat_sorted), np.array(spike_ids_sorted)
    # ----------------------------

    def clean_recording_data(self, raw_data: dict) -> dict:
        """Cleaning affects "pos_xy", "pos_times", "spike_ids", "spike_times".
        """
        (
            pos_xy_aligned,
            pos_times_aligned,
            spike_ids_aligned,
            spike_times_aligned,
        ) = self.align_spike_and_position_recording_data(
            raw_data["pos_xy_cm"],
            raw_data["pos_times_s"],
            raw_data["spike_ids"],
            raw_data["spike_times_s"],
        )
        
        # store cleaned data in new dictionary
        cleaned_data = raw_data.copy()
        cleaned_data["pos_xy_cm"] = pos_xy_aligned
        cleaned_data["pos_times_s"] = pos_times_aligned
        cleaned_data["spike_ids"] = spike_ids_aligned
        cleaned_data["spike_times_s"] = spike_times_aligned

        return cleaned_data

    # ----------------

    def align_spike_and_position_recording_data(
        self,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
    ) -> tuple:

        (spike_ids_aligned, spike_times_aligned) = self.remove_spikes_without_position(
            spike_ids, spike_times, pos_times
        )
        (pos_xy_aligned, pos_times_aligned) = self.remove_position_without_spikes(
            pos_xy, pos_times, spike_times
        )
        (pos_xy_aligned_cleaned, pos_times_aligned_cleand) = self.remove_nan_position(
            pos_xy_aligned, pos_times_aligned
            )

        return (
            pos_xy_aligned_cleaned,
            pos_times_aligned_cleand,
            spike_ids_aligned,
            spike_times_aligned,
        )
    
    def remove_nan_position(self, pos_xy: np.array, pos_times: np.array):
        nan_index = np.squeeze(np.argwhere(np.isnan(pos_xy)))
        pos_xy_cleaned = np.delete(pos_xy, nan_index)
        pos_times_cleaned = np.delete(pos_times, nan_index)
        return (pos_xy_cleaned, pos_times_cleaned)


    def remove_spikes_without_position(
        self, spike_ids: np.ndarray, spike_times: np.ndarray, pos_times: np.ndarray
    ):
        spikes_before_position_recording = spike_times < pos_times[0]
        spikes_after_position_recording = spike_times > pos_times[-1]
        spike_ids_aligned_to_position_recording = spike_ids[
            ~spikes_before_position_recording & ~spikes_after_position_recording
        ]
        spike_times_aligned_to_position_recording = spike_times[
            ~spikes_before_position_recording & ~spikes_after_position_recording
        ]
        return (
            spike_ids_aligned_to_position_recording,
            spike_times_aligned_to_position_recording,
        )

    def remove_position_without_spikes(
        self, pos_xy: np.ndarray, pos_times: np.ndarray, spike_times: np.ndarray
    ):
        position_before_spikes_recording = pos_times < spike_times[0]
        position_after_spikes_recording = pos_times > spike_times[-1]
        pos_xy_aligned_to_spikes_recording = pos_xy[
            ~position_before_spikes_recording & ~position_after_spikes_recording
        ]
        pos_times_aligned_to_spikes_recording = pos_times[
            ~position_before_spikes_recording & ~position_after_spikes_recording
        ]
        return (
            pos_xy_aligned_to_spikes_recording,
            pos_times_aligned_to_spikes_recording,
        )

    def confirm_all_30hz(self, pos_times: np.ndarray, gap_inds: np.ndarray) -> None:
        not30hz = np.round(pos_times[1:] - pos_times[:-1], 2) != np.round(
            self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S, 2
        )
        if len(gap_inds) > 0:
            not30hz[gap_inds] = 0
        if np.sum(not30hz) == 0:
            print("Data cleaning check: SUCCESSFUL, all position time frames are 30 Hz")
        else:
            print(
                f"Data cleaning check: WARNING, {np.sum(not30hz)} position time frames "
                "other than 30Hz"
            )
            print((pos_times[1:] - pos_times[:-1])[not30hz])

    # ----------------------------

    def calculate_velocity_info(self, rd=None) -> dict:
        velocity_info = dict()
        velocity_info["vel_times_s"] = self.calc_velocity_times(
            self.data["pos_times_s"]
        )
        velocity_info["vel_cm_per_s"] = self.calc_velocity(
            self.data["pos_xy_cm"]
        )
        (velocity_info["run_starts"], velocity_info["run_ends"]) = self.get_run_periods(
            velocity_info["vel_cm_per_s"], velocity_info["vel_times_s"], rd
        )
        return velocity_info

    @staticmethod
    def calc_velocity_times(pos_times: np.ndarray) -> np.ndarray:
        pos1 = pos_times[:-1]
        pos2 = pos_times[1:]
        return (pos1 + pos2) / 2

    def calc_velocity(
        self, pos_xy: np.ndarray) -> np.ndarray:
        distance = np.diff(pos_xy)
        # distance = np.sqrt(x_pos_diff ** 2 + y_pos_diff ** 2)
        velocity = distance / self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S
        return velocity

    def get_run_periods(
        self, velocity: np.ndarray, velocity_times: np.ndarray, rd=None
    ) -> tuple:
        # only use spikes from when animal was moving over running velocity threshold
        if np.any(np.isnan(velocity)):
            velocity_times = velocity_times[~np.isnan(velocity)]
            velocity = velocity[~np.isnan(velocity)]
        
        if not rd:
            run_boolean = abs(velocity) > self.params.velocity_run_threshold_cm_per_s
            run_starts, run_ends = utils.boolean_to_times(run_boolean, velocity_times)
        if rd=='+':
            run_boolean_pos = velocity > self.params.velocity_run_threshold_cm_per_s
            run_starts, run_ends = utils.boolean_to_times(run_boolean_pos, velocity_times)
        if rd=='-':
            run_boolean_pos = -velocity > self.params.velocity_run_threshold_cm_per_s
            run_starts, run_ends = utils.boolean_to_times(run_boolean_pos, velocity_times)

        return (run_starts, run_ends)

    # ----------------------------

    def calculate_place_fields(self, velocity_info) -> dict:
        """Calculate place fields"""
        place_field_data = dict()
        spike_ids = self.data["spike_ids"].copy()
        place_field_data["run_data"] = self.get_run_spike_and_pos_data(
            spike_ids,
            self.data["spike_times_s"],
            self.data["pos_xy_cm"],
            self.data["pos_times_s"],
            velocity_info,
        )

               
        place_field_data["spatial_grid"] = self.get_spatial_grid()
        
        place_field_data["position_histogram"] = self.calc_position_histogram(
            place_field_data["run_data"]["pos_xy_cm"], place_field_data["spatial_grid"]
        )
         
        place_field_data["spike_histograms"] = self.calc_spike_histograms(
            place_field_data["run_data"]["spike_times_s"],
            place_field_data["run_data"]["spike_ids"],
            self.data["pos_xy_cm"],
            self.data["pos_times_s"],
            place_field_data["spatial_grid"],
        )
        
        place_field_data["place_fields"] = self.calc_place_fields(
            place_field_data["position_histogram"],
            place_field_data["spike_histograms"],
            posterior=True,
        )
        
        place_field_data["place_fields_likelihood"] = self.calc_place_fields(
            place_field_data["position_histogram"],
            place_field_data["spike_histograms"],
            posterior=False,
        )
        
        place_field_data["mean_firing_rate_array"] = self.calc_run_mean_firing_rate(
            place_field_data["position_histogram"], place_field_data["spike_histograms"]
        )
        
        place_field_data["max_firing_rate_array"] = self.calc_max_tuning_curve_array(
            place_field_data["place_fields"]
        )
        # all cells are exc. place cells
        (
        place_field_data["place_cell_ids"],
        place_field_data["n_place_cells"],
        ) = (np.arange(0, self.data['n_cells'] ,1),
            self.data['n_cells'])
        return place_field_data
    
    def combine_place_field_data(self) -> dict:
        """Combine negative and positive running direction place field data"""
        place_field_data = dict()
        place_field_data["run_data"] = {'+': self.place_field_data_pos["run_data"], 
                                        '-': self.place_field_data_neg["run_data"]}
        place_field_data["spatial_grid"] = self.place_field_data_pos["spatial_grid"]
        place_field_data["position_histogram"] = {'+': self.place_field_data_pos["position_histogram"], 
                                                  '-': self.place_field_data_neg["position_histogram"]}
        place_field_data["spike_histograms"] = {'+': self.place_field_data_pos["spike_histograms"], 
                                                '-': self.place_field_data_neg["spike_histograms"]}
        place_field_data["place_fields"] = np.concatenate((self.place_field_data_pos["place_fields"],
                                                          self.place_field_data_neg["place_fields"]), axis=1)
        place_field_data["place_fields_likelihood"] = {'+': self.place_field_data_pos["place_fields_likelihood"], 
                                                       '-': self.place_field_data_neg["place_fields_likelihood"]}
        place_field_data["mean_firing_rate_array"] = np.mean([self.place_field_data_pos["mean_firing_rate_array"], 
                                                              self.place_field_data_neg["mean_firing_rate_array"]], 
                                                              axis=0)
        place_field_data["max_firing_rate_array"] =  {'+': self.place_field_data_pos["max_firing_rate_array"], 
                                                      '-': self.place_field_data_neg["max_firing_rate_array"]}
        place_field_data["place_cell_ids"] = self.place_field_data_pos["place_cell_ids"]
        place_field_data["n_place_cells"] = self.place_field_data_pos["n_place_cells"]
        return place_field_data

    @staticmethod
    def get_run_spike_and_pos_data(
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        velocity_info: dict,
    ) -> dict:
        run_data = dict()
        run_data["spike_times_s"] = np.array([])
        run_data["spike_ids"] = np.array([])
        run_x_pos = np.array([])
        for epoch in range(len(velocity_info["run_starts"])):
            start = velocity_info["run_starts"][epoch]
            end = velocity_info["run_ends"][epoch]
            # extract window indices
            spike_window_bool = utils.times_to_bool(spike_times, start, end)
            pos_window_bool = utils.times_to_bool(pos_times, start, end)
            # extract spikes and positions in this window
            window_spike_times = spike_times[spike_window_bool]
            window_spike_ids = spike_ids[spike_window_bool]
            window_x_pos = pos_xy[pos_window_bool]
            # append to list
            run_data["spike_times_s"] = np.append(
                run_data["spike_times_s"], window_spike_times
            )
            run_data["spike_ids"] = np.append(
                run_data["spike_ids"], window_spike_ids
            ).astype(int)
            run_x_pos = np.append(run_x_pos, window_x_pos)
        run_data["pos_xy_cm"] = np.array(run_x_pos)
        return run_data

    def get_spatial_grid(self):
        spatial_grid = dict()
        spatial_grid["x"] = np.linspace(0, self.data['well_locations'][1,0], self.params.n_bins_x + 1)
        # spatial_grid["y"] = np.linspace(0, 1, self.params.n_bins_y + 1)
        return spatial_grid

    def calc_position_histogram(
        self, run_pos_xy: np.ndarray, spatial_grid: dict
    ) -> np.ndarray:
        position_hist, _ = np.histogram(
            run_pos_xy,
            bins=spatial_grid["x"]
        )
        position_hist = (
            position_hist * self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S
        )
        return position_hist

    def calc_spike_histograms(
        self,
        spike_times: np.ndarray,
        spike_ids: np.ndarray,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        spatial_grid: dict,
    ) -> np.ndarray:
        spike_histograms = np.zeros(
            (self.data["n_cells"], self.params.n_bins_x)
        )
        for cell_id in range(self.data["n_cells"]):
            cell_spike_times = spike_times[spike_ids == cell_id]
            cell_spike_pos_xy = self.get_spike_positions(
                cell_spike_times, pos_xy, pos_times
            )
            if (len(cell_spike_times)) > 0:
                spike_hist, _ = np.histogram(
                    cell_spike_pos_xy,
                    bins=spatial_grid["x"]
                )
                spike_histograms[cell_id] = spike_hist
            else:
                spike_histograms[cell_id] = np.zeros(self.params.n_bins_x
                                                    )
        return spike_histograms

    def calc_place_fields(
        self,
        position_histogram: np.ndarray,
        spike_histograms: np.ndarray,
        posterior: bool = True,
    ) -> np.ndarray:
        place_fields = np.zeros(
            (self.data["n_cells"], self.params.n_bins_x)
        )
        for i in range(self.data["n_cells"]):
            place_fields[i] = self.calc_one_place_field(
                position_histogram, spike_histograms[i], posterior=posterior
            )
        return place_fields

    def calc_one_place_field(
        self,
        position_hist_s: np.ndarray,
        spike_hist: np.ndarray,
        posterior: bool = True,
    ) -> np.ndarray:
        if posterior:
            spike_hist_with_prior = (
                spike_hist + self.params.place_field_prior_alpha_s - 1
            )
            pos_hist_with_prior_s = (
                position_hist_s + self.params.place_field_prior_beta_s
            )
            place_field_raw = spike_hist_with_prior / pos_hist_with_prior_s
        else:
            place_field_raw = spike_hist / position_hist_s
            place_field_raw = np.nan_to_num(place_field_raw)
        if self.params.rotate_placefields:
            place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=0)
            place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=1)
        pf_gaussian_sd_bins = utils.cm_to_bins(self.params.place_field_gaussian_sd_cm)
        place_field_smoothed = gaussian_filter(
            place_field_raw, sigma=pf_gaussian_sd_bins
        )
        return place_field_smoothed

    def get_spike_positions(
        self, cell_spike_times: np.ndarray, pos_xy: np.ndarray, pos_times: np.ndarray
    ) -> np.ndarray:
        cell_spike_pos_xy = np.array(
            [
                self.find_position_during_spike(pos_xy, pos_times, time)
                for time in cell_spike_times
            ]
        )
        return cell_spike_pos_xy

    def find_position_during_spike(
        self, pos_xy: np.ndarray, pos_times: np.ndarray, spike_time: float
    ) -> np.ndarray:
        abs_diff = np.abs(pos_times - spike_time)
        min_diff = np.min(abs_diff)
        nearest_pos_xy = pos_xy[abs_diff == min_diff][0]
        return nearest_pos_xy

    def calc_run_mean_firing_rate(
        self, position_histogram: np.ndarray, spiking_histograms: np.ndarray
    ) -> np.ndarray:
        total_run_time = np.sum(position_histogram)
        total_spikes = np.sum(spiking_histograms, axis=1)
        mean_fr_array = total_spikes / total_run_time
        return mean_fr_array

    def calc_max_tuning_curve_array(self, place_fields: np.ndarray) -> np.ndarray:
        max_fr_array = np.max(place_fields, axis=1)
        return max_fr_array
