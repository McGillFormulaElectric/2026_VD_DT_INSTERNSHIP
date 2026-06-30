# Author : Anne-Sophie Nadeau
# Summary : Convert a Matlab file exported from Motec to a track map

class Scoring:
    def __init__(self, file_path):
        self.mat = mat.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        self.channels = {}

        for name, obj in self.mat.items():
            if name.startswith("__"):
                continue
            if hasattr(obj, "_fieldnames"):
                self.channels[name] = {"time": obj.Time, "value": obj.Value, "units": obj.Units}
                
    def accel_score(self, t_your):
        t_min = 3.94838  # [s] Michigan 2026
        t_max = t_min * 1.5

        if t_your < t_max:
            score = 95.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 4.5
        else:
            score = 4.5
        if score > 100:
            score = 100
        return score
    
    def autocross_score(self, t_your):
        t_min = 42.61350  # [s] Michigan 2026
        t_max = t_min * 1.45

        if t_your < t_max:
            score = 118.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 6.5
        else:
            score = 6.5
        if score > 125:
            score = 125
        return score
    
    def efficiency_score(self, t_your, t_co2min, co2min, energy, track):
        # t_your    = your event time
        # t_co2min  = time of CO2 min for event completion
        # co2min    = smallest mass of CO2 used by any competitor
        # co2your   = mass of CO2 used by your car
        
        t_min = 1420.853  # [s] Michigan 2026
        conversion_factor = 0.65  # [kgCO2/kWh] Electric
        co2your = energy * conversion_factor
        t_max = t_min * 1.45

        efficiency_factor_your = (t_min / track.number_laps) / (t_your / track.number_laps) * (co2min / track.number_laps) / (co2your / track.number_laps)
        efficiency_factor_min = (t_min / track.number_laps) / (t_max / track.number_laps) * (co2min / track.number_laps) / (track.s[-1] / 1000 * 20.02 / 100)
        efficiency_factor_max = (t_min / track.number_laps) / (t_co2min / track.number_laps) * (co2min / track.number_laps) / (co2min / track.number_laps)
        score = 100 * (efficiency_factor_your - efficiency_factor_min) / (efficiency_factor_max - efficiency_factor_min)

        return score