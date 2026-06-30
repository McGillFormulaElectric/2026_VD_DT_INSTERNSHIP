# Author : Anne-Sophie Nadeau
# Summary : Convert a Matlab file exported from Motec to a track map

class Scoring:
    def __init__(self, Results, track_endurance):
        track = track_endurance

        # score every iteration
        for k in range(len(Results["Energy (kWh)"])):
            skidpad    = self.getSkidpadScore(Results["Skid Pad (s)"][k])
            accel      = self.getAccelScore(Results["Acceleration (s)"][k])
            autox      = self.getAutocrossScore(Results["Autocross (s)"][k])
            endurance  = self.getEnduranceScore(Results["Endurance Time (s)"][k])
            efficiency = self.getEfficiencyScore(Results["Endurance Time (s)"][k], Results["Energy (kWh)"][k], track)
            total      = skidpad + accel + autox + endurance + efficiency

            Results["Skid Pad Score"].append(skidpad)
            Results["Accel Score"].append(accel)
            Results["AutoX Score"].append(autox)
            Results["Endurance Score"].append(endurance)
            Results["Efficiency Score"].append(efficiency)
            Results["Competition Points"].append(total)

        self.Results = Results
                
    def getAccelScore(self, t_your):
        t_min = 3.94838  # [s] Michigan 2026
        t_max = t_min * 1.5

        if t_your < t_max:
            score = 95.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 4.5
        else:
            score = 4.5
        if score > 100:
            score = 100
        return score
    
    def getAutocrossScore(self, t_your):
        t_min = 42.61350  # [s] Michigan 2026
        t_max = t_min * 1.45

        if t_your < t_max:
            score = 118.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 6.5
        else:
            score = 6.5
        if score > 125:
            score = 125
        return score
    
    def getEnduranceScore(self, t_your):
        t_min = 1420.853  # [s] Michigan 2026
        t_max = t_min * 1.45

        if t_your < t_max:
            score = 250 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1)
        else:
            score = 0
        if score > 275:
            score = 275
        return score
    
    def getSkidpadScore(self, t_your):
        t_min = 4.798  # [s] Michigan 2026
        t_max = t_min * 1.25

        if t_your < t_max:
            score = 71.5 * (((t_max / t_your) ** 2) - 1) / ((t_max / t_min) ** 2 - 1) + 3.5
        else:
            score = 3.5
        if score > 75:
            score = 75
        return score
    
    def getEfficiencyScore(self, t_your, energy, track):
        # t_your    = your event time
        # co2your   = mass of CO2 used by your car

        t_min = 1420.853            # [s] Michigan 2026 endurance benchmark
        co2_min = 1.591             # [kg] smallest CO2 of any competitor, Michigan 2026
        t_co2min = 1872.046         # [s] endurance time of the lowest-CO2 competitor, Michigan 2026
        conversion_factor = 0.65    # [kgCO2/kWh] Electric
        co2your = energy * conversion_factor
        t_max = t_min * 1.45

        efficiency_factor_your = (t_min / track.number_laps) / (t_your / track.number_laps) * (co2_min / track.number_laps) / (co2your / track.number_laps)
        efficiency_factor_min = (t_min / track.number_laps) / (t_max / track.number_laps) * (co2_min / track.number_laps) / (track.s[-1] / 1000 * 20.02 / 100)
        efficiency_factor_max = (t_min / track.number_laps) / (t_co2min / track.number_laps) * (co2_min / track.number_laps) / (co2_min / track.number_laps)
        score = 100 * (efficiency_factor_your - efficiency_factor_min) / (efficiency_factor_max - efficiency_factor_min)

        return score