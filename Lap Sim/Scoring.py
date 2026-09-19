# Author : Anne-Sophie Nadeau
# Summary : Scoring algorithms  

# Benchmarks contains the times for different events to compare the lapsim time to. Fill in with competition results
BENCHMARKS = {
    "MICHIGAN_2024": dict(
        t_accel      = 3.642,   # [s] fastest accel
        t_skidpad    = 4.898,     # [s] fastest skidpad
        t_autocross  = 46.776,   # [s] fastest autocross
        t_endurance  = 1581.258,  # [s] fastest endurance (used by endurance AND efficiency)
        t_lap_min    = 71.875,    # [s/lap]  efficiency header "Minimum" time   
        co2_lap_min  = 0.0723,    # [kg/lap] efficiency header "Minimum" kg
        ef_min = 0.249,   # On top of the effiency facttor column in the minumum row
        ef_max = 0.845,    # On top of effiency facttor column in the maximum row
    ),

    "MICHIGAN_2025": dict(
        t_accel      = 3.821,  
        t_skidpad    = 4.933,     
        t_autocross  = 45.734,   
        t_endurance  = 1369.936,      
        t_lap_min    = 62.270,    
        co2_lap_min  = 0.0967,   
        ef_min = 0.333,  
        ef_max = 0.848, 
    ),

    "MICHIGAN_2026": dict(
        t_accel      = 3.697,
        t_skidpad    = 4.782,
        t_autocross  = 43.937,
        t_endurance  = 1312.281,
        t_lap_min    = 59.649,    
        co2_lap_min  = 0.0840, 
        ef_min = 0.289,    
        ef_max = 0.797, 
    ),
}

#######################################################

#________SELECT_EVENT_HERE_____#
B = BENCHMARKS["MICHIGAN_2024"]

#######################################################


class Scoring:
    
    @staticmethod   
    def getAccelScore(t_your):
        t_min = B["t_accel"] 
        t_max = t_min * 1.5

        if t_your < t_max:
            score = 95.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 4.5
        else:
            score = 4.5
        if score > 100:
            score = 100
        return score
    
    @staticmethod
    def getAutocrossScore(t_your):
        t_min = B["t_autocross"] 
        t_max = t_min * 1.45

        if t_your < t_max:
            score = 118.5 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1) + 6.5
        else:
            score = 6.5
        if score > 125:
            score = 125
        return score
    
    @staticmethod
    def getEnduranceScore(t_your):
        t_min = B["t_endurance"]
        t_max = t_min * 1.45

        if t_your < t_max:
            score = 250 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1)
        else:
            score = 0
        if score > 250:
            score = 250

        endurance_lap_score = 25

        score = endurance_lap_score + score
        return score
    
    @staticmethod
    def getSkidpadScore(t_your):
        t_min = B["t_skidpad"]
        t_max = t_min * 1.25

        if t_your < t_max:
            score = 71.5 * (((t_max / t_your) ** 2) - 1) / ((t_max / t_min) ** 2 - 1) + 3.5
        else:
            score = 3.5
        if score > 75:
            score = 75
        return score
    
    @staticmethod
    def getEfficiencyScore(t_your, energy, track):
        # t_your    = your event time
        # co2your   = mass of CO2 used by your car

        t_lap_min = B["t_lap_min"]         
        co2_lap_min = B["co2_lap_min"]   
        ef_min = B["ef_min"]        
        ef_max = B["ef_max"]         
        conversion_factor = 0.65    # [kgCO2/kWh] Electric
        co2your = energy * conversion_factor
        t_max = t_lap_min * 1.45
        t_lap_your = t_your / track.n_laps

        if t_lap_your > t_max:
            return 0

        efficiency_factor_your = (t_lap_min) / (t_lap_your) * (co2_lap_min) / (co2your / track.n_laps)
        efficiency_factor_min = ef_min
        efficiency_factor_max = ef_max
        score = 100 * (efficiency_factor_your - efficiency_factor_min) / (efficiency_factor_max - efficiency_factor_min)

        co2_lap_max = 20.02 / 100          # [kg/lap] EV baseline, 1 km nominal lap
        co2_lap_your = co2your / track.n_laps
        if t_lap_your > t_max or co2_lap_your > co2_lap_max:
            return 0
        if score > 100:
            score = 100
        if score<0:
            score = 0

        return score