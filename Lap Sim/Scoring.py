class Scoring:
    # benchmarks from the official FSAE Electric 2026 (Michigan) event results
    T_MIN_ACCEL = 3.697        # [s] Univ of Wisconsin - Madison
    T_MIN_SKIDPAD = 4.782      # [s]
    T_MIN_AUTOCROSS = 43.937   # [s]
    T_MIN_ENDURANCE = 55.600 * 22   # [s] fastest single lap x 22 = 1223.2, Oregon State Univ lap 14
    T_MIN_EFF_LAP = 1312.281 / 22   # [s] official efficiency benchmark: fastest average lap = 59.649
    CO2_MIN_LAP = 1.847 / 22   # [kg/lap] Rensselaer Polytechnic Inst, lowest CO2 per lap = 0.0840
    CO2_MAX_LAP = 4.404 / 22   # [kg/lap] energy limit 20.02 kgCO2/100 km x 22 km = 0.2002, about 6.8 kWh total
    EFF_FACTOR_MAX = 0.797     # [-] best efficiency factor, Univ of Pittsburgh (not the lowest CO2 team)

    @staticmethod
    def getAccelScore(t_your):
        t_min = Scoring.T_MIN_ACCEL
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
        t_min = Scoring.T_MIN_AUTOCROSS
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
        # time score 0 to 250, plus 25 laps points for finishing all 22 laps
        t_min = Scoring.T_MIN_ENDURANCE
        t_max = t_min * 1.45

        if t_your < t_max:
            time_score = 250 * ((t_max / t_your) - 1) / ((t_max / t_min) - 1)
        else:
            time_score = 0
        if time_score > 250:
            time_score = 250
        return time_score + 25

    @staticmethod
    def getSkidpadScore(t_your):
        t_min = Scoring.T_MIN_SKIDPAD
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
        # t_your = your endurance time [s]
        # energy = your pack energy over the endurance [kWh], net of regen
        # everything per lap, like the official sheet, so a partial endurance still scores
        conversion_factor = 0.65                              # [kgCO2/kWh] electric
        t_min_lap = Scoring.T_MIN_EFF_LAP                     # [s] 59.649, official, not the fastest lap
        t_max_lap = t_min_lap * 1.45                          # [s] 86.491
        t_your_lap = t_your / track.n_laps                    # [s]
        co2_your_lap = energy * conversion_factor / track.n_laps   # [kg]

        # too slow or too much energy = zero points
        if t_your_lap > t_max_lap or co2_your_lap > Scoring.CO2_MAX_LAP:
            return 0.0

        # factor = relative lap time x relative CO2 per lap
        efficiency_factor_your = (t_min_lap / t_your_lap) * (Scoring.CO2_MIN_LAP / co2_your_lap)
        efficiency_factor_min = (t_min_lap / t_max_lap) * (Scoring.CO2_MIN_LAP / Scoring.CO2_MAX_LAP)   # 0.289
        score = 100 * (efficiency_factor_your - efficiency_factor_min) / (Scoring.EFF_FACTOR_MAX - efficiency_factor_min)
        return min(max(score, 0.0), 100.0)