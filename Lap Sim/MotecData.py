# Author: Ludih
# Summary: This script is used to read and process data from 
#          Motec files to python in order to correlate/validate the Lap Sim.

import scipy.io as sio

class MotecData:

    def __init__(self, file_path):

        self.mat = sio.loadmat(
            file_path,
            struct_as_record=False,
            squeeze_me=True
        )

        self.channels = {}

        for name, obj in self.mat.items():

            if name.startswith("__"):
                continue

            if hasattr(obj, "_fieldnames"):

                self.channels[name] = {
                    "time": obj.Time,
                    "value": obj.Value,
                    "units": obj.Units
                }

    def getValue(self, channel):
        return self.channels[channel]["value"]

    def getTime(self, channel):
        return self.channels[channel]["time"]
    
    def getUnits(self, channel):
        return self.channels[channel]["units"]
