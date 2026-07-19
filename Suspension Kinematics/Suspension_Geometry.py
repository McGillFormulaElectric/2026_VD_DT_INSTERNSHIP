# Author: Ludih
# Summary: This script holds the suspension points.

import numpy as np

###------------------Points------------------------###

class Front_Left:
    """
    Holds the suspension points for the front left corner of the car, in meters.
    All points are in the chassis frame, with x forward, y left, z up, origin at the center of the car on the ground.
    """
    def __init__(self):
        # Lower Wishbone
        self.lower_wishbone_fore   = np.array([0.7544, 0.2105, 0.1441])
        self.lower_wishbone_aft    = np.array([0.4436, 0.2502, 0.1401])
        self.lower_ball_joint      = np.array([0.762, 0.5012, 0.1143])

        # Upper Wishbone
        self.upper_wishbone_fore   = np.array([0.8144, 0.2318, 0.2955])
        self.upper_wishbone_aft    = np.array([0.507, 0.2597, 0.2945])
        self.upper_ball_joint      = np.array([0.7468, 0.4775, 0.2845])

        # Tie Rod
        self.tie_rod_chassis       = np.array([0.838, 0.179, 0.1458])
        self.tie_rod_outer         = np.array([0.8179, 0.5012, 0.1143])

        # Wheel Points
        self.upper_wheel_pt        = np.array([0.7625, 0.575, 0.406])
        self.contact_patch         = np.array([0.7625, 0.575, 0])     #Also lower wheel point, contact patch is the same as the lower wheel point
        self.fore_wheel_pt         = np.array([0.9655, 0.575, 0.203])
        self.aft_wheel_pt          = np.array([0.5595, 0.575, 0.203])

        # Pushrod and Bellcrank
        self.pushrod_outboard      = np.array([0.7467, 0.4522, 0.3014])
        self.pushrod_inboard       = np.array([0.7467, 0.174, 0.6181])
        self.bc_fore_mount         = np.array([0.7823, 0.1346, 0.5893])      
        self.bc_aft_mount          = np.array([0.6134, 0.1346, 0.5893])
        self.heave_damper          = np.array([0.7468, 0.1372, 0.6182])
        self.roll_damper           = np.array([0.6718, 0.1276, 0.6293])

    def mirror(self): #Mirrors the points across the y-axis to get the right side of the car, since most points are symmetric
        """
        Returns a new instance of the same class with all points mirrored across the y-axis.
        This is used to create the right side of the car from the left side.
        """
        mirrored = self.__class__.__new__(self.__class__)       #Make a new instance of the same class with nothing in it yet since init is not called
        for name, val in vars(self).items():       #Filling in the empty instance with the mirrored values of the original 
            mirrored_val = val.copy()            #Copy the array so that the original is not modified
            mirrored_val[1] *= -1               # negate y
            setattr(mirrored, name, mirrored_val)   #Give the mirrored value to the new mirrored instance witht he same name as the original
        return mirrored

class Rear_Left:
    """
    Holds the suspension points for the rear left corner of the car.
    All points are in the chassis frame, with x forward, y left, z up, origin at the center of the car on the ground.
    """
    def __init__(self):
        # Lower Wishbone
        self.lower_wishbone_fore   = np.array([-0.4943, 0.2626, 0.1496])
        self.lower_wishbone_aft    = np.array([-0.8755, 0.2098, 0.1504])
        self.lower_ball_joint      = np.array([-0.7417, 0.4775, 0.1092])

        # Upper Wishbone
        self.upper_wishbone_fore   = np.array([-0.5128, 0.2728, 0.293])
        self.upper_wishbone_aft    = np.array([-0.8706, 0.2376, 0.2927])
        self.upper_ball_joint      = np.array([-0.7417, 0.4775, 0.2845])

        # Tie Rod
        self.tie_rod_chassis       = np.array([-0.8685, 0.2393, 0.2603])
        self.tie_rod_outer         = np.array([-0.8331, 0.5232, 0.2603])

        # Wheel Points
        self.upper_wheel_pt        = np.array([-0.7625, 0.575, 0.406])
        self.contact_patch         = np.array([-0.7625, 0.575, 0])
        self.fore_wheel_pt         = np.array([-0.5595, 0.575, 0.203])
        self.aft_wheel_pt          = np.array([-0.9655, 0.575, 0.203])

        # Pushrod and Bellcrank
        self.pushrod_outboard      = np.array([-0.7417, 0.4507, 0.3012])
        self.pushrod_inboard       = np.array([-0.7417, 0.1651, 0.5461])
        self.bc_fore_mount         = np.array([-0.7099, 0.1372, 0.5207])
        self.bc_aft_mount          = np.array([-0.8750, 0.1372, 0.5207])
        self.heave_damper          = np.array([-0.7417, 0.1448, 0.5525])
        self.roll_damper           = np.array([-0.8166, 0.1300, 0.5607])

    def mirror(self):
        mirrored = self.__class__.__new__(self.__class__)
        for name, val in vars(self).items():
            mirrored_val = val.copy()
            mirrored_val[1] *= -1   # negate y
            setattr(mirrored, name, mirrored_val)
        return mirrored


#Building the four corners of the car, front left, front right, rear left, rear right

FL = Front_Left()
FR = FL.mirror()   #Points are mirrored across the y-axis
FR.roll_damper            = np.array([0.6718, -0.1276, 0.5492])   #Roll damper point is not symmetric, override and input it manually 
RL = Rear_Left()
RR = RL.mirror()
RR.roll_damper            = np.array([-0.8166, -0.130, 0.4806])


###------------------Lengths------------------------###

class Lengths: #Defines the lengths of the suspension components, used for kinematics calculations
  """
  Holds the lengths of the suspension components for a corner of the car.
  All lengths are in meters.
  """
  def __init__(self, c): # c = corner points instance (FL, FR, RL, RR)
    #Wishbones
    self.lower_fore = np.linalg.norm(c.lower_wishbone_fore-c.lower_ball_joint)
    self.lower_aft = np.linalg.norm(c.lower_wishbone_aft-c.lower_ball_joint)
    self.upper_fore = np.linalg.norm(c.upper_wishbone_fore-c.upper_ball_joint)
    self.upper_aft = np.linalg.norm(c.upper_wishbone_aft-c.upper_ball_joint)
    self.knuckle = np.linalg.norm(c.lower_ball_joint-c.upper_ball_joint)

    #Steering
    self.tie_rod = np.linalg.norm(c.tie_rod_chassis-c.tie_rod_outer)
    self.upper_steering_arm = np.linalg.norm(c.upper_ball_joint-c.tie_rod_outer)
    self.lower_steering_arm = np.linalg.norm(c.lower_ball_joint-c.tie_rod_outer)

    #upper wheel
    self.ubj2upper_wheel = np.linalg.norm(c.upper_ball_joint-c.upper_wheel_pt)
    self.out_tierod2upper_wheel = np.linalg.norm(c.tie_rod_outer-c.upper_wheel_pt)
    self.lbj2upper_wheel = np.linalg.norm(c.upper_wheel_pt-c.lower_ball_joint)

    #lower wheel
    self.ubj2lower_wheel = np.linalg.norm(c.contact_patch-c.upper_ball_joint)
    self.out_tierod2lower_wheel = np.linalg.norm(c.contact_patch-c.tie_rod_outer)
    self.lbj2lower_wheel = np.linalg.norm(c.contact_patch-c.lower_ball_joint)

    #fore wheel
    self.upper_out2fore_wheel = np.linalg.norm(c.fore_wheel_pt-c.upper_ball_joint)
    self.out_tierod2fore_wheel = np.linalg.norm(c.fore_wheel_pt-c.tie_rod_outer)
    self.lower_out2fore_wheel = np.linalg.norm(c.fore_wheel_pt-c.lower_ball_joint)

    #aft wheel
    self.upper_out2aft_wheel = np.linalg.norm(c.aft_wheel_pt-c.upper_ball_joint)
    self.out_tierod2aft_wheel = np.linalg.norm(c.aft_wheel_pt-c.tie_rod_outer)
    self.lower_out2aft_wheel = np.linalg.norm(c.aft_wheel_pt-c.lower_ball_joint)

    #Lengths needed for lower pushrod point
    self.fore_wishbone2lower_pushrod = np.linalg.norm(c.upper_wishbone_fore-c.pushrod_outboard)
    self.aft_wishbone2lower_pushrod = np.linalg.norm(c.upper_wishbone_aft-c.pushrod_outboard)
    self.ubj2lower_pushrod = np.linalg.norm(c.upper_ball_joint-c.pushrod_outboard)

    #Lengths needed for upper pushrod point
    self.pushrod = np.linalg.norm(c.pushrod_inboard-c.pushrod_outboard)
    self.bc_fore2upper_pushrod = np.linalg.norm(c.pushrod_inboard-c.bc_fore_mount)
    self.bc_aft2upper_pushrod = np.linalg.norm(c.pushrod_inboard-c.bc_aft_mount)

    #Lengths needed for heave damper point
    self.upper_pushrod2heave_damper = np.linalg.norm(c.heave_damper-c.pushrod_inboard)
    self.bc_fore2heave_damper = np.linalg.norm(c.heave_damper-c.bc_fore_mount)
    self.bc_aft2heave_damper = np.linalg.norm(c.heave_damper-c.bc_aft_mount)

    #Lengths needed for roll damper point
    self.upper_pushrod2roll_damper = np.linalg.norm(c.roll_damper-c.pushrod_inboard)
    self.bc_fore2roll_damper = np.linalg.norm(c.roll_damper-c.bc_fore_mount)
    self.bc_aft2roll_damper = np.linalg.norm(c.roll_damper-c.bc_aft_mount)

# Building the lengths for the front and rear of the car
F_len = Lengths(FL)   #Lengths are the same for the left and right side of the car currently, roll damper lengths can be different for left and right side of the car, but they are not currently.
R_len = Lengths(RL)


###------------------Vehicle dimensions------------------------###

# Derived from the contact patches so they stay consistent
#Used in ackermann calculations and roll motion ratio calculations in Kinematic_Plots.py and Kinematic_Outputs.py
wheelbase   = FL.contact_patch[0] - RL.contact_patch[0]
track_front = FL.contact_patch[1] - FR.contact_patch[1]
track_rear  = RL.contact_patch[1] - RR.contact_patch[1]