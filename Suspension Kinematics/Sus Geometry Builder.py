# Author: Ludih
# Summary: This code builds the geometry of the suspension system based on the input points and lengths. It defines classes for wishbone assemblies, steering geometry, wheel points, bellcrank assembly, and corner assembly. 

import numpy as np
import Three_sphere_intersection as tsi

class wishbone_assembly:

  def __init__(self, in_fore_pt, in_aft_pt, out_pt):
    self.in_fore_pt = in_fore_pt
    self.in_aft_pt = in_aft_pt
    self.out_pt = out_pt

    #Define Vectors
    self.spread_vec = self.in_fore_pt-self.in_aft_pt
    self.fore_vec = self.out_pt-self.in_fore_pt
    self.aft_vec = self.in_aft_pt-self.out_pt

    #Spherical Coords
    self.L_fore = np.linalg.norm(self.fore_vec)
    self.L_aft = np.linalg.norm(self.aft_vec)
    self.L_spread = np.linalg.norm(self.spread_vec)

    self.phi_spread = np.arccos(self.spread_vec[2]/self.L_spread)
    self.theta_spread = np.arctan2(self.spread_vec[1],self.spread_vec[0])

    self.phi_fore = np.arccos(self.fore_vec[2]/self.L_fore)
    self.theta_fore = np.arctan2(self.fore_vec[1],self.fore_vec[0])

    self.phi_aft = np.arccos(self.aft_vec[2]/self.L_aft)
    self.theta_aft = np.arctan2(self.aft_vec[1],self.aft_vec[0])


  def get_bumped_out_pt(self, h_bump):

    phi_fore = np.arccos((self.fore_vec[2]+h_bump)/self.L_fore)
    phi_aft = np.arccos((self.aft_vec[2]+h_bump)/self.L_aft)

    #Solve for remaining coords
    A = self.L_fore*np.sin(phi_fore)
    B = self.L_aft*np.sin(phi_aft)
    C = self.L_spread*np.sin(self.phi_spread)

    theta_fore = self.theta_spread + np.arccos((A**2+C**2-B**2)/(2*A*-C))

    theta_aft = self.theta_spread - np.arccos((B**2+C**2-A**2)/(2*B*-C))

    #Updates
    out_pt = self.in_fore_pt + np.array([self.L_fore*np.sin(phi_fore)*np.cos(theta_fore),
                                           self.L_fore*np.sin(phi_fore)*np.sin(theta_fore),
                                           self.L_fore*np.cos(phi_fore)])
    return np.round(out_pt,5)

class steering_geometry:
  def __init__(self, lower_wishbone, upper_wishbone, in_tie_rod_pt, out_tie_rod_pt):
    self.lower_wishbone = lower_wishbone
    self.upper_wishbone = upper_wishbone
    self.in_tie_rod_pt = in_tie_rod_pt
    self.out_tie_rod_pt = out_tie_rod_pt

    self.L_tie_rod = np.linalg.norm(out_tie_rod_pt-in_tie_rod_pt)
    self.L_upper_steering_arm = np.linalg.norm(self.upper_wishbone.out_pt-out_tie_rod_pt)
    self.L_lower_steering_arm = np.linalg.norm(self.lower_wishbone.out_pt-out_tie_rod_pt)

class wheel_points:
  def __init__(self, lower_wishbone, upper_wishbone, steering_geometry, lower_wheel_pt, upper_wheel_pt, fore_wheel_pt, aft_wheel_pt):
    self.lower_wishbone = lower_wishbone
    self.upper_wishbone = upper_wishbone
    self.steering_geometry = steering_geometry
    self.lower_wheel_pt = lower_wheel_pt
    self.upper_wheel_pt = upper_wheel_pt
    self.fore_wheel_pt = fore_wheel_pt
    self.aft_wheel_pt = aft_wheel_pt

    #Lengths for upper wheel point
    self.L_upper_out2upper_wheel = np.linalg.norm(self.upper_wheel_pt-self.upper_wishbone.out_pt)
    self.L_out_tierod2upper_wheel = np.linalg.norm(self.upper_wheel_pt-self.steering_geometry.out_tie_rod_pt)
    self.L_lower_out2upper_wheel = np.linalg.norm(self.upper_wheel_pt-self.lower_wishbone.out_pt)

    #Lengths for lower wheel point
    self.L_upper_out2lower_wheel = np.linalg.norm(self.lower_wheel_pt-self.upper_wishbone.out_pt)
    self.L_out_tierod2lower_wheel = np.linalg.norm(self.lower_wheel_pt-self.steering_geometry.out_tierod_pt)
    self.L_lower_out2lower_wheel = np.linalg.norm(self.lower_wheel_pt-self.lower_wishbone.out_pt)

    #Lengths for fore wheel point
    self.L_upper_out2fore_wheel = np.linalg.norm(self.fore_wheel_pt-self.upper_wishbone.out_pt)
    self.L_out_tierod2fore_wheel = np.linalg.norm(self.fore_wheel_pt-self.steering_geometry.out_tierod_pt)
    self.L_lower_out2fore_wheel = np.linalg.norm(self.fore_wheel_pt-self.lower_wishbone.out_pt)

    #Lengths for aft wheel point
    self.L_upper_out2aft_wheel = np.linalg.norm(self.aft_wheel_pt-self.upper_wishbone.out_pt)
    self.L_out_tierod2aft_wheel = np.linalg.norm(self.aft_wheel_pt-self.steering_geometry.out_tierod_pt)
    self.L_lower_out2aft_wheel = np.linalg.norm(self.aft_wheel_pt-self.lower_wishbone.out_pt)


class bellcrank_assembly:
  def __init__(self, upper_wishbone, lower_pushrod_pt, upper_pushrod_pt, fore_bc_mount_pt,
               aft_bc_mount_pt, heave_damper_pt, roll_damper_pt):

    self.upper_wishbone = upper_wishbone
    self.lower_pushrod_pt = lower_pushrod_pt
    self.upper_pushrod_pt = upper_pushrod_pt
    self.fore_bc_mount_pt = fore_bc_mount_pt
    self.aft_bc_mount_pt = aft_bc_mount_pt
    self.heave_damper_pt = heave_damper_pt
    self.roll_damper_pt = roll_damper_pt

    #Lengths needed for lower pushrod point
    self.L_fore_wishbone2lower_pushrod = np.linalg.norm(self.lower_pushrod_pt-self.upper_wishbone.in_fore_pt)
    self.L_aft_wishbone2lower_pushrod = np.linalg.norm(self.lower_pushrod_pt-self.upper_wishbone.in_aft_pt)
    self.L_out_wishbone2lower_pushrod = np.linalg.norm(self.lower_pushrod_pt-self.upper_wishbone.out_pt)

    #Lengths needed for upper pushrod point
    self.L_pushrod = np.linalg.norm(self.lower_pushrod_pt-self.upper_pushrod_pt)
    self.L_fore_bc_mount2upper_pushrod = np.linalg.norm(self.upper_pushrod_pt-self.fore_bc_mount_pt)
    self.L_aft_bc_mount2upper_pushrod = np.linalg.norm(self.upper_pushrod_pt-self.aft_bc_mount_pt)

    #Lengths needed for heave damper point
    self.L_upper_pushrod2heave_damper = np.linalg.norm(self.heave_damper_pt-self.upper_pushrod_pt)
    self.L_fore_bc_mount2heave_damper = np.linalg.norm(self.heave_damper_pt-self.fore_bc_mount_pt)
    self.L_aft_bc_mount2heave_damper = np.linalg.norm(self.heave_damper_pt-self.aft_bc_mount_pt)

    #Lengths needed for roll damper point
    self.L_fore_bc_mount2roll_damper = np.linalg.norm(self.roll_damper_pt-self.fore_bc_mount_pt)
    self.L_heave_damper2roll_damper = np.linalg.norm(self.roll_damper_pt-self.heave_damper_pt)
    self.L_upper_pushrod2roll_damper = np.linalg.norm(self.roll_damper_pt-self.upper_pushrod_pt)


class corner_assembly:
  def __init__(self, lower_wishbone, upper_wishbone, steering_geometry, wheel_points, bellcrank_assembly):
    self.lower_wishbone = lower_wishbone
    self.upper_wishbone = upper_wishbone
    self.steering_geometry = steering_geometry
    self.wheel_points = wheel_points
    self.bellcrank_assembly = bellcrank_assembly

  def get_bumped_upper_out_pt(self, h_bump):

    #Solution #1 non-incremental steps
    lower_out_pt = self.lower_wishbone.get_bumped_out_pt(h_bump)
    length_knuckle = np.linalg.norm(self.lower_wishbone.out_pt-self.upper_wishbone.out_pt)

    upper_out_pt = tsi(lower_out_pt, self.upper_wishbone.in_fore_pt, self.upper_wishbone.in_aft_pt,length_knuckle,
                                 self.upper_wishbone.L_fore, self.upper_wishbone.L_aft)

    d1 = np.linalg.norm(self.upper_wishbone.out_pt-upper_out_pt[0])
    d2 = np.linalg.norm(self.upper_wishbone.out_pt-upper_out_pt[1])

    if d1<d2:
      upper_out_pt = upper_out_pt[0]
    else:
      upper_out_pt = upper_out_pt[1]

    return np.round(lower_out_pt,5) , np.round(upper_out_pt,5)

  def get_out_tie_rod_pt(self, steering_rack_input, h_bump):
      in_tie_rod_pt = self.steering_geometry.in_tie_rod_pt+np.array([0,steering_rack_input,0])

      lower_wishbone_out_pt,upper_wishbone_out_pt = self.get_bumped_upper_out_pt(h_bump)

      out_tie_rod_pt = tsi(in_tie_rod_pt, upper_wishbone_out_pt,
                                                 lower_wishbone_out_pt,
                                                 self.steering_geometry.L_tie_rod,
                                                 self.steering_geometry.L_upper_steering_arm,
                                                 self.steering_geometry.L_lower_steering_arm)

      d1 = np.linalg.norm(self.steering_geometry.out_tie_rod_pt-out_tie_rod_pt[0])
      d2 = np.linalg.norm(self.steering_geometry.out_tie_rod_pt-out_tie_rod_pt[1])

      if d1<d2:
        out_tie_rod_pt = out_tie_rod_pt[0]
      else:
        out_tie_rod_pt = out_tie_rod_pt[1]

      return np.round(out_tie_rod_pt,5)

  def get_wheel_pts(self, steering_rack_input, h_bump):

    out_tie_rod_pt = self.get_out_tie_rod_pt(steering_rack_input, h_bump)
    lower_wishbone_out_pt,upper_wishbone_out_pt = self.get_bumped_upper_out_pt(h_bump)

    upper_wheel_pt = tsi(out_tie_rod_pt, upper_wishbone_out_pt,
                                          lower_wishbone_out_pt, self.wheel_points.L_out_tierod2upper_wheel,
                                          self.wheel_points.L_upper_out2upper_wheel,
                                          self.wheel_points.L_lower_out2upper_wheel)


    d1 = np.linalg.norm(self.wheel_points.upper_wheel_pt-upper_wheel_pt[0])
    d2 = np.linalg.norm(self.wheel_points.upper_wheel_pt-upper_wheel_pt[1])

    if d1<d2:
      upper_wheel_pt = upper_wheel_pt[0]
    else:
      upper_wheel_pt = upper_wheel_pt[1]


    lower_wheel_pt = tsi(out_tie_rod_pt, upper_wishbone_out_pt,
                                          lower_wishbone_out_pt, self.wheel_points.L_out_tierod2lower_wheel,
                                          self.wheel_points.L_upper_out2lower_wheel,
                                          self.wheel_points.L_lower_out2lower_wheel)

    d1 = np.linalg.norm(self.wheel_points.lower_wheel_pt-lower_wheel_pt[0])
    d2 = np.linalg.norm(self.wheel_points.lower_wheel_pt-lower_wheel_pt[1])

    if d1<d2:
      lower_wheel_pt = lower_wheel_pt[0]
    else:
      lower_wheel_pt = lower_wheel_pt[1]

    fore_wheel_pt = tsi(out_tie_rod_pt, upper_wishbone_out_pt,
                                          lower_wishbone_out_pt, self.wheel_points.L_out_tierod2fore_wheel,
                                          self.wheel_points.L_upper_out2fore_wheel,
                                          self.wheel_points.L_lower_out2fore_wheel)

    d1 = np.linalg.norm(self.wheel_points.fore_wheel_pt-fore_wheel_pt[0])
    d2 = np.linalg.norm(self.wheel_points.fore_wheel_pt-fore_wheel_pt[1])

    if d1<d2:
      fore_wheel_pt = fore_wheel_pt[0]
    else:
      fore_wheel_pt = fore_wheel_pt[1]

    aft_wheel_pt = tsi(out_tie_rod_pt, upper_wishbone_out_pt,
                                          lower_wishbone_out_pt, self.wheel_points.L_out_tierod2aft_wheel,
                                          self.wheel_points.L_upper_out2aft_wheel,
                                          self.wheel_points.L_lower_out2aft_wheel)

    d1 = np.linalg.norm(self.wheel_points.aft_wheel_pt-aft_wheel_pt[0])
    d2 = np.linalg.norm(self.wheel_points.aft_wheel_pt-aft_wheel_pt[1])

    if d1<d2:
      aft_wheel_pt = aft_wheel_pt[0]
    else:
      aft_wheel_pt = aft_wheel_pt[1]

    return np.round(lower_wheel_pt,5), np.round(upper_wheel_pt,5), np.round(fore_wheel_pt,5), np.round(aft_wheel_pt,5)

  def get_bellcrank_pts(self, bellcrank_assembly, h_bump):
    upper_wishbone_in_fore_pt = self.upper_wishbone.in_fore_pt
    upper_wishbone_in_aft_pt = self.upper_wishbone.in_aft_pt
    upper_wishbone_out_pt = self.get_bumped_upper_out_pt(h_bump)[1]


    lower_pushrod_pt = tsi(upper_wishbone_in_fore_pt, upper_wishbone_in_aft_pt,
                            upper_wishbone_out_pt, self.bellcrank_assembly.L_fore_wishbone2lower_pushrod,
                             self.bellcrank_assembly.L_aft_wishbone2lower_pushrod,
                                 self.bellcrank_assembly.L_out_wishbone2lower_pushrod)

    plane_normal, plane_d = plane_from_three_points(upper_wishbone_out_pt, upper_wishbone_in_aft_pt, upper_wishbone_in_fore_pt)


    if is_point_above_plane(lower_pushrod_pt[0], plane_normal, plane_d):
      lower_pushrod_pt = lower_pushrod_pt[0]
    else:
      lower_pushrod_pt = lower_pushrod_pt[1]

    upper_pushrod_pt = tsi(lower_pushrod_pt, self.bellcrank_assembly.fore_bc_mount_pt,
                            self.bellcrank_assembly.aft_bc_mount_pt, self.bellcrank_assembly.L_pushrod,
                             self.bellcrank_assembly.L_fore_bc_mount2upper_pushrod,
                             self.bellcrank_assembly.L_aft_bc_mount2upper_pushrod)

    plane_normal, plane_d = plane_from_three_points(lower_pushrod_pt, self.bellcrank_assembly.fore_bc_mount_pt,
                                                    self.bellcrank_assembly.aft_bc_mount_pt)

    if is_point_above_plane(upper_pushrod_pt[0], plane_normal, plane_d):
      upper_pushrod_pt = upper_pushrod_pt[0]
    else:
      upper_pushrod_pt = upper_pushrod_pt[1]


    heave_damper_pt = tsi(upper_pushrod_pt, self.bellcrank_assembly.fore_bc_mount_pt,
                            self.bellcrank_assembly.aft_bc_mount_pt,
                              self.bellcrank_assembly.L_upper_pushrod2heave_damper,
                             self.bellcrank_assembly.L_fore_bc_mount2heave_damper,
                             self.bellcrank_assembly.L_aft_bc_mount2heave_damper)

    if abs(heave_damper_pt[0][1])< abs(heave_damper_pt[1][1]):
      heave_damper_pt = heave_damper_pt[0]
    else:
      heave_damper_pt = heave_damper_pt[1]

    roll_damper_pt = tsi(self.bellcrank_assembly.fore_bc_mount_pt, heave_damper_pt,
                            upper_pushrod_pt,self.L_fore_bc_mount2roll_damper,
                               self.L_heave_damper2roll_damper,
                               self.L_upper_pushrod2roll_damper)

    d1 = np.linalg.norm(self.bellcrank_assembly.roll_damper_pt-roll_damper_pt[0])
    d2 = np.linalg.norm(self.bellcrank_assembly.roll_damper_pt-roll_damper_pt[1])

    if d1<d2:
      roll_damper_pt = roll_damper_pt[0]
    else:
      roll_damper_pt = roll_damper_pt[1]

    return np.round(lower_pushrod_pt,5), np.round(upper_pushrod_pt,5), np.round(heave_damper_pt,5), np.round(roll_damper_pt,5)


  def tierod_shim2toe(self, tierod_length):
    steering_rack_input = 0
    h_bump = 0
    temp = self.steering_geometry.L_tie_rod
    self.steering_geometry.L_tie_rod = tierod_length
    out_tie_rod_pt = self.get_out_tie_rod_pt(steering_rack_input, h_bump)
    wheel_pts = self.get_wheel_pts(steering_rack_input, h_bump)
    self.steering_geometry.L_tie_rod = temp

    return out_tie_rod_pt, wheel_pts
    #Then you can define these as the new points if satisfied with static toe or just to check how much toe for shim thickness


  def get_all_points(self, steering_rack_input, h_bump):
    y=y



class get_parameter:
  def __init__(self, steering_geometry):
    self.steering_geometry = steering_geometry



