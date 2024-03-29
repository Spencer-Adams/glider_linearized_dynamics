import json
import math
import numpy as np
from scipy.linalg import eig
# from tabulate import tabulate
np.set_printoptions(precision=8)


class aircraft_eigens:
    """This class currently works for a 2DOF 2nd order differential equation. Will be expanded to handle N degrees of freedom hopefully."""
    def __init__(self, json_file):
        self.json_file = json_file
        self.load_json()
        self.calc_cbar_w()
        self.assemble_B_matrix()
        self.assemble_A_matrix()
        self.is_underdamped = False
        self.is_overdamped = False
        self.is_critically_damped = False


    def load_json(self):
        """This function pulls in all the input values from the json"""
        with open(self.json_file, 'r') as json_handle:
            input_vals = json.load(json_handle)
            self.name = input_vals["aircraft"]["name"]
            self.launch_kinetic_energy = input_vals["aircraft"]["launch_kinetic_energy[ft-lbf]"]
            self.wing_area = input_vals["aircraft"]["wing_area[ft^2]"]
            self.wing_span = input_vals["aircraft"]["wing_span[ft]"]
            self.weight = input_vals["aircraft"]["weight[lbf]"]
            self.Ixx = input_vals["aircraft"]["Ixx[slugs*ft^2]"]
            self.Iyy = input_vals["aircraft"]["Iyy[slugs*ft^2]"]
            self.Izz = input_vals["aircraft"]["Izz[slugs*ft^2]"]
            self.Ixy = input_vals["aircraft"]["Ixy[slugs*ft^2]"]
            self.Ixz = input_vals["aircraft"]["Ixz[slugs*ft^2]"]
            self.Iyz = input_vals["aircraft"]["Iyz[slugs*ft^2]"]
            self.density = input_vals["analysis"]["density[slugs/ft^3]"]
            self.CL_0 = input_vals["aerodynamics"]["CL"]["0"]
            self.CL_alpha = input_vals["aerodynamics"]["CL"]["alpha"]
            self.CL_qbar = input_vals["aerodynamics"]["CL"]["qbar"]
            self.CL_alpha_hat = input_vals["aerodynamics"]["CL"]["alpha_hat"]
            self.CY_beta = input_vals["aerodynamics"]["CY"]["beta"]
            self.CY_pbar = input_vals["aerodynamics"]["CY"]["pbar"]
            self.CY_rbar = input_vals["aerodynamics"]["CY"]["rbar"]
            self.CD_L0 = input_vals["aerodynamics"]["CD"]["L0"]
            self.CD_L = input_vals["aerodynamics"]["CD"]["L"]
            self.CD_L2 = input_vals["aerodynamics"]["CD"]["L2"]
            self.CD_qbar = input_vals["aerodynamics"]["CD"]["qbar"]
            self.Cl_beta = input_vals["aerodynamics"]["Cl"]["beta"]
            self.Cl_pbar = input_vals["aerodynamics"]["Cl"]["pbar"]
            self.Cl_rbar = input_vals["aerodynamics"]["Cl"]["rbar"]
            self.Cm_0 = input_vals["aerodynamics"]["Cm"]["0"]
            self.Cm_alpha = input_vals["aerodynamics"]["Cm"]["alpha"]
            self.Cm_qbar = input_vals["aerodynamics"]["Cm"]["qbar"]
            self.Cm_alpha_hat = input_vals["aerodynamics"]["Cm"]["alpha_hat"]
            self.Cn_beta = input_vals["aerodynamics"]["Cn"]["beta"]
            self.Cn_pbar = input_vals["aerodynamics"]["Cn"]["pbar"]
            self.Cn_rbar = input_vals["aerodynamics"]["Cn"]["rbar"]
            self.gravity = 32.17 ###[ft/s^2]
            self.theta_initial = 0.0 ###[deg]

   
    def calc_cbar_w(self):
        self.cbar_w = self.wing_area/self.wing_span


    def calc_R_rho_x(self):
        """This function is a non-dimensional mass component in the x direction"""
        numerator = 4*self.weight/self.gravity
        denominator = self.density*self.wing_area*self.cbar_w
        self.R_rho_x = numerator/denominator

    
    def calc_R_rho_y(self):
        """This function is a non-dimensional mass component in the x direction"""
        numerator = 4*self.weight/self.gravity
        denominator = self.density*self.wing_area*self.wing_span
        self.R_rho_y = numerator/denominator


    def calc_R_yy(self):
        numerator = 8*self.Iyy
        denominator = self.density*self.wing_area*self.cbar_w**3
        self.R_yy = numerator/denominator


    def calc_launch_velocity(self):
        self.Vo = np.sqrt(self.weight/(0.5*self.density*self.CL_0*self.wing_area))
        # self.Vo = np.sqrt(2*self.launch_kinetic_energy*self.gravity/self.weight)
    

    def calc_R_g_x(self):
        self.R_g_x = self.gravity*self.cbar_w/(2*self.Vo**2)


    def calc_R_g_y(self):
        self.R_g_y = self.gravity*self.wing_span/(2*self.Vo**2)


    def calc_CD_mu_hat(self):
        """This function calculates non-dimensional CD,mu"""
        self.CD_mu_hat = 0.0

    
    def calc_CD_alpha_hat(self):
        self.CD_alpha_hat = 0.0


    def calc_CL_mu_hat(self):
        self.CL_mu_hat = 0.0
    
    
    def calc_Cm_mu_hat(self):
        self.Cm_mu_hat = 0.0


    def calc_CD_alpha(self):
        """Using equation 10.59 in the eng handbook"""
        self.CD_alpha = self.CD_L*self.CL_alpha + 2*self.CD_L2*self.CL_0*self.CL_alpha


    def calc_CDo(self):
        """Using equation 10.48 in the eng handbook"""
        self.CDo = self.CD_L0 + self.CD_L*self.CL_0 + self.CD_L2*self.CL_0**2


    def assemble_B_matrix(self):
        """This assembles the B_matrix for the eigen problem"""
        B_array = np.zeros((6,6))
        self.calc_R_rho_x()
        self.calc_R_rho_y()
        self.calc_CD_mu_hat()
        self.calc_CD_alpha_hat()
        self.calc_CL_mu_hat()
        self.calc_Cm_mu_hat()
        self.calc_R_yy()
        B_array[0,0] = self.R_rho_x + self.CD_mu_hat
        B_array[0,1] = self.CD_alpha_hat
        B_array[1,0] = self.CL_mu_hat
        B_array[1,1] = self.R_rho_x + self.CL_alpha_hat
        B_array[2,0] = -self.Cm_mu_hat
        B_array[2,1] = -self.Cm_alpha_hat
        B_array[2,2] = self.R_yy
        B_array[3,3] = 1
        B_array[4,4] = 1
        B_array[5,5] = 1
        self.B_matrix = B_array

    
    def assemble_A_matrix(self):
        """This function assembles the A matrix using eq 10.77 in the aero handbook"""
        A_array = np.zeros((6,6))
        self.calc_launch_velocity()
        self.calc_R_g_x()
        self.calc_R_g_y()
        self.calc_CDo()
        self.calc_CD_alpha()
        A_array[0,0] = -2*self.CDo
        A_array[0,1] = self.CL_0-self.CD_alpha
        A_array[0,2] = -self.CD_qbar
        A_array[0,5] = -self.R_rho_x*self.R_g_x*np.cos(self.theta_initial)
        A_array[1,0] = -2*self.CL_0
        A_array[1,1] = -self.CL_alpha-self.CDo
        A_array[1,2] = -self.CL_qbar + self.R_rho_x
        A_array[1,5] = -self.R_rho_x*self.R_g_x*np.sin(self.theta_initial)
        A_array[2,0] = 2*self.Cm_0
        A_array[2,1] = self.Cm_alpha
        A_array[2,2] = self.Cm_qbar
        A_array[3,0] = np.cos(self.theta_initial)
        A_array[3,1] = np.sin(self.theta_initial)
        A_array[3,5] = -np.sin(self.theta_initial)
        A_array[4,0] = -np.sin(self.theta_initial)
        A_array[4,1] = np.cos(self.theta_initial)
        A_array[4,5] = -np.cos(self.theta_initial)
        A_array[5,2] = 1
        self.A_matrix = A_array


    def assemble_identity_matrix(self, B_matrix):
        """The identity matrix is Binverse multiplied by B"""
        B_matrix_inverse = invert_matrix(B_matrix)
        Identity = np.matmul(B_matrix_inverse,B_matrix)
        return Identity
    

    def assemble_C_matrix(self, A_matrix, B_matrix):
        """This function multiplies the inverse of the B_matrix by the A_matrix"""
        B_inverse = invert_matrix(B_matrix)
        C_matrix = np.matmul(B_inverse,A_matrix)
        return C_matrix
    

    def get_C_matrix(self):
        """runs the eigen solver code in order"""
        self.assemble_A_matrix()
        self.assemble_B_matrix()
        matrix_I = self.assemble_identity_matrix(self.B_matrix)
        self.C_matrix = self.assemble_C_matrix(self.A_matrix, self.B_matrix)
    

    def get_eigens(self):
        self.eigvals, self.eigvecs = eig(self.C_matrix)
        self.real_eigvals = np.real(self.eigvals)
        self.imag_eigvals = np.imag(self.eigvals)
        self.real_eigvecs = np.real(self.eigvecs)
        self.imag_eigvecs = np.imag(self.eigvecs)


    def calc_damping_rate(self, eigenvals):
        """Takes in an eigen value pair (one real or two complex conjugates)"""
        sigma = -np.real(eigenvals)
        return sigma
    

    def calc_99_percent_damp_time(self, sigma):
        """calculates time to damp out 99% of the way"""
        damp_time_99 = -np.log(0.01)/sigma
        return damp_time_99
    

    def calc_double_time(self, sigma):
        """This calculates the time to double the mode amplitude in a divergent mode"""
        double_time = -np.log(2)/sigma
        return double_time


    def calc_damped_natural_freq(self, eigenvals):
        """returns the magnitude of the imaginary part of eigenvals"""
        damped_nat_freq = np.imag(eigenvals)
        damped_nat_freq = abs(damped_nat_freq)
        return damped_nat_freq
    

    def calc_period(self, damped_natural_freq):
        """returns the period of oscillation on an underdamped system"""
        period = 2*np.pi/(damped_natural_freq)
        return period
    

    def calc_undamped_natural_freq(eigs):
        """This calculates the undamped natural frequency as a function of lambda 1 and lambda 2"""
        undamped_natural_frequency = np.sqrt(eigs[0]*eigs[1])
        return undamped_natural_frequency
    

    def calc_damping_ratio(self, lambda1, lambda2):
        """Use if there's a complex conjugate pair"""
        zeta = (-(lambda1+lambda2)/(2*np.real(np.sqrt(lambda1*lambda2))))
        return zeta
    

    def calc_is_underdamped(self, zeta):
        """Determines if underdamped using zeta"""
        is_underdamped = False
        is_overdamped = False
        is_critically_damped = False
        if zeta > 1:
            is_underdamped = False
            is_overdamped = True
            is_critically_damped = False
        elif zeta == 1:
            is_underdamped = False
            is_overdamped = False
            is_critically_damped = True
        else:
            is_underdamped = True
            is_overdamped = False
            is_critically_damped = False
        self.is_underdamped = is_underdamped
        self.is_overdamped = is_overdamped
        self.is_critically_damped = is_critically_damped  


    def calc_amplitude(self, real_eig_vector, imag_eig_vector):
        """This calculates the amplitude of the oscillation if the mode is underdamped, otherwise it tells the user that this particular mode is not underdamped"""
        amplitude = np.sqrt(real_eig_vector**2 + imag_eig_vector**2)
        return amplitude


    def calc_phase_angle(self, real_eig_vector, imag_eig_vector):
        """This calculates the phase angle of the oscillation if the mode is underdamped, otherwise it tells the user that this particular mode is not underdamped"""
        phase_angle = math.atan2(imag_eig_vector,real_eig_vector)
        return phase_angle
    

    def calc_eigen_properties(self):
        amplitudes = np.zeros((len(self.eigvals), len(self.eigvals)))
        phase_angles = np.zeros((len(self.eigvals), len(self.eigvals)))
        sigmas = np.zeros((len(self.eigvals)))
        damp_time_99 = []
        double_time = []
        damped_natural_freq = []
        periods = []
        is_underdamped = []
        for i in range(len(self.eigvecs)):
            sigmas[i] = self.calc_damping_rate(self.eigvals[i])
            if np.real(self.eigvals[i]) < 0:
                this_99_damp = self.calc_99_percent_damp_time(sigmas[i])
                damp_time_99.append(this_99_damp)
                double_time.append("NA (real part of eigenvalue is not positive)")
            elif np.real(self.eigvals[i]) > 0:
                damp_time_99.append("NA (real part of eigenvalue is not negative)")
                this_double_time = self.calc_double_time(sigmas[i])
                double_time.append(this_double_time)
            else:
                damp_time_99.append("NA (real part of eigenvalue is not negative)")
                double_time.append("NA (real part of eigenvalue is not positive)")
            if np.abs(np.imag(self.eigvals[i])) > 0:
                this_damped_nat_freq = self.calc_damped_natural_freq(self.eigvals[i])
                damped_natural_freq.append(this_damped_nat_freq)
                this_period = self.calc_period(this_damped_nat_freq)
                periods.append(this_period)
            else:
                damped_natural_freq.append("NA (There is no imaginary portion of the eigenvalue, so there is no oscillatory motion)")
                periods.append("NA (There is no imaginary portion of the eigenvalue, so there is no oscillatory motion)")
            for j in range(len(self.eigvecs)):
                phase_angles[j,i] = self.calc_phase_angle(self.real_eigvecs[j,i],self.imag_eigvecs[j,i])
                amplitudes[j,i] = self.calc_amplitude(self.real_eigvecs[j,i],self.imag_eigvecs[j,i])
        self.phase_angles = phase_angles
        self.amplitudes = amplitudes
        self.sigmas = sigmas
        self.calc_phase_angles_degrees()
        self.damp_time_99 = np.array(damp_time_99)
        self.double_time = np.array(double_time)
        self.damped_natural_frequency = np.array(damped_natural_freq)
        self.periods = np.array(periods)


    def assemble_modes(self):
        modes = []
        checked = set()  # To keep track of visited eigenvalues

        for i in range(len(self.eigvals)):
            # Check if the eigenvalue has already been visited
            if i in checked:
                continue
            # Add current eigenvalue to the mode
            mode = [self.eigvals[i]]
            # Check if it's a real eigenvalue (0 + 0j)
            if np.isclose(self.eigvals[i], 0):
                modes.append(mode)
                continue
            # Check if it has a complex conjugate pair
            for j in range(i + 1, len(self.eigvals)):
                if np.isclose(self.eigvals[i], np.conj(self.eigvals[j])):
                    # Add complex conjugate pair to the mode
                    mode.append(self.eigvals[j])
                    checked.add(j)  # Mark complex conjugate as visited
            # Add assembled mode to the list of modes
            modes.append(mode)
        self.modes = np.array(modes, dtype=object)  # specify dtype=object to handle different shapes


    def calc_mode_values(self):
        """This function determines if a longitudinal mode is short period, long period, or translational.
        Then it calculates the damping rate, 99% damping time, damped frequency, and period of said mode by calling calc_mode_vals_for_category"""

        translational_modes = []
        short_period_modes = []
        phugoid_modes = []

        # Ensure modes array is created with dtype=object
        modes = np.array(self.modes, dtype=object)

        # Separate out the translational modes
        for mode in modes:
            if np.allclose(mode, 0):
                translational_modes.append(mode)

        # Identify phugoid mode as the mode with the smallest damping rate among non-translational modes
        non_translational_modes = [mode for mode in modes if mode not in translational_modes]
        if non_translational_modes:
            phugoid_modes.append(min(non_translational_modes, key=lambda x: np.abs(np.real(x[0]))))

        # All other modes are short period modes
        short_period_modes.extend(mode for mode in modes if mode not in translational_modes + phugoid_modes)

        # Calculate values for each mode category
        self.calc_mode_values_for_category(translational_modes, 'translational')
        self.calc_mode_values_for_category(short_period_modes, 'short_period')
        self.calc_mode_values_for_category(phugoid_modes, 'phugoid')
          

    def calc_mode_values_for_category(self, modes, category):
        """Calculate damping rate, 99% damping time, damped frequency, and period for modes in a specific category"""

        for mode in modes:
            if category == 'translational':
                # print(f"{category.capitalize()} mode:", mode, "\n")
                print('__________________________________________________________________________________________________')
                self.print_formatting(category + " Mode", mode)
                print("")
            else:
                damping_rate = self.calc_damping_rate(mode)
                damp_time_99 = self.calc_99_percent_damp_time(damping_rate)
                damped_freq = self.calc_damped_natural_freq(mode)
                period = self.calc_period(damped_freq)

                # Print or store the calculated values as needed
                print('__________________________________________________________________________________________________')
                self.print_formatting(category + " Mode ", mode)
                print("")
                self.print_formatting(category + " Damping rate, sigma:", damping_rate)
                self.print_formatting(category + " 99% Damping time:", damp_time_99)
                self.print_formatting(category + " Damped natural frequency:", damped_freq)
                self.print_formatting(category + " Period:", period)
                print("")
        
        
    def print_formatting(self, start_of_string, value):
        """This assists in formatting the print function"""
        #checks if the value passed in is a string, changes print formatting accordingly
        if isinstance(value, str):
            print(start_of_string, "{:s}".format(value))
        #checks if the value passed in is a float, changes print formatting accordingly
        elif isinstance(value, (int, float)):
            print(start_of_string, "{: 18.12f}".format(value))
        #checks if the value passed in is complex, changes print formatting accordingly
        elif isinstance(value, complex):
            print(start_of_string, "{: 18.12f}".format(value.real), "+", "{: 8.12f}".format(value.imag), "j")
        #checks if the value passed in is a tuple, changes print formatting accordingly
        elif isinstance(value, (list, tuple, np.ndarray)):
            print(start_of_string, value)
        else:
            raise ValueError("Unsupported type for formatting: {}".format(type(value)))

    
    def print_eigens(self):
        """Prints the eigen values and vectors all pretty in the terminal"""
        for i in range(len(self.eigvals)):
            print('__________________________________________________________________________________________________')
            self.print_formatting("Dimensionless EigenValue = ", self.eigvals[i])
            self.print_formatting("Damping rate, σ [1/sec] = ", self.sigmas[i])
            self.print_formatting("99 Percent Damping time [sec] = ",self.damp_time_99[i])
            self.print_formatting("Time to double [sec] = ",self.double_time[i])
            self.print_formatting("Damped Natural Frequency, ωd   = ",self.damped_natural_frequency[i])
            self.print_formatting("Period, τ [sec] = ", self.periods[i])
            print("")
            print("Dimensionless EigenVector:   Real             Imaginary           Amplitude       Phase Angle[deg]")
            print('              ▲µ     {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[0,i],self.imag_eigvecs[0,i],self.amplitudes[0,i],self.phase_angles_degrees[0,i]))
            print('              ▲α     {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[1,i],self.imag_eigvecs[1,i],self.amplitudes[1,i],self.phase_angles_degrees[1,i]))
            print('              ▲q     {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[2,i],self.imag_eigvecs[2,i],self.amplitudes[2,i],self.phase_angles_degrees[2,i]))
            print('              ▲ξx     {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[3,i],self.imag_eigvecs[3,i],self.amplitudes[3,i],self.phase_angles_degrees[3,i]))
            print('              ▲ξz    {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[4,i],self.imag_eigvecs[4,i],self.amplitudes[4,i],self.phase_angles_degrees[4,i]))
            print('              ▲Θ     {: 18.12f} {: 18.12f} {: 18.12f} {: 18.12f}'.format(self.real_eigvecs[5,i],self.imag_eigvecs[5,i],self.amplitudes[5,i],self.phase_angles_degrees[5,i]))
    

    def hard_code_calc_mode_vals(self):
        # first short period
        self.first_short_period_damping_rate = self.calc_damping_rate(self.modes[2])
        self.first_short_period_99_damp_time = self.calc_99_percent_damp_time(self.first_short_period_damping_rate)
        self.first_short_period_damped_frequency = self.calc_damped_natural_freq(self.modes[2])
        self.first_short_period_period = self.calc_period(self.first_short_period_damped_frequency)
        # second short period
        self.sec_short_period_damping_rate = self.calc_damping_rate(self.modes[3])
        self.sec_short_period_99_damp_time = self.calc_99_percent_damp_time(self.sec_short_period_damping_rate)
        self.sec_short_period_damped_frequency = self.calc_damped_natural_freq(self.modes[3])
        self.sec_short_period_period = self.calc_period(self.sec_short_period_damped_frequency)
        # phugoid
        self.phugoid_damping_rate = self.calc_damping_rate(self.modes[4][0])
        self.phugoid_99_damp_time = self.calc_99_percent_damp_time(self.phugoid_damping_rate)
        self.phugoid_damped_frequency = self.calc_damped_natural_freq(self.modes[4][0])
        self.phugoid_period = self.calc_period(self.phugoid_damped_frequency)     
    
    
    def calc_phase_angles_degrees(self):
        """This converts the phase angle matrix into degrees"""
        self.phase_angles_degrees = np.degrees(self.phase_angles)


def invert_matrix(matrix):
    """This function takes in a matrix and inverts it"""
    inverse = np.linalg.inv(matrix)
    return inverse


if __name__ == "__main__":
    eigen_object = aircraft_eigens("0000.json")

    eigen_object.get_C_matrix()
    eigen_object.get_eigens()
    eigen_object.calc_eigen_properties()
    eigen_object.assemble_modes()
    eigen_object.hard_code_calc_mode_vals()
    eigen_object.print_eigens()
    eigen_object.calc_mode_values()

    # print(eigen_object.Vo)


    
    
