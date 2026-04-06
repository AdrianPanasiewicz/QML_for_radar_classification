from Data.Primitives.environment_classes import Drone, Radar, Context
from Data.Primitives.noise_models import AdditiveWhiteGaussianNoise
import numpy as np
import scipy
import sympy
from matplotlib import pyplot as plt

class SyntheticSignalGenerator:
    def __init__(self, drone: Drone, radar: Radar, noise_model=None):
        self.drone = drone
        self.radar = radar
        self.noise_model = noise_model

        self.base_Psi_f = self._prepare_Psi_functions()

    def _prepare_sympy_Psi_function(self):
        theta, Phi_p, f_rot, t, lamb, A_r, f_c, R, V_rad, L_1, L_2 = sympy.symbols("θ Φ_p f_rot t λ A_r f_c, R, V_rad, L_1, L_2")
        n, N = sympy.symbols("n N", integer=True, positive=True)

        alpha = sympy.sin(sympy.Abs(theta) + Phi_p) + sympy.sin(sympy.Abs(theta) - Phi_p)
        beta  = sympy.sign(theta) * alpha
        Omega_n = 2 * sympy.pi * (f_rot * t + n / N)
        gamma_n = 4 * sympy.pi / lamb * sympy.cos(theta) * sympy.sin(Omega_n)

        first_part = A_r * sympy.exp(sympy.I*(2*sympy.pi*f_c*t - 4*sympy.pi/lamb*(R+V_rad*t)))

        # alpha, beta, Omega_n, gamma_n = sympy.symbols("α β Ω_n γ_n") # Just for verifying the equation

        element_second_part = (alpha + beta * sympy.cos(Omega_n)) * sympy.exp(-sympy.I*(L_1+L_2)/2) * sympy.sinc((L_2-L_1)/2*gamma_n)

        Psi = first_part + sympy.Sum(element_second_part, (n,1,N))

        # display(Phi)
        return Psi

    def _partially_substitute_Psi(self, Psi):
        theta, Phi_p, f_rot, t, lamb, A_r, f_c, R, V_rad, L_1, L_2 = sympy.symbols("θ Φ_p f_rot t λ A_r f_c, R, V_rad L_1 L_2")
        n, N = sympy.symbols("n N", integer=True, positive=True)

        subs = {
            f_rot: self.drone.f_rot,
            N:     self.drone.N,
            L_1:   self.drone.L_1,
            L_2:   self.drone.L_2,
            lamb:  self.radar.λ,
            f_c:   self.radar.f_c,

        }
        Psi_sub = Psi.subs(subs).doit()
        # display(Psi_sub)

        return Psi_sub

    def _prepare_Psi_functions(self):
        full_Psi = self._prepare_sympy_Psi_function()
        partially_substituted_Psi = self._partially_substitute_Psi(full_Psi)
        return partially_substituted_Psi

    def _lambidfy_Psi(self):
        t, R, V_rad, theta, Phi_p, A_r = sympy.symbols("t R V_rad θ Φ_p A_r")
        args = [t, A_r, Phi_p, R, V_rad, theta]
        return sympy.lambdify(args, self.base_Psi_f, "numpy")

    def set_noise_parameters(self, params):
        if self.noise_model is not None:
            self.noise_model.set_parameters(params)
        else:
            raise AttributeError("Error model was not set")

    def generate_signal(self, context, stft_form=True):
        if not isinstance(context, Context):
            raise ValueError("Provided context is not of Context class")
        # if (self.noise_model is not None) and self.noise_model.params_set_flag == False:
        #     raise AttributeError("The noise_model parameters were not set. Please set them with set_noise_parameters(signal_params)")

        Psi_f = self._lambidfy_Psi()
        # this is a re-arrangement of dB = 10\log_{10}{A_r^2/\sigma^2}
        sigma = context.A_r * np.power(10,-context.snr/20)
        self.set_noise_parameters({"sigma":sigma})
        t_array = np.arange(context.t_start,context.t_stop,context.dt)

        if self.noise_model is None or isinstance(self.noise_model, AdditiveWhiteGaussianNoise):
            R     = context.resolve(context.R,     t_array)
            V_rad = context.resolve(context.V_rad, t_array)
            θ     = context.resolve(context.θ,     t_array)
            Φ_p   = context.resolve(context.Φ_p,   t_array)
            A_r   = context.resolve(context.A_r,   t_array)
            signal = Psi_f(t_array, R, V_rad, θ, Φ_p, A_r)
            if isinstance(self.noise_model, AdditiveWhiteGaussianNoise):
                signal = self.noise_model.apply_noise(signal)

        else:       # Not tested, so this might throw errors
            signal = []
            for t in t_array:
                R     = context.resolve(context.R,     t)
                V_rad = context.resolve(context.V_rad, t)
                θ     = context.resolve(context.θ,     t)
                Φ_p   = context.resolve(context.Φ_p,   t)
                A_r   = context.resolve(context.A_r,   t)
                params = {
                    "t": t
                }
                signal.append(self.noise_model.apply_noise(Psi_f(t_array, R, V_rad, θ, Φ_p, A_r), params))
            signal = np.array(signal)

        if stft_form:
            stft_signal = self.apply_stft(signal, context)
            return t_array, stft_signal
        else:
            return t_array, signal

    @classmethod
    def apply_stft(self, signal, context):
        f, t, Zreal = scipy.signal.stft(
        signal.real, 1/context.dt, window='hamming', nperseg=32, noverlap=16, return_onesided=True)
        Xreal = 20*np.log10(np.abs(Zreal))

        f, t, Zimag = scipy.signal.stft(
        signal.imag, 1/context.dt, window='hamming', nperseg=32, noverlap=16, return_onesided=True)
        Ximag = 20*np.log10(np.abs(Zimag))

        return np.stack((Xreal[1:,:], Ximag[1:,:]))

    def plot_drone_spectrogram(self, stft_signal, context):

        f_pts = stft_signal.shape[1]
        delta_t = 16 * context.dt
        delta_f = (1 / context.dt) / 32

        fig, axs = plt.subplots(2, 1, figsize=(12, 4), sharex=True, sharey=True)
        fig.suptitle(f"Drone: {self.drone.name}")

        fig.supxlabel(f"Time t (dt={delta_t:g} s) [s]")
        fig.supylabel(f"Freq. f ({f_pts} bins, df={delta_f:g} Hz) [Hz]")

        im1 = axs[0].imshow(stft_signal[0], origin='lower', aspect='auto', cmap='viridis')
        axs[0].set_title("Real")
        fig.colorbar(im1, ax=axs[0], label="Magnitude |S(t,f)|")

        im2 = axs[1].imshow(stft_signal[1], origin='lower', aspect='auto', cmap='viridis')
        axs[1].set_title("Imag.")
        fig.colorbar(im2, ax=axs[1], label="Magnitude |S(t,f)|")

        fig.tight_layout()
        plt.show()