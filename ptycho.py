import numpy as np
from scipy.fft import fft, ifft, fftshift, fftfreq
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy import optimize
from scipy.integrate import simps
from scipy.optimize import minimize_scalar
import matplotlib.font_manager as font_manager
import matplotlib.ticker as mticker


class ptycho():
    def __init__(self) -> None:        
        self.font = font_manager.FontProperties(family='Comic Sans MS',
                                   weight='bold',
                                   style='normal', size=12)

    
    def Gaussian(self, x: np.ndarray, FWHM: float, x0: float= 0.0 , order: int = 1):
        '''
            This function provide a way to build a gaussian distrubition, and the order is for
            super-gaussian distibution. [https://en.wikipedia.org/wiki/Gaussian_function]

            # Paramaters:
            x: 1D array, time usaly,
            x0: float, the mean value (centred value),
            FWHM: float, Full Width Half MAx,
            order: inetger, the default is '1'.

            # Note the "x" array should be in the same unit as FWHM.
        '''

        return np.exp( - np.log(2)*(4*( (x-x0)/FWHM  )**2)**order  ) 

        
    def fwhm_eval(self, x, y, rel_h = .5):
        '''
            This function evaluate the Full width Half MAx of the signal (Electric Field)
            #Paramaters:
            x : 1D array, time usaly,
            y : 1D array, Data point,
            rel_h: float, optional the relative height that we want to evaluate the fwhm as a seconf heighest prominence
            of the signal ...>> docs [scipy.signal.peak_widths] 
        '''
        # I find this function more useful to evaluate the fwhm of very structured signal
        # before we had a naive function to deal with such problem
        
        peaks, _ = find_peaks(y) # find the peaks
        res = peak_widths(y, peaks, rel_height=rel_h) # Check documentaion basically a tuple with : width, width_hight& position 
        ind = np.where( peaks == np.argmax(y) )[0][0] # search for the index od the width corresponding to the highest peak
        # we interpolate then we just do the difference ...
        return np.interp( res[3][ind], (0, len(x)), (min(x), max(x)) ) - np.interp( res[2][ind], (0, len(x)), (min(x), max(x)) )

    def rms(self, x, y):
        '''
            This function evaluate the root mean square of the pulse, (it is convenient to use this quantity to analyze
            short pulses)
            # Paramaters:
            x: 1D array, time axis (s),
            x: 1D array, pulse temporal data.
            # return:
            rms in second.
        '''

        y = np.roll(y,int(len(y)/2)-np.argmax(abs(y)))
        num = simps(x**2*y,x)
        den = simps(y,x)

        return np.sqrt(num/den)
        
    
    def FFT(self, x):
        '''
            This function is a short cut for tedious line of code !!
        '''
        return fftshift( fft( fftshift(x) ) )

    def IFFT(self, x):
        '''
            This function is a short cut for tedious line of code !!
        '''
        return fftshift( ifft( fftshift(x) ) )

    def G_trebino(self, I, psi_squared):
        
        ''' 
            This Function evaluate the error between the input and the recovred 2D maps. [DeLong, Optics letters(1994).]
            # Paramaters:
            I: 2D array, the recorded/simulated ptychogram,
            psi_squared: 2D array, the retrieved ptychogrma.

            => Credit to J.L for writing this function.
        '''
        N_taus, NFFT = np.shape(I)
        def error (gamma):
            return  np.sqrt(np.sum(np.abs(I - gamma * psi_squared)**2) / NFFT / N_taus)
        res= minimize_scalar (error)
        Gamma_Trebino = res.x 
        return error(Gamma_Trebino)  

    def remove_lin_phase(self, pulse: np.ndarray, w_axis: np.ndarray):
        '''
            This function just remove linear spectral phase components due to w_0.
            # Paramaters:
            pulse: 1D array, temporal complex E.F pulse,
            w_axis: 1D array, angular frequency (Hz).
        '''
        E_w = self.FFT(pulse)
        
        phase = np.unwrap(np.angle(E_w))

        lin_ = np.polyfit(phase, w_axis, 1)
    
        E_w *= np.exp(-1j * np.polyval(lin_, w_axis))

        return self.IFFT(E_w) 
        
        
    def pty_signal (self, om, E, G, delays):   # SFG complex amplitude
        '''
            E, G should in time domain
            om an delays in PHz and fs (resp)
            return the complex spectrogram "Not real measurement you need the square the magnitude for that purpose"
        '''
        NFFT = len(om)
        N_taus = len(delays)

        out = np.empty((N_taus,NFFT),dtype=np.complex128)
        for j in range(N_taus):
            out[j, :] = self.FFT(self.IFFT( self.FFT( G ) * np.exp( 1j * delays[j] * om) ) * E )  # in freq domain

        return out  

    def single_ptycho(self,gate: np.ndarray, test:np.ndarray, w_axis: np.ndarray , t_axis: np.ndarray ,  delay: np.ndarray, \
                  Imn: np.ndarray, update_g= True, gate_wonder= False,\
                    pulse_wonder= False, beta: float=.15, gamma: float=.1):
    
        '''
        # Paramaters:

            gate: 1D temporal data of the gate;
            Ob_w: 1D temporal data of the object;
            w_axis: 1D angular frequency (PHz);
            t_axis: 1D time axis (fs);
            delay: 1D delay axis (fs);
            Imn: 2D array == pthychogram;        
            update_gate: boolean;
            gate\pulse wonder: boolean, center the pulses around zeros;
            beta and gamma: float (optional)

        '''
     
        Pr_new = gate
        Ob_new = test
        
        
        for m in np.random.permutation(np.arange(Imn.shape[0])) : 

            ## initialisation

            Pr_old = Pr_new
            Ob_old = Ob_new
            
            # Updating the object:

            Pr_shifted = self.IFFT( self.FFT( Pr_old ) * np.exp( 1.0j * w_axis * delay[m]) ) 

            Xi_old = Ob_old * Pr_shifted 

            Xi_new = self.IFFT( np.sqrt( Imn[m, :] ) * np.exp( 1.0j * np.angle( self.FFT( Xi_old ) ) ) ) 


            Ob_new = Ob_old + beta * ( np.conjugate( Pr_shifted ) / np.max( abs(Pr_shifted)**2 ) ) * ( Xi_new - Xi_old ) 

            #Ob_new = self.remove_lin_phase(Ob_new, w_axis=w_axis*1e15)
            
            if pulse_wonder:
                # Undo the phase ramp:

                max_ind = np.argmax( abs( Ob_new ) )
                if abs(max_ind-Imn.shape[1]//2) > 20:
                    Ob_new =  np.roll( abs(Ob_new), -max_ind + Imn.shape[1]//2 )*np.exp(1j*np.angle(Ob_new))

            if update_g == True:
                # Updating the object:

                Ob_shifted = self.IFFT( self.FFT( Ob_old ) * np.exp( - 1.0j * w_axis * delay[m]) ) 

                Xi_old = Ob_shifted * Pr_old 

                Xi_new = self.IFFT( np.sqrt( Imn[m, :] ) * np.exp( 1.0j * np.angle( self.FFT( Xi_old ) ) ) ) 

                Pr_new = Pr_old + gamma * ( np.conjugate( Ob_shifted ) / np.max( abs(Ob_shifted)**2 ) ) * ( Xi_new - Xi_old )

                if gate_wonder:
                    # Undo the phase ramp:

                    max_ind = np.argmax( abs( Pr_new ) )
                    if abs(max_ind-Imn.shape[1]//2) > 20:
                        Pr_new =  np.roll( abs(Pr_new), -max_ind + Imn.shape[1]//2 ) 

                Pr_new = abs(Pr_new) # the probe is free phase pulse
            

        return Ob_new, Pr_new
    
    def plot_panel(self, E_t: np.ndarray, G_t: np.ndarray, Time: np.ndarray, om: np.ndarray, delay: np.ndarray,\
                Inew: np.ndarray, Iold: np.ndarray, \
                    om_lim = None, E_t_lim =None, G_t_lim  = None, figsize: tuple =  (12, 4)):    


        '''
        # Paramaters:

            E_t: 1D temporal complex data of the retrieved pulse (Electric field);
            G_t: 1D temporal complex data of the gate pulse (Electric field);
            Tim: 1D Time axis (fs);
            om: 1D angular frequency (PHz);
            delay: 1D delay axis (fs);
            Inew: 2D array == recovred pthychogram;
            Iold: 2D array == recorded pthychogram;
            om_lim: tuple, angular frequency axis limits for the plot (PHz) -optinal-;
            E\G_t_lim: tuple, time axis limits for the plot (fs) -optinal-; 
                => E :i.e test pulse\ G: i.e => Gate pulse;

        '''

        marginal = np.sum( np.sqrt( Inew ), axis=0 )
        marginal /= np.max(marginal)

        # set the limits for the temporal plots 
        if G_t_lim == None:
            ind_g = np.where(Iold.T == np.max(Iold.T))[0][0]
            init_G = np.interp(Time, delay-delay[np.argmax(Iold.T[ind_g, :])], Iold.T[ind_g, :])
            fwhm_G = self.fwhm_eval(Time, abs(init_G)**2)
            G_t_lim = (-3*fwhm_G, 3*fwhm_G)
        
        fwhm_G = self.fwhm_eval(Time, abs(G_t)**2)
        rms_G = self.rms(Time, abs(G_t)**2)            

        fwhm_E = self.fwhm_eval(Time, abs(E_t)**2)    
        rms_E = self.rms(Time, abs(E_t)**2)
        
        if E_t_lim == None:        
            E_t_lim = (-4*fwhm_E, 4*fwhm_E)    

        # set the limits for the 2D maps
        if om_lim == None:
            #mask = marginal > 1e-3
            #om_lim = ( om[mask[0]], om[mask[-1]] )
            om_lim = ( om[0], om[-1] )

        plt.rcParams["figure.figsize"] = figsize
        
        fig, axs = plt.subplots(ncols=4, nrows=1)

        # The reconstruction
        
        im = axs[0].pcolormesh(om, delay, Inew, cmap="jet") 
        axs[0].set_title("Reconstruced", font="serif", size=18)
        axs[0].set_xlabel("$\omega$ (PHz)", font="serif", size=18)
        axs[0].set_ylabel("Delay (fs)", font="serif", size=18)       
        axs[0].set_xlim([om_lim[0], om_lim[1]])
        
        fig.colorbar(im)

        # The difference
        
        im = axs[1].pcolormesh(om, delay, abs(Iold-Inew), cmap="jet")        
        axs[1].set_title("Difference", font="serif", size=18)
        axs[1].set_xlabel("$\omega$ (PHz)", font="serif", size=18)
        axs[1].set_ylabel("Delay (fs)", font="serif", size=18)
        axs[1].set_xlim([om_lim[0], om_lim[1]])
        fig.colorbar(im)

        # the test pulse
        E_int = abs(E_t)**2/max(abs(E_t)**2)
        E_int = np.roll(E_int, (np.argmax(E_int)-len(E_int)//2))
        axs[2].plot(Time,  E_int, label="FWHM = {:.2f} fs\n rms = {:.2f} fs".format(fwhm_E, rms_E))
        axs[2].set_xlabel("Time (fs)", font="serif", size=18)
        axs[2].set_ylabel("Intensity (a.u)", font="serif", size=18)
        axs[2].set_title("Test pulse (Ret)", font="serif", size=18)   
        axs[2].legend(loc=3, prop=self.font, bbox_to_anchor=(-.05, -.7))
        axs[2].set_xlim([E_t_lim[0], E_t_lim[1]])

        # Gate pulse
        
        axs[3].plot(Time, abs(G_t)**2/max(abs(G_t)**2), label="FWHM = {:.2f} fs\n rms = {:.2f} fs".format(fwhm_G, rms_G))
        axs[3].set_xlabel("Time (fs)", font="serif", size=18)
        axs[3].set_ylabel("Intensity (a.u)", font="serif", size=18)
        axs[3].set_title("Gate pulse (Ret)", font="serif", size=18)
        axs[3].legend(loc=3, prop=self.font, bbox_to_anchor=(-.05, -.7))
        axs[3].set_xlim([G_t_lim[0], G_t_lim[1]])

        fig.tight_layout(h_pad=2)
        plt.show()

    def read_trace(self, path: str, title: str = None, wl_lim: tuple = None, om_lim: tuple = None,\
                   draw: bool= False, figsize: tuple= (8, 6), N_int: int = None, COM: tuple = None):
        '''
            This function read the recorder/simulated ptychogram
            # Paramaters:
            path: string, path of the data location;
            the others are optional
            # return
            the trace processed and interpolated, the grid of time and angilar frequency, the delay and the marginal
            whether temporal or frequency domain.
        '''

        data = np.loadtxt(path, delimiter="\t")

        delay = data[1:-1,0]

        # wavelength
        wl = data[0,1:-1]
        N = len(wl)
        om = 2e9*3e8*np.pi*np.linspace(1/wl[-1], 1/wl[0], len(wl)) 
        
        
        # process trace:
        trace = data[1:-1, 1:-1] 
        trace *= wl*wl
        trace[trace < 0] = 0
        trace /= np.max(trace)
            
        # marginal 
        guess_w = np.sum( np.sqrt(trace), axis=0 )

        guess_w /= max( guess_w )
        
        guess_t = abs(self.IFFT(guess_w))
        guess_t /= max( guess_t )
        
        # axes

        dwl = np.mean( np.diff(wl) )
        wl_0 = wl[ np.argmax( guess_w ) ]
        dw = 2e9*np.pi*3e8*dwl/(wl_0**2)
        #dt = np.pi / (0.5 * len(wl) * dw)

        #time = -np.floor(0.5 * len(wl)) * dt + np.arange(len(wl)) * dt
        time = 2*np.pi*( -np.floor(N//2) + np.arange(N) ) /(dw*N) # s

        if COM != None:
            GDD = self.COM(trace, om, delay, lim=COM, draw=True )[0][1]
            phase = np.zeros_like(om)
            a1, a2 = np.where(om*1e-15 > COM[0])[0][0] , np.where(om*1e-15 < COM[-1])[0][-1]
            phase[a1:a2] = (-.5*GDD*om[a1:a2]*om[a1:a2])
            
            guess_t = self.IFFT(guess_w*np.exp(1j*phase))
            
        if N_int != None:
            ptychogram = np.empty(shape=(len(delay), N_int))

            om_int = np.linspace(om[0], om[-1], N_int)
            time_int = np.linspace(time[0], time[-1], N_int)
            wl = np.linspace(wl[0], wl[-1], N_int)
            if COM != None:
                phase = np.interp( om_int, om, np.angle(guess_t), left=0, right=0)
                abs_guess_t = np.interp( time_int, time, abs(guess_t), left=0, right=0)
                guess_t = abs_guess_t*np.exp(1j*phase)
            else:
                guess_t = np.interp( time_int, time, guess_t, left=0, right=0)
            for i, row in enumerate(trace):

                ptychogram[i, :] = np.interp( om_int, om, row, left=0, right=0) 

            ptychogram /= np.max(ptychogram)
            trace = ptychogram            

            guess_w = np.interp( om_int, om, guess_w, left=0, right=0)
            
            om = om_int
            time = time_int
        

        if draw:
            
            
            fig = plt.figure(figsize=figsize)

            # plot in wavelength trace + marginal

            ax = plt.subplot(121)
            ax.pcolormesh(wl, delay, trace, cmap="jet")
            axm = ax.twinx()
            axm.plot(wl, guess_w, color="white")
            axm.set_ylabel("Aggregated intensity over delay axis" )
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Delay (fs)")

            if wl_lim != None:
                ax.set_xlim([wl_lim[0], wl_lim[1]])

            ax = plt.subplot(122)

            # plot in angular frequency trace + marginal

            C = ax.pcolormesh(om/1e15, delay, trace[:, ::-1], cmap="jet")
            ax.set_xlabel("$\omega$ (PHz)")
            ax.set_ylabel("Delay (fs)")

            if om_lim == None:                                
                mask = guess_w > .1
                om_lim = (om[mask][0]/1e15, om[mask][-1]/1e15)
                
            ax.set_xlim([*om_lim])    
            fig.colorbar(C, pad=0.2)

            axm = ax.twinx()
            axm.plot(om/1e15, guess_w[::-1], color="white", label="SFG spectra")   

            if title == None:
               fig.suptitle("Ptychogram", font="serif" ,fontsize=14)
            else:
               fig.suptitle(title) 

            
            fig.tight_layout(h_pad=1, w_pad=2)
            plt.show()       

        return   trace[:, ::-1], om, time, delay, guess_w, guess_t

    def plot_phase(self, w, pulse, Wl_g : float = None, lim: tuple = None, order: int = 4,\
                   figsize: tuple = (8, 6), title: str = None):
        
        '''
            This function serve as tool to visualize both the pulse and the phase in frequency domain.
            # Paramaters:
            w : 1D array, angular frequency (Hz),
            pulse : 1D complex array, temporal Electric field,
            Wl_g : float, the central wavelength (nm) of the gate (optional),
            lim: tuple, bonds for the phase fitting purpose, e.g: lim = (4.5, 5.5) # in PHz.
            order: integer, Order of the phase fit (default = 4).
        '''
        
        spec_pulse = self.FFT(pulse)
        
        spec_intensity = abs(spec_pulse)
        spec_intensity /=  max(spec_intensity)
        
        phase = np.unwrap(np.angle(spec_pulse))

        if Wl_g != None:
            w_g = 2e9*np.pi*3e8/Wl_g
            w = w - w_g
        
        fig = plt.figure(figsize=figsize)

        ax = plt.subplot(111)
        
        ax.plot(w/1e15, spec_intensity, label="Pulse")
        ax.set_xlabel("$\omega$ (PHz)", font="serif", fontsize=18)
        ax.set_ylabel("Intensity (a.u)", font="serif", fontsize=18)
        ax.legend(loc = 3, prop=self.font, bbox_to_anchor=(1.2, .9))
        
        axp = ax.twinx()
        axp.plot(w/1e15, phase, "orange", label="$\phi(\omega)$: All components.")
        axp.set_ylabel("Phase (rad)", font="serif", fontsize=18)
        if lim != None:
            a, b  = np.where(w/1e15 > lim[0])[0][0], np.where(w/1e15 < lim[-1])[0][-1]
            coeffs  = np.polyfit(w[a:b], phase[a:b], 4)
            new_phase = np.polyval(coeffs, w[a:b]) - np.polyval(coeffs[::-1][0:2], w[a:b])
            axp.plot(w[a:b]/1e15, new_phase, "--r", linewidth=3, label="$\phi(\omega)$: Free from \n $GD(\omega)$ & $\phi_0$.")
            return phase, coeffs
        axp.legend(loc = 3, prop=self.font, bbox_to_anchor=(1.2, .7))
                    
        if title == None:
            fig.suptitle("Spectral Profile", font="serif" ,fontsize=14)
        else:
            fig.suptitle(title)
               
        fig.tight_layout(h_pad=1, w_pad=2)
        plt.show()
        return phase

    def COM(self, trace: np.ndarray, w: np.ndarray, delay: np.ndarray, order: tuple = (1, ), lim: tuple = None,\
            figsize: tuple = (6, 4), draw: bool = True, title: str = None):
        '''
            This function Fetch for Group delay Disspersion from the 2D map using what we call The Center of The Mass.
            (Credit to J.L)

            # Paramaters:
            trace: 2D array, the ptychogram,
            w: 1D array, the angular frequency axis (Hz),
            delay: 1D array, the dealy axis in fs.
            order: tuple of integer, default is 1 (order of fitting), you can pas (1,2, 3 , ... etc),
            lim: tuple, the limits for which the fitting is performed (eg: lim = (4.5, 5.5) # in PHz).
            
            # return:
            list of Coefficients in descending order (generally the second elements represents theGroup delay Disspersion in s**2).

            # recomendations:
            Charge the trace first using the function "read_trace()".

            # Note: I find the following paper "Lazić Ultramicroscopy (2016)" interestting. The COM method should have a mathematical
            proof as a support for its validity.
        '''

        marg = np.sum( np.sqrt(trace), axis=0)
        marg /= np.max(marg)
        
        if lim != None:
            a1, a2 = np.where(w*1e-15 > lim[0])[0][0] , np.where(w*1e-15 < lim[-1])[0][-1]
        else:
            mask = marg > 1e-3
            a1, a2 = np.where(w*1e-15 > w[mask[0]] )[0][0] , np.where(w*1e-15 <  w[mask[-1]] )[0][-1]
               
    
        com = np.empty(shape=(trace.shape[1]))
        

        for i, tau_slice in enumerate(trace.T):

            int_tau = np.sum(tau_slice)
            com[i] = np.sum(delay*1e-15 * tau_slice)/int_tau

        coeff = []
        for i in order:
            coeff.append(np.polyfit(w[a1:a2], com[a1:a2], i)[::-1])
            print(f"The coefficient of the plynomial fit of order {i}: {coeff[-1]}" )

        if draw :    
            plt.rcParams["figure.figsize"] = figsize

            ax = plt.subplot()

            ax.plot(w[a1:a2]/1e15, marg[a1:a2], label="Marginal")
            ax.set_ylabel("Amplitude (a.u)", font="serif", size=18)
            ax.set_xlabel("$\omega$ (PHz)", font="serif", size=18)
            ax.legend(loc=1, prop=self.font, bbox_to_anchor=(1.5, .6))

            if title == None:
                title = "Center of Mass method"
            ax.set_title(title, font="serif", size=18)
                
            axp = ax.twinx()
            axp.plot(w[a1:a2]/1e15, com[a1:a2], "orange", label="COM")
            axp.set_ylabel("COM", font="serif", size=18)

            color_list = ["b", "g", "r", "c", "m", "y", "k"]
            for (c, cl) in zip(coeff, color_list):

                com_fit = np.polyval(c[::-1], w[a1:a2])            
                axp.plot(w[a1:a2]/1e15, com_fit, ls="--", color=cl, linewidth=2, label="Fit order: {}".format(len(c)-1))
                
            axp.legend(loc=1, prop=self.font, bbox_to_anchor=(1.5, 1)) 
            plt.show()


        return coeff
            




