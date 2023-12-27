import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class AbramsStrogatz:
    """
    A class that implements the Abrams-Strogatz language competition model.
    
    a (float): exponent of the power-law dependence of the probability of adopting a language. Keep as default value (1.31)
    s (float): prestige/popularity of the language. Default is 0.45.
    t_ini (float): initial time of the simulation. Default is 0.
    t_fin (float): final time of the simulation. Deafult is 50.
    initial_state (list): initial fraction of speakers of each language. Default is 0.6 and 0.4
    """
    def __init__(self, a=1.31, s=0.45, t_ini=0, t_fin=50, initial_state=[0.6, 0.4]):
        """
        Constructor for the AbramsStrogatz class.
        
        alpha (float): exponent of the power-law dependence of the probability of adopting a language. Keep as default value (1.31)
        s (float): prestige/popularity of the language. Default is 0.45.
        t_ini (float): initial time of the simulation. Default is 0.
        t_fin (float): final time of the simulation. Deafult is 50.
        initial_state (list): initial fraction of speakers of each language. Default is 0.6 and 0.4
        """
        self.a = a
        self.s = s
        self.t_ini = t_ini
        self.t_fin = t_fin
        self.time_grid = np.linspace(self.t_ini, self.t_fin, 1000)
        self.initial_state = initial_state
        
    def state_derivative(self, time, state):
        """
        Computes the derivative of the state vector for the language competition model.

        Parameters:
        time (float): current time.
        state (list): current state vector.

        Returns:
        np.array: derivative of the state vector.
        """
        x, y = state
        if x <= 0:
            x = 1e-8  # Set a small positive value for x to avoid invalid power operation
        prob_yx = self.s * x**self.a
        prob_xy = (1 - self.s) * (1 - x)**self.a
        derivative = np.zeros_like(state)
        derivative[0] = y * prob_yx - x * prob_xy
        derivative[1] = x * prob_xy - y * prob_yx
        return derivative
        
    def solve(self):
        """
        Solves the differential equations for the language competition model.

        Returns:
        scipy.integrate.OdeSolution: solution object.
        """
        sol = solve_ivp(self.state_derivative, (self.t_ini, self.t_fin), self.initial_state, t_eval=self.time_grid)
        return sol
    
    def plot(self, title=None, savefig=False):
        """
        Plots the fraction of speakers of each language over time.
        """
        sol = self.solve()
        default_title = 'Abrams-Strogatz Language Competition Model: \nx = '+ str(self.initial_state[0]) + ', y = ' + str(self.initial_state[1]) + ', s = ' + str(self.s)
        plt.style.use('ggplot')
        plt.rcParams['font.size'] = 10
        plt.xlabel('Time')
        plt.ylabel('Fraction of speakers')
        if title:
            plt.title(title)
        else:
            plt.title(default_title)
        plt.gca().set_ylim([-0.05, 1.1])
        plt.plot(sol.t, sol.y[0], label='Language 1')
        plt.plot(sol.t, sol.y[1], label='Language 2')
        plt.legend()
        if savefig:
            plt.savefig(default_title+'.png')
        plt.show()
        
class Castello:
    """
    A class representing the Castello language competition model.
    
    a: float, optional
        Exponent parameter that determines the shape of the transfer function.
    s: float, optional
        Prestige of the language
    t_ini: float, optional
        Initial time for the simulation.
    t_fin: float, optional
        Final time for the simulation.
    initial_state: list of floats, optional
        Initial values for the fraction of speakers of languages 1 and 2 and bilinguals, respectively.

    state_derivative:
        Computes the derivative of the state vector at a given time point.
    solve:
        Solves the differential equations that describe the model and returns the solution.
    plot:
        Plots the solution of the model for the given parameters.
    """

    def __init__(self, a=1.31, s=0.45, t_ini=0, t_fin=50, initial_state=[0.7, 0.1, 0.2]):
        """
        Initializes the Castello model with default parameter values.

        a (float): a parameter determining the non-linearity of the function that describes the advantage of a language.
        s (float): the proportion of speakers who prefer language A over B.
        t_ini (float): the initial time for the simulation.
        t_fin (float): the final time for the simulation.
        state0 (list): the initial state of the system as a list of three floats [x0, y0, b0], representing the initial
                       fractions of speakers of language A, language B, and bilingual speakers, respectively.
        """
        self.a = a
        self.s = s
        self.ti = t_ini
        self.tf = t_fin
        self.time_grid = np.linspace(self.ti, self.tf, 1000)
        self.initial_state = np.array(initial_state).flatten()

    def state_derivative(self, t, state):
        """
        Computes the derivatives of the state variables of the system at a given time.

        t (float): the current time.
        state (list): the current state of the system as a list of three floats [x, y, b], representing the fractions
                      of speakers of language A, language B, and bilingual speakers, respectively.

        Returns:
        der (list): the derivatives of the state variables at time t.
        """
        x, y, b = state
        if x <= 0:
            x = 1e-8  # Set a small positive value for x to avoid invalid power operation
        Pyx = self.s * x**self.a
        Pxy = (1-self.s)*(1-x)**self.a
        Pxb = (1-self.s)*x*(y**self.a)
        Pbx = self.s*(1-x-y)*((1-y)**self.a)
        Pyb = self.s*y*(x**self.a)
        Pby = (1-self.s)*(1-x-y)*((1-x)**self.a)
        der = np.zeros_like(state)
        der[0] = y*Pyx + b*Pbx - x*(Pxy + Pxb)
        der[1] = x*Pxy + b*Pbx - y*(Pyx + Pyb)
        der[2] = x*Pxb + y*Pyb - b*(Pbx + Pby)
        return der

    def solve(self):
        """
        Solves the system of ordinary differential equations using the solve_ivp method.

        Returns:
        sol (object): a solution object returned by solve_ivp, containing the time grid and the values of the
                      state variables at each time point.
        """
        sol = solve_ivp(self.state_derivative, (self.ti, self.tf), self.initial_state, t_eval=self.time_grid)
        return sol

    def plot(self, title=None, savefig=False):
        """
        Plots the solution of the system of ordinary differential equations using Matplotlib.

        Returns:
        None
        """
        sol = self.solve()
        default_title = 'Castello Language Competition Model: \nx = '+ str(self.initial_state[0]) + ', y = ' + str(self.initial_state[1]) + ', b = ' + str(self.initial_state[2]) + ', s = ' + str(self.s)
        plt.style.use('ggplot')
        plt.rcParams['font.size'] = 10
        plt.xlabel('Time')
        plt.ylabel('Fraction of speakers')
        if title:
            plt.title(title)
        else:
            plt.title(default_title)
        plt.gca().set_ylim([-0.05, 1.1])
        plt.plot(sol.t, sol.y[0], label='Language 1')
        plt.plot(sol.t, sol.y[1], label='Language 2')
        plt.plot(sol.t, sol.y[2], label='Bilinguals')
        plt.legend()
        if savefig:
            plt.savefig(default_title+'.png')
        plt.show()
        
class Mira:
    """
    This class implements the Mira model for language competition.

    a: float, optional
        Exponent parameter that determines the shape of the transfer function.
    sx: float, optional
        Fraction of the population that interacts with monolinguals of language 1.
    sy: float, optional
        Fraction of the population that interacts with monolinguals of language 2.
    k: float, optional
        Parameter that modulates the similarity of the languages.
    t_ini: float, optional
        Initial time for the simulation.
    t_fin: float, optional
        Final time for the simulation.
    initial_state: list of floats, optional
        Initial values for the fraction of speakers of languages 1 and 2 and bilinguals, respectively.

    state_derivative:
        Computes the derivative of the state vector at a given time point.
    solve:
        Solves the differential equations that describe the model and returns the solution.
    plot:
        Plots the solution of the model for the given parameters.
    """

    def __init__(self, a=1.1961, sx=0.6311, sy=0.3689, k=0.7714, t_ini=0, t_fin=50, initial_state=[0.7, 0.1, 0.2]):
        self.a = a
        self.sx = sx
        self.sy = sy
        self.k = k
        self.t_ini = t_ini
        self.t_fin = t_fin
        self.time_grid = np.linspace(self.t_ini, self.t_fin, 1000)
        self.initial_state = initial_state
    
    def state_derivative(self, time, state):
        """
        Computes the derivative of the state vector at a given time point.
        
        time: float
            Time point at which the derivative is evaluated.
        state: numpy array
            Vector containing the current values of the variables that describe the system.
        
        Returns:
        --------
        numpy array
            Vector containing the derivatives of the variables that describe the system.
        """
        x, y, b = state
        if x <= 0 or y <=0:
            x = 1e-8  # Set a small positive value for x to avoid invalid power operation
            y = 1e-8  # Set a small positive value for y to avoid invalid power operation
        if 1-x <= 0 or 1-y <= 0:
            Pxb = 0
            Pxy = 0
            Pby = 0
        else:
            Pxb = self.k*(1-self.sy)*((1-x)**self.a)
            Pxy = (1-self.k)*(1-self.sy)*((1-x)**self.a)
            Pby = (1-self.k)*(1-self.sy)*((1-x)**self.a)
        Pyb = self.k*self.sx*((1-y)**self.a)
        Pyx = (1-self.k)*self.sx*((1-y)**self.a)
        Pbx = (1-self.k)*self.sx*((1-y)**self.a)
        der = np.zeros_like(state)
        der[0] = y*Pyx + b*Pbx - x*(Pxy + Pxb)
        der[1] = x*Pxy + b*Pbx - y*(Pyx + Pyb)
        der[2] = x*Pxb + y*Pyb - b*(Pbx + Pby)
        return der
    
    def solve(self):
        """
        Solves the differential equation for the Mira Language Competition model using the initial conditions 
        and parameters set in the object's initialization.
       
        Returns:
        -------
        sol : scipy.integrate.OdeSolution object
        A solution object containing the results of the differential equation solver.
        """
        sol = solve_ivp(self.state_derivative, (self.t_ini, self.t_fin), self.initial_state, t_eval=self.time_grid)
        return sol
    
    def plot(self, title=None, savefig=True):
        """
        Plots the results of the Mira Language Competition model for the given parameters and initial conditions.
        
        Parameters:
        -----------
        title : str, optional
            The title of the plot. If not given, a default title will be generated based on the object's parameters.
        savefig : bool, optional
            Whether to save the plot as a PNG file with the default title. Default is True.

        Returns:
        -------
        None
        """
        sol = self.solve()
        default_title = 'Mira Language Competition Model: \nx = '+ str(self.initial_state[0]) + ', y = ' + str(self.initial_state[1]) + ', b = ' + str(self.initial_state[2]) + ', \nsx = ' + str(self.sx) + ', sy = ' + str(self.sy) +  ', k = ' + str(self.k)
        plt.style.use('ggplot')
        plt.rcParams['font.size'] = 7
        plt.xlabel('Time')
        plt.ylabel('Fraction of speakers')
        if title:
            plt.title(title)
        else:
            plt.title(default_title)
        plt.gca().set_ylim([-0.05, 1.1])
        plt.plot(sol.t, sol.y[0], label='Language 1')
        plt.plot(sol.t, sol.y[1], label='Language 2')
        plt.plot(sol.t, sol.y[2], label='Bilinguals')
        plt.legend()
        if savefig:
            plt.savefig(default_title+'.png')
        plt.show()
        

