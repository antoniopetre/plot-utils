import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from utils.makeRatio import makeRatio
from utils.handleUncertainties import handleUncertainties
from config.configuration import configuration

class EfficiencyPlots:
    def __init__ (self, 
                  lc_dict_arr, cp_dict_arr,
                  annotate=None,
                  legend=None,
                  c=['red', 'black'],
                  save=None):

        self.lc_dict_arr = lc_dict_arr
        self.cp_dict_arr = cp_dict_arr

        self.annotate=annotate
        self.legend=legend
        self.c = c
        self.save = save

        self.eta_bins = np.linspace(-1.4, 1.4, 10)
        self.phi_bins = np.linspace(-np.pi, np.pi, 12)
        self.energy_bins = np.linspace(0, 600, 15)

        self.makePlot(cumulative=False)
        self.makePlot(cumulative=True)
        for layer in range(0, 7):
            self.makePlot(cumulative=False, layer=layer)

    def makePlot(self, cumulative, layer=None):
        cumulative = int(cumulative)
        fig, axs = plt.subplots(1, 3, figsize=(30, 15))
        for cp_dict, k in zip(self.cp_dict_arr, range(0, len(self.cp_dict_arr))):
            if (layer == None):
                eta_eff, phi_eff, energy_eff = count(cp_dict, self.lc_dict_arr[k])
            else:
                eta_eff, phi_eff, energy_eff = countPerLayer(cp_dict, self.lc_dict_arr[k], layer)
              
            num_eta_counts, _ = np.histogram(eta_eff[0][cumulative], bins=self.eta_bins)            
            denom_eta_counts, _ = np.histogram(eta_eff[1], bins=self.eta_bins)            
           
            eta_ratio, eta_ratio_error = makeRatio(num_eta_counts, denom_eta_counts)
            eta_ratio_error = handleUncertainties(eta_ratio, eta_ratio_error)
            axs[0].errorbar(self.eta_bins[:-1], \
                            eta_ratio, yerr=eta_ratio_error, \
                            fmt=".", markersize=20, color=self.c[k], \
                            label=self.legend[k])
            axs[0].set_xlabel(r"$\eta$")
            axs[0].set_xlim(-1.2, 1.2) 
       
            num_phi_counts, _ = np.histogram(phi_eff[0][cumulative], bins=self.phi_bins)            
            denom_phi_counts, _ = np.histogram(phi_eff[1], bins=self.phi_bins)            
                                                                                            
            phi_ratio, phi_ratio_error = makeRatio(num_phi_counts, denom_phi_counts)
            phi_ratio_error = handleUncertainties(phi_ratio, phi_ratio_error)
            axs[1].errorbar(self.phi_bins[:-1], \
                            phi_ratio, yerr=phi_ratio_error, \
                            fmt=".", markersize=20, color=self.c[k], \
                            label=self.legend[k])
            axs[1].set_xlabel(r"$\phi$")
            axs[1].set_xlim(-np.pi, np.pi) 

            num_energy_counts, _ = np.histogram(energy_eff[0][cumulative], bins=self.energy_bins)            
            denom_energy_counts, _ = np.histogram(energy_eff[1], bins=self.energy_bins)            
                                                                                            
            energy_ratio, energy_ratio_error = makeRatio(num_energy_counts, denom_energy_counts)
            energy_ratio_error = handleUncertainties(energy_ratio, energy_ratio_error)
            axs[2].errorbar(self.energy_bins[:-1], \
                            energy_ratio, yerr=energy_ratio_error, \
                            fmt=".", markersize=20, color=self.c[k], \
                            label=self.legend[k])
            axs[2].set_xlabel(r"Energy (sim) [GeV]")

        for ax in axs:
            hep.cms.text(text=configuration["cmsText"], ax=ax)
            ax.set_ylim(-1e-2, 1.1)
            ax.set_ylabel('Efficiency')
            ax.grid(True)
            ax.legend()
            if self.annotate != None:
                if "Energy (sim) [GeV]" not in ax.get_xlabel():
                    ax.annotate(text=self.annotate, xy=(ax.get_xlim()[0]*0.9, 0.4), fontsize=20)
                else:
                    ax.annotate(text=self.annotate, xy=(ax.get_xlim()[1]*0.2, 0.4), fontsize=20)

        if (layer == None):
            if (cumulative) :
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison_cumulative.pdf")
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency_cumulative.pdf")
            else:
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison.pdf")
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency.pdf")
    
        else:
            if (cumulative) :
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison_cumulative_layer%s.pdf" % layer)
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency_cumulative_layer%s.pdf" % layer)
            else:
                if (self.save != None and len(self.c) == 2):
                    fig.savefig(self.save+"efficiency_comparison_layer%s.pdf" % layer)
                elif (self.save != None):
                    fig.savefig(self.save+"efficiency_layer%s.pdf" % layer)

        fig.clf()


def count(cp_dict, lc_dict):
    num_eta, denom_eta = [], []
    num_phi, denom_phi = [], []
    num_energy, denom_energy = [], []

    cumulative_eta, cumulative_phi, cumulative_energy = [], [], []
    for event in range(0, len(cp_dict)):
        event = str(event+1)
        for cp in range(0, len(cp_dict[event]['CaloParticleEnergy'])):
            total_shared_energy = 0
            if (len(cp_dict[event]['CP2LCscore'][cp]) == 0) : continue
            denom_eta.append(cp_dict[event]['CaloParticleEta'][cp])
            denom_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
            #denom_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
            denom_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
            if len(cp_dict[event]['SharedEnergy'][cp]) == 0 : continue
            for energy in cp_dict[event]['SharedEnergy'][cp]:
                total_shared_energy += energy / cp_dict[event]['CaloParticleEnergy'][cp]
            max_shared_energy = max(cp_dict[event]['SharedEnergy'][cp])
            max_shared_energy /= cp_dict[event]['CaloParticleEnergy'][cp]
            if max_shared_energy > 0.7:
                num_eta.append(cp_dict[event]['CaloParticleEta'][cp])         
                num_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
                num_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
            if total_shared_energy > 0.7:
                cumulative_eta.append(cp_dict[event]['CaloParticleEta'][cp])
                cumulative_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
                cumulative_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])

    return [[num_eta,cumulative_eta], denom_eta], [[num_phi, cumulative_phi], denom_phi], [[num_energy, cumulative_energy], denom_energy]

def countPerLayer(cp_dict, lc_dict, layer):
    num_eta, denom_eta = [], []                                                                                                            
    num_phi, denom_phi = [], []
    num_energy, denom_energy = [], []
                                                                                                                                       
    cumulative_eta, cumulative_phi, cumulative_energy = [], [], []
    for event in range(0, len(cp_dict)):
         event = str(event+1)
         for cp in range(0, len(cp_dict[event]['CaloParticleEnergy'])):
             total_shared_energy = 0
             if (len(cp_dict[event]['CP2LCscore'][cp]) == 0) : continue
             denom_eta.append(cp_dict[event]['CaloParticleEta'][cp])
             denom_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
             denom_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
             if len(cp_dict[event]['SharedEnergy'][cp]) == 0 : continue
             for energy in cp_dict[event]['SharedEnergy'][cp]:
                 total_shared_energy += energy / cp_dict[event]['CaloParticleEnergy'][cp]
             max_shared_energy = max(cp_dict[event]['SharedEnergy'][cp])
             arg_max_shared_energy = np.argmax(cp_dict[event]['SharedEnergy'][cp])
             if (lc_dict[event]['LayerClustersLayer'][arg_max_shared_energy] != layer): continue
             max_shared_energy /= cp_dict[event]['CaloParticleEnergy'][cp]
             if max_shared_energy > 0.7:
                 num_eta.append(cp_dict[event]['CaloParticleEta'][cp])         
                 num_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
                 num_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
             if total_shared_energy > 0.7:
                 cumulative_eta.append(cp_dict[event]['CaloParticleEta'][cp])
                 cumulative_phi.append(cp_dict[event]['CaloParticlePhi'][cp])
                 cumulative_energy.append(cp_dict[event]['CaloParticleEnergy'][cp])
                                                                                                                                           
    return [[num_eta,cumulative_eta], denom_eta], [[num_phi, cumulative_phi], denom_phi], [[num_energy, cumulative_energy], denom_energy]











 
