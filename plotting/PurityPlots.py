import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak

from utils.makeRatio import makeRatio
from utils.handleUncertainties import handleUncertainties
from config.configuration import configuration

class PurityPlots:
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
        self.energy_bins = np.linspace(0, 600, 10)

        self.makePlot(No_min_associatedLcs=0) # make purity plots
        #self.makePlot(No_min_associatedLcs=1) # make duplicate rate plots

    def makePlot(self, No_min_associatedLcs=0):
        fig, axs = plt.subplots(1, 3, figsize=(30, 15))
        for cp_dict, k in zip(self.cp_dict_arr, range(0, len(self.cp_dict_arr))):
            eta_pur, phi_pur, energy_pur = count(cp_dict, No_min_associatedLcs)
        
            #print(f'No_min_associatedLcs={No_min_associatedLcs}')
            check_no_outside1 = ak.flatten(eta_pur[0]) > 1.4
            check_no_outside2 = ak.flatten(eta_pur[0]) < -1.4
            #print(f'no_outside1={ak.count_nonzero(check_no_outside1)}')
            #print(f'no_outside2={ak.count_nonzero(check_no_outside2)}')

            num_eta_counts, _ = np.histogram(ak.flatten(eta_pur[0]), bins=self.eta_bins)            
            denom_eta_counts, _ = np.histogram(ak.flatten(eta_pur[1]), bins=self.eta_bins)            
           
            #print(num_eta_counts)
            #print(denom_eta_counts)
            
            eta_ratio, eta_ratio_error = makeRatio(num_eta_counts, denom_eta_counts)
            eta_ratio_error = handleUncertainties(eta_ratio, eta_ratio_error)
            axs[0].errorbar(self.eta_bins[:-1], \
                            eta_ratio, yerr=eta_ratio_error, \
                            fmt=".", markersize=20, color=self.c[k], \
                            label=self.legend[k])
            axs[0].set_xlabel(r"$\eta$")
            axs[0].set_xlim(-1.2, 1.2) 
       
            num_phi_counts, _ = np.histogram(ak.flatten(phi_pur[0]), bins=self.phi_bins)            
            denom_phi_counts, _ = np.histogram(ak.flatten(phi_pur[1]), bins=self.phi_bins)            
                                                                                            
            phi_ratio, phi_ratio_error = makeRatio(num_phi_counts, denom_phi_counts)
            phi_ratio_error = handleUncertainties(phi_ratio, phi_ratio_error)
            axs[1].errorbar(self.phi_bins[:-1], \
                            phi_ratio, yerr=phi_ratio_error, \
                            fmt=".", markersize=20, color=self.c[k], \
                            label=self.legend[k])
            axs[1].set_xlabel(r"$\phi$")
            axs[1].set_xlim(-np.pi, np.pi) 

            num_energy_counts, _ = np.histogram(ak.flatten(energy_pur[0]), bins=self.energy_bins)            
            denom_energy_counts, _ = np.histogram(ak.flatten(energy_pur[1]), bins=self.energy_bins)            
                                                                                            
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
            ax.set_ylabel('Purity')
            ax.grid(True)
            ax.legend()
            if self.annotate != None:
                if "Energy (sim) [GeV]" not in ax.get_xlabel():
                    ax.annotate(text=self.annotate, xy=(ax.get_xlim()[0]*0.9, 0.4), fontsize=20)
                else:
                    ax.annotate(text=self.annotate, xy=(ax.get_xlim()[1]*0.2, 0.4), fontsize=20)

        if (self.save != None):
            fig.savefig(self.save+f"purity_comparison_NoMinAssociatedLcs={No_min_associatedLcs+1}.pdf")

        fig.clf()
 
def count(cp_dict, No_min_associatedLcs):
    num_eta, denom_eta = [], []
    num_phi, denom_phi = [], []
    num_energy, denom_energy = [], []
    
    num_dup_eta, num_dup_phi, num_dup_energy = [], [], []

    denom_eta = cp_dict['caloParticleEta']
    denom_phi = cp_dict['caloParticlePhi']
    denom_energy = cp_dict['caloParticleEnergy']

    score_lowerThan02 = cp_dict['simToRecoAssociation'] < 0.2
    no_associatedLcs = ak.count_nonzero(score_lowerThan02, axis=2)

    # 0 only in this example, can be customized
    no_associatedLcs_moreThan0 = no_associatedLcs > No_min_associatedLcs

    num_eta = cp_dict['caloParticleEta'][no_associatedLcs_moreThan0]
    num_phi = cp_dict['caloParticlePhi'][no_associatedLcs_moreThan0]
    num_energy = cp_dict['caloParticleEnergy'][no_associatedLcs_moreThan0]    

    breakpoint()

    if False:
        for event in range(0, len(cp_dict)):
            event = str(event+1)
            # For each calo particle in each event
            for cp in range(len(cp_dict[event]['caloParticleEnergy'])):
                if len(cp_dict[event]['simToRecoAssociation'][cp]) == 0 : continue
                denom_eta.append(cp_dict[event]['caloParticleEta'][cp])
                denom_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                denom_energy.append(cp_dict[event]['caloParticleEnergy'][cp])
                n_associated_lcs = 0
                for score in cp_dict[event]['simToRecoAssociation'][cp]:
                    if score < 0.2:
                        n_associated_lcs += 1
                if n_associated_lcs > 0:
                    num_eta.append(cp_dict[event]['caloParticleEta'][cp])         
                    num_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                    num_energy.append(cp_dict[event]['caloParticleEnergy'][cp])
                    if n_associated_lcs > 1:
                        num_dup_eta.append(cp_dict[event]['caloParticleEta'][cp])
                        num_dup_phi.append(cp_dict[event]['caloParticlePhi'][cp])
                        num_dup_energy.append(cp_dict[event]['caloParticleEnergy'][cp])

    return [num_eta, denom_eta], [num_phi, denom_phi], [num_energy, denom_energy]
